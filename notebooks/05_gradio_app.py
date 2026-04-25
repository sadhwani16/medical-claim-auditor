# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05 — AyushAudit AI — Interactive App (Gradio)
# MAGIC Runs a full web UI inside Databricks. Generates a public shareable URL.
# MAGIC
# MAGIC **Run all cells — the app link appears at the bottom of the last cell.**

# COMMAND ----------

# MAGIC %pip install -q --upgrade typing_extensions
# MAGIC %pip install -q gradio langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu requests deep-translator langdetect pdfplumber PyMuPDF
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os, json, re, time, requests
from datetime import datetime
from pyspark.sql import SparkSession
import gradio as gr

spark = SparkSession.builder.getOrCreate()

# ── Config ─────────────────────────────────────────────────────────────────────
CATALOG           = "workspace"
SCHEMA            = "pmjay_audit"
VOLUME            = "files"
VECTOR_STORE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/vector_store"
FRAUD_THRESHOLD   = 0.55

# Paste your Sarvam API key here
SARVAM_API_KEY    = ""   # sk_...

# COMMAND ----------

# MAGIC %md ## Step 1 — Load FAISS vector index

# COMMAND ----------

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Loading embedding model…")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

index_path = os.path.join(VECTOR_STORE_PATH, "pmjay_faiss_index")
vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
print("Vector store loaded.")

def retrieve_rules(query: str, k: int = 6) -> list[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [f"[{d.metadata.get('source','?')}] {d.page_content}" for d in docs]

# COMMAND ----------

# MAGIC %md ## Step 2 — Translation helper

# COMMAND ----------

from langdetect import detect, LangDetectException

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali",
    "gu": "Gujarati", "mr": "Marathi", "pa": "Punjabi",
}

SARVAM_LANG_MAP = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN",
    "gu": "gu-IN", "mr": "mr-IN", "pa": "pa-IN",
}

def translate_claim(text: str) -> tuple[str, str]:
    """Returns (translated_text, detected_language_name)"""
    try:
        lang = detect(text[:500])
    except LangDetectException:
        lang = "en"

    lang_name = LANG_NAMES.get(lang, lang)
    if lang == "en":
        return text, lang_name

    # Try Sarvam AI
    sarvam_code = SARVAM_LANG_MAP.get(lang)
    if sarvam_code and SARVAM_API_KEY:
        try:
            parts = []
            for chunk in [text[i:i+900] for i in range(0, len(text), 900)]:
                resp = requests.post(
                    "https://api.sarvam.ai/translate",
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    json={"input": chunk, "source_language_code": sarvam_code,
                          "target_language_code": "en-IN", "mode": "formal"},
                    timeout=20,
                )
                parts.append(resp.json().get("translated_text", chunk) if resp.status_code == 200 else chunk)
            return " ".join(parts), lang_name
        except Exception:
            pass

    # Fallback: Google Translate
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=lang, target="en").translate(text[:4999]), lang_name
    except Exception:
        return text, lang_name

# COMMAND ----------

# MAGIC %md ## Step 3 — Audit engine

# COMMAND ----------

AUDIT_PROMPT = """\
You are a senior medical claim auditor for Ayushman Bharat PM-JAY.

=== CLAIM ===
{claim_text}

=== RELEVANT PM-JAY GUIDELINES ===
{rules_text}

Audit this claim. Respond ONLY with valid JSON:
{{
  "status": "APPROVED" or "FLAGGED",
  "risk_score": <float 0.0-1.0>,
  "severity_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "fraud_category": "NONE" or "OVERSTAY" or "UPCODING" or "PHANTOM_BILLING" or "DUPLICATE" or "UNNECESSARY_PROCEDURE" or "MISSING_DOCS",
  "reason": "<one sentence>",
  "cited_rule": "<exact PM-JAY rule>",
  "package_code": "<HBP code if known>",
  "violations": ["<issue 1>", "<issue 2>"],
  "claim_amount_flag": <true/false>,
  "estimated_excess_amount": <INR number or 0>,
  "los_violation_days": <number or 0>,
  "procedure_mismatch": <true/false>,
  "documentation_issues": ["<issue>"],
  "pre_auth_required": <true/false>,
  "audit_confidence": <float 0.0-1.0>,
  "recommended_action": "<next step>"
}}"""


def call_sarvam_llm(prompt: str) -> str | None:
    if not SARVAM_API_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"},
            json={"model": "sarvam-m", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 800, "temperature": 0.0},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Sarvam LLM error: {e}")
    return None


def heuristic_audit(claim_text: str) -> dict:
    text  = claim_text.lower()
    risk  = 0.1
    violations = []
    category   = "NONE"

    los_match = re.search(r"(\d+)\s*day", text)
    if los_match:
        days = int(los_match.group(1))
        if days > 5 and ("cataract" in text or "phaco" in text):
            violations.append(f"LOS {days} days exceeds PM-JAY max 1 day for cataract")
            risk = 0.92; category = "OVERSTAY"
        elif days > 3 and "appendic" in text:
            violations.append(f"LOS {days} days exceeds PM-JAY max 3 days for appendectomy")
            risk = 0.85; category = "OVERSTAY"

    if re.search(r"(upcod|higher package|reclassif)", text):
        violations.append("Possible upcoding detected"); risk = max(risk, 0.75); category = "UPCODING"
    if re.search(r"(phantom|ghost|not performed|no record)", text):
        violations.append("Possible phantom billing"); risk = max(risk, 0.90); category = "PHANTOM_BILLING"

    status   = "FLAGGED" if risk >= FRAUD_THRESHOLD else "APPROVED"
    severity = "CRITICAL" if risk > 0.8 else "HIGH" if risk > 0.6 else "MEDIUM" if risk > 0.35 else "LOW"
    return {
        "status": status, "risk_score": risk, "severity_level": severity,
        "fraud_category": category, "reason": violations[0] if violations else "Claim complies with PM-JAY guidelines",
        "cited_rule": "Heuristic rule-based check", "package_code": "",
        "violations": violations, "claim_amount_flag": False, "estimated_excess_amount": 0,
        "los_violation_days": 0, "procedure_mismatch": False, "documentation_issues": [],
        "pre_auth_required": False, "audit_confidence": 0.5,
        "recommended_action": "Escalate for manual review" if status == "FLAGGED" else "Approve for reimbursement",
    }


def run_audit_pipeline(claim_text: str) -> dict:
    t0 = time.time()
    translated, lang_name = translate_claim(claim_text)
    rules      = retrieve_rules(translated)
    rules_text = "\n\n---\n\n".join(rules)
    prompt     = AUDIT_PROMPT.format(claim_text=translated, rules_text=rules_text)

    raw = call_sarvam_llm(prompt)
    if raw:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                result["_latency"] = round(time.time() - t0, 2)
                result["_language"] = lang_name
                result["_llm"] = "sarvam-m"
                result["_rules"] = rules
                return result
            except json.JSONDecodeError:
                pass

    result = heuristic_audit(translated)
    result["_latency"] = round(time.time() - t0, 2)
    result["_language"] = lang_name
    result["_llm"] = "heuristic"
    result["_rules"] = rules
    return result

# COMMAND ----------

# MAGIC %md ## Step 4 — PDF text extraction

# COMMAND ----------

import pdfplumber, fitz

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        import io
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass
    if len(text.strip()) < 50:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
        except Exception:
            pass
    return text.strip()

# COMMAND ----------

# MAGIC %md ## Step 5 — Load batch results from Delta

# COMMAND ----------

def load_batch_results() -> tuple[list, list]:
    """Load previous audit results from Delta table."""
    try:
        df = spark.table(f"{CATALOG}.{SCHEMA}.audit_results").toPandas()
        headers = ["File", "Status", "Risk", "Severity", "Category",
                   "Excess (INR)", "LOS Vio. Days", "Confidence", "Reason"]
        rows = []
        for _, r in df.iterrows():
            rows.append([
                r.get("file_name", ""),
                r.get("status", ""),
                f"{r.get('risk_score', 0):.2f}",
                r.get("severity_level", ""),
                r.get("fraud_category", ""),
                f"₹{r.get('estimated_excess_amount', 0):,.0f}",
                str(r.get("los_violation_days", 0)),
                f"{r.get('audit_confidence', 0):.2f}",
                r.get("reason", "")[:80],
            ])
        return headers, rows
    except Exception as e:
        return ["Error"], [[str(e)]]

# COMMAND ----------

# MAGIC %md ## Step 6 — Build Gradio UI

# COMMAND ----------

def audit_from_text(claim_text: str):
    if not claim_text.strip():
        return ("", "", "", "", "", "", [], "", "")
    result = run_audit_pipeline(claim_text)
    return format_result(result)


def audit_from_file(file_obj):
    if file_obj is None:
        return ("", "", "", "", "", "", [], "", "")
    with open(file_obj.name, "rb") as f:
        raw = f.read()
    if file_obj.name.endswith(".pdf"):
        text = extract_text_from_pdf(raw)
    else:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        return ("No text could be extracted from the file.",) + ("",) * 8
    result = run_audit_pipeline(text)
    return format_result(result)


def format_result(result: dict):
    status    = result.get("status", "ERROR")
    risk      = result.get("risk_score", 0)
    severity  = result.get("severity_level", "")
    category  = result.get("fraud_category", "NONE")
    reason    = result.get("reason", "")
    action    = result.get("recommended_action", "")
    violations= result.get("violations", [])
    lang      = result.get("_language", "English")
    latency   = result.get("_latency", 0)

    verdict = f"{'🚨 FLAGGED FOR FRAUD' if status == 'FLAGGED' else '✅ APPROVED'}"
    risk_str = f"{risk:.0%}"
    meta     = f"Language: {lang} | Model: {result.get('_llm','?')} | Latency: {latency}s"
    kpis     = (
        f"Severity: {severity} | Category: {category}\n"
        f"Excess Amount: ₹{result.get('estimated_excess_amount', 0):,.0f} | "
        f"LOS Violation: {result.get('los_violation_days', 0)} days\n"
        f"Procedure Mismatch: {result.get('procedure_mismatch', False)} | "
        f"Pre-Auth Required: {result.get('pre_auth_required', False)}\n"
        f"Confidence: {result.get('audit_confidence', 0):.0%} | "
        f"Package: {result.get('package_code', 'N/A')}"
    )
    vio_str  = "\n".join(f"⛔ {v}" for v in violations) if violations else "None"
    rules_str = "\n\n".join(result.get("_rules", []))

    return verdict, risk_str, reason, kpis, vio_str, action, rules_str, meta, json.dumps(result, indent=2)


SAMPLE_CLAIMS = {
    "Cataract Fraud (7-day overstay)": """DISCHARGE SUMMARY
Hospital: Sunrise Superspeciality Hospital, Lucknow | PM-JAY ID: UP-HOS-09912
Patient: Ramesh Chandra Gupta, 52M | Ayushman Card: 4421-9901-XXXX
Admission: 05-Mar-2024 | Discharge: 12-Mar-2024
Diagnosis: Senile Cataract Left Eye (ICD-10: H25.9)
Procedure: Phacoemulsification with IOL Implantation
Length of Stay: 7 days — no documented clinical justification
PM-JAY Package: H/03 | Claimed: Rs.10,000 + Rs.4,900 bed charges (7 x Rs.700)
Total Claimed: Rs.14,900 | Package Rate: Rs.10,000""",

    "Cataract Compliant (1-day)": """DISCHARGE SUMMARY
Hospital: Shri Ram Medical Centre, Jaipur | PM-JAY ID: RAJ-HOS-04271
Patient: Sunita Devi, 58F | Ayushman Card: 7712-3849-XXXX
Admission: 10-Feb-2024 | Discharge: 10-Feb-2024
Diagnosis: Senile Cataract Right Eye (ICD-10: H25.9)
Procedure: Phacoemulsification with IOL Implantation
Length of Stay: 1 day (day-care)
PM-JAY Package: H/03 | Claimed: Rs.9,500 | Package Rate: Rs.10,000""",

    "CABG Upcoding Fraud": """DISCHARGE SUMMARY
Hospital: Apollo Reach Hospital, Karimnagar | PM-JAY ID: TEL-HOS-00231
Patient: Lakshmi Narayana, 61M | Ayushman Card: 5531-2210-XXXX
Admission: 01-Apr-2024 | Discharge: 03-Apr-2024
Diagnosis: Stable Angina (ICD-10: I20.8)
Procedure Claimed: CABG — PM-JAY Package C/07
Actual Procedure: Coronary Angiography only
Claimed: Rs.1,50,000 (CABG rate) | Actual Procedure Rate: Rs.8,000""",
}


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    title="AyushAudit AI — PM-JAY Claim Auditor",
) as demo:

    gr.Markdown("""
    # 🏥 AyushAudit AI
    ### Agentic Medical Claim Auditor for Ayushman Bharat PM-JAY
    Powered by RAG on official government guidelines + Sarvam AI
    """)

    with gr.Tabs():

        # ── Tab 1: Single Claim Audit ─────────────────────────────────────────
        with gr.Tab("🔍 Audit a Claim"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    claim_input = gr.Textbox(
                        label="Paste claim text",
                        lines=12,
                        placeholder="Paste discharge summary or claim text here…",
                    )
                    file_input = gr.File(
                        label="Or upload PDF / TXT",
                        file_types=[".pdf", ".txt"],
                    )

                    with gr.Row():
                        sample_dd = gr.Dropdown(
                            label="Load sample claim",
                            choices=list(SAMPLE_CLAIMS.keys()),
                        )
                        load_btn = gr.Button("Load", size="sm")

                    audit_btn = gr.Button("🚀 Run Audit", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Result")
                    verdict_out  = gr.Textbox(label="Verdict", lines=1)
                    risk_out     = gr.Textbox(label="Risk Score", lines=1)
                    reason_out   = gr.Textbox(label="Reason", lines=2)
                    kpis_out     = gr.Textbox(label="KPIs", lines=5)
                    vio_out      = gr.Textbox(label="Violations", lines=3)
                    action_out   = gr.Textbox(label="Recommended Action", lines=1)
                    meta_out     = gr.Textbox(label="Audit Meta", lines=1)

            with gr.Accordion("📚 Retrieved PM-JAY Guidelines", open=False):
                rules_out = gr.Textbox(label="Relevant rules used for this audit", lines=10)
            with gr.Accordion("📄 Full Audit JSON", open=False):
                json_out = gr.JSON(label="Raw audit output")

            outputs = [verdict_out, risk_out, reason_out, kpis_out,
                       vio_out, action_out, rules_out, meta_out, json_out]

            load_btn.click(
                fn=lambda choice: SAMPLE_CLAIMS.get(choice, ""),
                inputs=sample_dd, outputs=claim_input,
            )
            audit_btn.click(fn=audit_from_text, inputs=claim_input, outputs=outputs)
            file_input.change(fn=audit_from_file, inputs=file_input, outputs=outputs)

        # ── Tab 2: Batch Results from Delta ──────────────────────────────────
        with gr.Tab("📋 Batch Results"):
            gr.Markdown("### Previous batch audit results from Delta Lake")
            refresh_btn    = gr.Button("🔄 Load Results from Delta", variant="secondary")
            batch_table    = gr.Dataframe(
                headers=["File", "Status", "Risk", "Severity", "Category",
                         "Excess (INR)", "LOS Vio. Days", "Confidence", "Reason"],
                label="Audit Results",
                wrap=True,
            )

            def refresh_batch():
                headers, rows = load_batch_results()
                return rows

            refresh_btn.click(fn=refresh_batch, outputs=batch_table)

        # ── Tab 3: Analytics ──────────────────────────────────────────────────
        with gr.Tab("📊 Analytics"):
            gr.Markdown("### Batch audit analytics from Delta Lake")
            analytics_btn = gr.Button("📊 Load Analytics", variant="secondary")
            stats_out     = gr.Textbox(label="Summary Statistics", lines=12)

            def load_analytics():
                try:
                    df = spark.sql(f"""
                        SELECT
                            COUNT(*)                                              AS total_claims,
                            SUM(CASE WHEN status='FLAGGED'  THEN 1 ELSE 0 END)   AS flagged,
                            SUM(CASE WHEN status='APPROVED' THEN 1 ELSE 0 END)   AS approved,
                            ROUND(AVG(risk_score), 3)                             AS avg_risk_score,
                            ROUND(AVG(audit_confidence), 3)                       AS avg_confidence,
                            ROUND(SUM(estimated_excess_amount), 0)                AS total_excess_inr,
                            SUM(los_violation_days)                               AS total_los_vio_days,
                            SUM(CASE WHEN claim_amount_flag  THEN 1 ELSE 0 END)   AS amount_overruns,
                            SUM(CASE WHEN procedure_mismatch THEN 1 ELSE 0 END)   AS proc_mismatches
                        FROM {CATALOG}.{SCHEMA}.audit_results
                    """).collect()[0]

                    cats = spark.sql(f"""
                        SELECT fraud_category, COUNT(*) as count
                        FROM {CATALOG}.{SCHEMA}.audit_results
                        GROUP BY fraud_category ORDER BY count DESC
                    """).collect()

                    lines = [
                        f"Total Claims Audited : {df.total_claims}",
                        f"Flagged              : {df.flagged} ({df.flagged/max(df.total_claims,1):.0%})",
                        f"Approved             : {df.approved}",
                        f"Avg Risk Score       : {df.avg_risk_score}",
                        f"Avg Confidence       : {df.avg_confidence}",
                        f"Total Excess (INR)   : ₹{df.total_excess_inr:,.0f}",
                        f"LOS Violation Days   : {df.total_los_vio_days}",
                        f"Amount Overruns      : {df.amount_overruns}",
                        f"Procedure Mismatches : {df.proc_mismatches}",
                        "",
                        "Fraud Category Breakdown:",
                    ] + [f"  {r.fraud_category:25} {r.count}" for r in cats]
                    return "\n".join(lines)
                except Exception as e:
                    return f"Error loading analytics: {e}"

            analytics_btn.click(fn=load_analytics, outputs=stats_out)

# COMMAND ----------

# MAGIC %md ## Step 7 — Launch the app

# COMMAND ----------

demo.launch(share=True)
