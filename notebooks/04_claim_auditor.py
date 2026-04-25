# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 — Agentic Claim Auditor (RAG + Indian LLM + MLflow)
# MAGIC
# MAGIC LLM priority chain (all Indian / open-source):
# MAGIC 1. **Sarvam AI** — `sarvam-m` model, best for Indian medical context (sarvam.ai)
# MAGIC 2. **Airavata** (AI4Bharat) — Hindi instruction-tuned LLM, free via HuggingFace
# MAGIC 3. **Fallback** — rule-based heuristic audit (no API needed, always works)
# MAGIC
# MAGIC Note on **Aram-1**: Not yet publicly available as an API.
# MAGIC Note on **IndicTrans2**: Used in Notebook 02 for translation, not for auditing.

# COMMAND ----------

# MAGIC %pip install mlflow langchain langchain-community sentence-transformers faiss-cpu requests -q

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    TimestampType, IntegerType, BooleanType
)
import json, os, re, time, requests
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG           = "hive_metastore"
SCHEMA            = "pmjay_audit"
VECTOR_STORE_PATH = "/tmp/pmjay_audit/vector_store"
FRAUD_THRESHOLD   = 0.55

# ── API Keys — fill in what you have ─────────────────────────────────────────
# Option 1: Sarvam AI — get free key at https://sarvam.ai → Dashboard → API Keys
SARVAM_API_KEY = ""   # e.g. "your-sarvam-key"

# Option 2: HuggingFace token — free at https://huggingface.co → Settings → Access Tokens
# Used to call Airavata (AI4Bharat) model via HF Inference API
HF_TOKEN = ""         # e.g. "hf_xxxx"

# Which provider to try first: "sarvam" | "airavata" | "heuristic"
LLM_PROVIDER = "sarvam" if SARVAM_API_KEY else ("airavata" if HF_TOKEN else "heuristic")

mlflow.set_experiment("/pmjay-audit/claim-auditor")

# COMMAND ----------

# MAGIC %md ## Step 1: Load FAISS vector store

# COMMAND ----------

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

index_path = os.path.join(VECTOR_STORE_PATH, "pmjay_faiss_index")
vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
print("Vector store loaded.")


def retrieve_rules(query: str, k: int = 6) -> list[dict]:
    docs = vector_store.similarity_search(query, k=k)
    return [{"text": d.page_content, "source": d.metadata.get("source", "?")} for d in docs]

# COMMAND ----------

# MAGIC %md ## Step 2: Audit prompt

# COMMAND ----------

AUDIT_PROMPT = """\
You are a senior medical claim auditor for Ayushman Bharat PM-JAY (India's government health insurance scheme).

=== CLAIM SUBMITTED ===
{claim_text}

=== RELEVANT PM-JAY GOVERNMENT GUIDELINES ===
{rules_text}

Carefully audit this claim against the guidelines above.
Respond ONLY with valid JSON — no extra text before or after:

{{
  "status": "APPROVED" or "FLAGGED",
  "risk_score": <float 0.0-1.0, where 1.0 = definite fraud>,
  "severity_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "fraud_category": "NONE" or "OVERSTAY" or "UPCODING" or "PHANTOM_BILLING" or "DUPLICATE" or "UNNECESSARY_PROCEDURE" or "MISSING_DOCS",
  "reason": "<one clear sentence explaining the decision>",
  "cited_rule": "<exact PM-JAY package name or rule from the guidelines>",
  "package_code": "<HBP package code if identifiable, else empty string>",
  "violations": ["<specific violation 1>", "<specific violation 2>"],
  "claim_amount_flag": <true if claimed amount exceeds guideline package rate>,
  "estimated_excess_amount": <excess amount in INR as number, 0 if none>,
  "los_violation_days": <extra days beyond guideline limit, 0 if none>,
  "procedure_mismatch": <true if procedure does not match diagnosis>,
  "documentation_issues": ["<missing doc 1>", "<issue 2>"],
  "pre_auth_required": <true if pre-authorization was required>,
  "audit_confidence": <float 0.0-1.0, your confidence in this decision>,
  "recommended_action": "<approve / manual review / reject / investigate>"
}}"""

# COMMAND ----------

# MAGIC %md ## Step 3: LLM backends (Sarvam AI → Airavata → Heuristic)

# COMMAND ----------

def call_sarvam(prompt: str) -> str | None:
    """Sarvam AI — sarvam-m model. Get key at sarvam.ai"""
    if not SARVAM_API_KEY:
        return None
    try:
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {SARVAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sarvam-m",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
                "temperature": 0.0,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        print(f"  Sarvam error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  Sarvam exception: {e}")
    return None


def call_airavata(prompt: str) -> str | None:
    """Airavata (AI4Bharat) via HuggingFace Inference API. Get token at huggingface.co"""
    if not HF_TOKEN:
        return None
    try:
        resp = requests.post(
            "https://api-inference.huggingface.co/models/ai4bharat/airavata",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 800, "temperature": 0.01},
            },
            timeout=90,
        )
        if resp.status_code == 200:
            data = resp.json()
            # HF returns list of generated_text
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
        print(f"  Airavata error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"  Airavata exception: {e}")
    return None


def heuristic_audit(claim_text: str, rules: list[dict]) -> dict:
    """
    Rule-based fallback when no LLM API is available.
    Detects common fraud patterns by keyword matching.
    """
    text_lower = claim_text.lower()
    violations = []
    risk = 0.0
    category = "NONE"

    overstay_keywords = ["extended stay", "additional days", "prolonged", "overstay"]
    upcoding_keywords = ["upgraded", "higher package", "reclassified", "premium procedure"]
    phantom_keywords  = ["not performed", "no record", "absent", "never admitted"]
    doc_keywords      = ["no discharge summary", "missing reports", "no prescription"]

    if any(k in text_lower for k in overstay_keywords):
        violations.append("Possible overstay detected")
        risk += 0.35
        category = "OVERSTAY"
    if any(k in text_lower for k in upcoding_keywords):
        violations.append("Possible upcoding detected")
        risk += 0.40
        category = "UPCODING"
    if any(k in text_lower for k in phantom_keywords):
        violations.append("Possible phantom billing")
        risk += 0.60
        category = "PHANTOM_BILLING"
    if any(k in text_lower for k in doc_keywords):
        violations.append("Documentation incomplete")
        risk += 0.20
        category = category if category != "NONE" else "MISSING_DOCS"

    risk = min(risk, 1.0)
    status = "FLAGGED" if risk >= FRAUD_THRESHOLD else "APPROVED"
    severity = "CRITICAL" if risk > 0.8 else "HIGH" if risk > 0.6 else "MEDIUM" if risk > 0.35 else "LOW"

    return {
        "status": status, "risk_score": risk, "severity_level": severity,
        "fraud_category": category, "reason": f"Heuristic audit: {', '.join(violations) or 'No issues found'}",
        "cited_rule": "Keyword-based rule matching", "package_code": "",
        "violations": violations, "claim_amount_flag": False,
        "estimated_excess_amount": 0, "los_violation_days": 0,
        "procedure_mismatch": False, "documentation_issues": [],
        "pre_auth_required": False, "audit_confidence": 0.5,
        "recommended_action": "Manual review" if status == "FLAGGED" else "Approve",
    }


def parse_llm_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def call_llm(claim_text: str, rules: list[dict]) -> tuple[dict, str]:
    """Returns (audit_dict, method_used)"""
    rules_text = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in rules
    )
    prompt = AUDIT_PROMPT.format(claim_text=claim_text, rules_text=rules_text)

    # Try Sarvam AI first
    raw = call_sarvam(prompt)
    if raw:
        result = parse_llm_json(raw)
        if result:
            return result, "sarvam-m"

    # Try Airavata (AI4Bharat)
    raw = call_airavata(prompt)
    if raw:
        result = parse_llm_json(raw)
        if result:
            return result, "airavata"

    # Heuristic fallback — always works
    return heuristic_audit(claim_text, rules), "heuristic"

# COMMAND ----------

# MAGIC %md ## Step 4: Create audit results table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.audit_results (
  claim_id                STRING,
  file_name               STRING,
  status                  STRING,
  risk_score              FLOAT,
  severity_level          STRING,
  fraud_category          STRING,
  reason                  STRING,
  cited_rule              STRING,
  package_code            STRING,
  violations              STRING,
  claim_amount_flag       BOOLEAN,
  estimated_excess_amount FLOAT,
  los_violation_days      INTEGER,
  procedure_mismatch      BOOLEAN,
  pre_auth_required       BOOLEAN,
  documentation_issues    STRING,
  audit_confidence        FLOAT,
  recommended_action      STRING,
  retrieval_k             INTEGER,
  llm_model               STRING,
  audit_latency_sec       FLOAT,
  audited_at              TIMESTAMP,
  mlflow_run_id           STRING
) USING DELTA
""")
print("Audit results table ready.")

# COMMAND ----------

# MAGIC %md ## Step 5: Run audit pipeline with MLflow tracking

# COMMAND ----------

print(f"LLM provider: {LLM_PROVIDER}")
print(f"Sarvam key set: {bool(SARVAM_API_KEY)}")
print(f"HuggingFace token set: {bool(HF_TOKEN)}")

with mlflow.start_run(run_name=f"audit-{datetime.utcnow().strftime('%Y%m%d-%H%M')}") as run:
    mlflow.log_param("llm_provider",    LLM_PROVIDER)
    mlflow.log_param("embed_model",     "all-MiniLM-L6-v2")
    mlflow.log_param("retrieval_k",     6)
    mlflow.log_param("fraud_threshold", FRAUD_THRESHOLD)

    claims_df = (
        spark.table(f"{CATALOG}.{SCHEMA}.claims_translated")
        .join(spark.table(f"{CATALOG}.{SCHEMA}.claims_raw").select("claim_id", "file_name"), on="claim_id")
        .join(spark.table(f"{CATALOG}.{SCHEMA}.audit_results").select("claim_id"), on="claim_id", how="left_anti")
    )

    claims      = claims_df.collect()
    print(f"\nAuditing {len(claims)} claims…\n")

    results      = []
    flagged      = 0
    errors       = 0
    total_excess = 0.0
    total_los    = 0
    methods_used = {}
    start_total  = time.time()

    for row in claims:
        t0          = time.time()
        rules       = retrieve_rules(row.translated_text)
        audit, method = call_llm(row.translated_text, rules)
        latency     = round(time.time() - t0, 2)

        methods_used[method] = methods_used.get(method, 0) + 1

        if audit["status"] == "FLAGGED":
            flagged += 1
        if audit["status"] == "ERROR":
            errors += 1

        total_excess += float(audit.get("estimated_excess_amount") or 0)
        total_los    += int(audit.get("los_violation_days") or 0)

        results.append((
            row.claim_id,
            row.file_name,
            audit["status"],
            float(audit.get("risk_score", 0.0)),
            audit.get("severity_level", "LOW"),
            audit.get("fraud_category", "NONE"),
            audit.get("reason", ""),
            audit.get("cited_rule", ""),
            audit.get("package_code", ""),
            json.dumps(audit.get("violations", [])),
            bool(audit.get("claim_amount_flag", False)),
            float(audit.get("estimated_excess_amount") or 0),
            int(audit.get("los_violation_days") or 0),
            bool(audit.get("procedure_mismatch", False)),
            bool(audit.get("pre_auth_required", False)),
            json.dumps(audit.get("documentation_issues", [])),
            float(audit.get("audit_confidence", 0.5)),
            audit.get("recommended_action", "Manual review"),
            6,
            method,
            latency,
            datetime.utcnow(),
            run.info.run_id,
        ))

        risk = audit.get("risk_score", 0)
        sev  = audit.get("severity_level", "?")
        cat  = audit.get("fraud_category", "?")
        print(f"  [{audit['status']:8}] {row.file_name:35} risk={risk:.2f} sev={sev:8} cat={cat} via={method} ({latency}s)")

    total_time = round(time.time() - start_total, 2)
    fraud_rate = flagged / len(claims) if claims else 0
    avg_risk   = sum(r[3] for r in results) / len(results) if results else 0
    avg_conf   = sum(r[16] for r in results) / len(results) if results else 0

    mlflow.log_metric("claims_audited",            len(claims))
    mlflow.log_metric("flagged_count",             flagged)
    mlflow.log_metric("fraud_rate_pct",            round(fraud_rate * 100, 1))
    mlflow.log_metric("error_count",               errors)
    mlflow.log_metric("avg_risk_score",            round(avg_risk, 3))
    mlflow.log_metric("avg_audit_confidence",      round(avg_conf, 3))
    mlflow.log_metric("total_estimated_excess_inr",round(total_excess, 2))
    mlflow.log_metric("total_los_violation_days",  total_los)
    mlflow.log_metric("total_audit_seconds",       total_time)
    mlflow.log_param("methods_used", json.dumps(methods_used))

    schema = StructType([
        StructField("claim_id",                StringType()),
        StructField("file_name",               StringType()),
        StructField("status",                  StringType()),
        StructField("risk_score",              FloatType()),
        StructField("severity_level",          StringType()),
        StructField("fraud_category",          StringType()),
        StructField("reason",                  StringType()),
        StructField("cited_rule",              StringType()),
        StructField("package_code",            StringType()),
        StructField("violations",              StringType()),
        StructField("claim_amount_flag",       BooleanType()),
        StructField("estimated_excess_amount", FloatType()),
        StructField("los_violation_days",      IntegerType()),
        StructField("procedure_mismatch",      BooleanType()),
        StructField("pre_auth_required",       BooleanType()),
        StructField("documentation_issues",    StringType()),
        StructField("audit_confidence",        FloatType()),
        StructField("recommended_action",      StringType()),
        StructField("retrieval_k",             IntegerType()),
        StructField("llm_model",               StringType()),
        StructField("audit_latency_sec",       FloatType()),
        StructField("audited_at",              TimestampType()),
        StructField("mlflow_run_id",           StringType()),
    ])

    spark.createDataFrame(results, schema).write.format("delta").mode("append").saveAsTable(
        f"{CATALOG}.{SCHEMA}.audit_results"
    )

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"  Claims audited:          {len(claims)}")
    print(f"  Flagged:                 {flagged} ({fraud_rate:.0%})")
    print(f"  Avg risk score:          {avg_risk:.3f}")
    print(f"  Avg audit confidence:    {avg_conf:.3f}")
    print(f"  Total estimated excess:  INR {total_excess:,.0f}")
    print(f"  Total LOS violations:    {total_los} days")
    print(f"  LLM methods used:        {methods_used}")
    print(f"  Total time:              {total_time}s")
    print(f"  MLflow run ID:           {run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## Step 6: Full results with all KPIs

# COMMAND ----------

spark.sql(f"""
  SELECT
    file_name,
    status,
    ROUND(risk_score, 2)           AS risk,
    severity_level                 AS severity,
    fraud_category,
    claim_amount_flag              AS amt_flag,
    estimated_excess_amount        AS excess_inr,
    los_violation_days             AS los_days,
    procedure_mismatch             AS proc_mismatch,
    pre_auth_required              AS pre_auth,
    ROUND(audit_confidence, 2)     AS confidence,
    llm_model,
    reason,
    recommended_action
  FROM {CATALOG}.{SCHEMA}.audit_results
  ORDER BY risk_score DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Step 7: Summary statistics

# COMMAND ----------

spark.sql(f"""
  SELECT
    COUNT(*)                                              AS total_claims,
    SUM(CASE WHEN status='FLAGGED'  THEN 1 ELSE 0 END)   AS flagged,
    SUM(CASE WHEN status='APPROVED' THEN 1 ELSE 0 END)   AS approved,
    ROUND(AVG(risk_score), 3)                             AS avg_risk_score,
    ROUND(AVG(audit_confidence), 3)                       AS avg_confidence,
    ROUND(SUM(estimated_excess_amount), 0)                AS total_excess_inr,
    SUM(los_violation_days)                               AS total_los_vio_days,
    SUM(CASE WHEN claim_amount_flag  THEN 1 ELSE 0 END)   AS amount_overruns,
    SUM(CASE WHEN procedure_mismatch THEN 1 ELSE 0 END)   AS proc_mismatches,
    SUM(CASE WHEN pre_auth_required  THEN 1 ELSE 0 END)   AS pre_auth_flags
  FROM {CATALOG}.{SCHEMA}.audit_results
""").show()

# COMMAND ----------

spark.sql(f"""
  SELECT fraud_category, severity_level, COUNT(*) AS count,
         ROUND(AVG(risk_score), 3) AS avg_risk
  FROM {CATALOG}.{SCHEMA}.audit_results
  GROUP BY fraud_category, severity_level
  ORDER BY count DESC
""").show()
