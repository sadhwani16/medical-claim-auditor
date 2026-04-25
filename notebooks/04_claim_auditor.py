# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 — Agentic Claim Auditor (RAG + LLM + MLflow)
# MAGIC Retrieves relevant government guideline rules via FAISS, audits each claim
# MAGIC using an LLM, and logs 15+ KPIs per claim to Delta Lake + MLflow.

# COMMAND ----------

# MAGIC %pip install mlflow langchain langchain-community sentence-transformers faiss-cpu anthropic -q

# COMMAND ----------

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    TimestampType, IntegerType, BooleanType
)
import json, os, re, time
from datetime import datetime
import anthropic

spark = SparkSession.builder.getOrCreate()

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG           = "hive_metastore"
SCHEMA            = "pmjay_audit"
VECTOR_STORE_PATH = "/dbfs/FileStore/pmjay_audit/vector_store"

# Paste your Anthropic API key here (get one free at console.anthropic.com)
ANTHROPIC_API_KEY = ""   # e.g. "sk-ant-api03-..."

# Fraud risk threshold — claims above this score are auto-FLAGGED
FRAUD_THRESHOLD   = 0.55

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

# MAGIC %md ## Step 2: Audit prompt with 15 KPI fields

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
  "risk_score": <float 0.0–1.0, where 1.0 = definite fraud>,
  "severity_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
  "fraud_category": "NONE" or "OVERSTAY" or "UPCODING" or "PHANTOM_BILLING" or "DUPLICATE" or "UNNECESSARY_PROCEDURE" or "MISSING_DOCS",
  "reason": "<one clear sentence explaining the decision>",
  "cited_rule": "<exact PM-JAY package name or rule number from the guidelines>",
  "package_code": "<HBP package code if identifiable, else empty string>",
  "violations": ["<specific violation 1>", "<specific violation 2>"],
  "claim_amount_flag": <true if claimed amount exceeds guideline package rate, else false>,
  "estimated_excess_amount": <excess amount in INR as a number, 0 if none>,
  "los_violation_days": <number of extra days beyond guideline limit, 0 if none>,
  "procedure_mismatch": <true if procedure does not match the stated diagnosis, else false>,
  "documentation_issues": ["<missing or incomplete doc 1>", "<issue 2>"],
  "pre_auth_required": <true if pre-authorization was required for this procedure>,
  "audit_confidence": <float 0.0–1.0, your confidence in this audit decision>,
  "recommended_action": "<specific next step: approve / manual review / reject / investigate>"
}}"""

# COMMAND ----------

# MAGIC %md ## Step 3: LLM call (Anthropic Claude)

# COMMAND ----------

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def call_llm(claim_text: str, rules: list[dict]) -> dict:
    rules_text = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in rules
    )
    prompt = AUDIT_PROMPT.format(claim_text=claim_text, rules_text=rules_text)

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  LLM error: {e}")

    return {
        "status": "ERROR", "risk_score": 0.0, "severity_level": "LOW",
        "fraud_category": "NONE", "reason": "LLM call failed",
        "cited_rule": "", "package_code": "", "violations": [],
        "claim_amount_flag": False, "estimated_excess_amount": 0,
        "los_violation_days": 0, "procedure_mismatch": False,
        "documentation_issues": [], "pre_auth_required": False,
        "audit_confidence": 0.0, "recommended_action": "Manual review",
    }

# COMMAND ----------

# MAGIC %md ## Step 4: Create audit results table with all KPIs

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.audit_results (
  claim_id              STRING,
  file_name             STRING,

  -- Core decision
  status                STRING,
  risk_score            FLOAT,
  severity_level        STRING,
  fraud_category        STRING,

  -- Rule reference
  reason                STRING,
  cited_rule            STRING,
  package_code          STRING,
  violations            STRING,

  -- Financial KPIs
  claim_amount_flag     BOOLEAN,
  estimated_excess_amount FLOAT,

  -- Clinical KPIs
  los_violation_days    INTEGER,
  procedure_mismatch    BOOLEAN,
  pre_auth_required     BOOLEAN,

  -- Documentation KPIs
  documentation_issues  STRING,

  -- Audit meta
  audit_confidence      FLOAT,
  recommended_action    STRING,
  retrieval_k           INTEGER,
  llm_model             STRING,
  audit_latency_sec     FLOAT,
  audited_at            TIMESTAMP,
  mlflow_run_id         STRING
) USING DELTA
""")

print("Audit results table ready.")

# COMMAND ----------

# MAGIC %md ## Step 5: Run audit pipeline with MLflow tracking

# COMMAND ----------

with mlflow.start_run(run_name=f"audit-{datetime.utcnow().strftime('%Y%m%d-%H%M')}") as run:
    mlflow.log_param("llm_model",      "claude-haiku-4-5-20251001")
    mlflow.log_param("embed_model",    "all-MiniLM-L6-v2")
    mlflow.log_param("retrieval_k",    6)
    mlflow.log_param("fraud_threshold", FRAUD_THRESHOLD)

    # Load only claims not yet audited
    claims_df = (
        spark.table(f"{CATALOG}.{SCHEMA}.claims_translated")
        .join(spark.table(f"{CATALOG}.{SCHEMA}.claims_raw").select("claim_id", "file_name"),
              on="claim_id")
        .join(
            spark.table(f"{CATALOG}.{SCHEMA}.audit_results").select("claim_id"),
            on="claim_id", how="left_anti",
        )
    )

    claims = claims_df.collect()
    print(f"Auditing {len(claims)} claims…\n")

    results  = []
    flagged  = 0
    errors   = 0
    total_excess  = 0.0
    total_los_vio = 0
    start_total   = time.time()

    for row in claims:
        t0    = time.time()
        rules = retrieve_rules(row.translated_text)
        audit = call_llm(row.translated_text, rules)
        latency = round(time.time() - t0, 2)

        if audit["status"] == "FLAGGED":
            flagged += 1
        if audit["status"] == "ERROR":
            errors += 1

        total_excess  += float(audit.get("estimated_excess_amount", 0) or 0)
        total_los_vio += int(audit.get("los_violation_days", 0) or 0)

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
            float(audit.get("estimated_excess_amount", 0) or 0),
            int(audit.get("los_violation_days", 0) or 0),
            bool(audit.get("procedure_mismatch", False)),
            bool(audit.get("pre_auth_required", False)),
            json.dumps(audit.get("documentation_issues", [])),
            float(audit.get("audit_confidence", 0.0)),
            audit.get("recommended_action", "Manual review"),
            6,
            "claude-haiku-4-5-20251001",
            latency,
            datetime.utcnow(),
            run.info.run_id,
        ))

        risk = audit.get("risk_score", 0)
        sev  = audit.get("severity_level", "?")
        cat  = audit.get("fraud_category", "?")
        print(f"  [{audit['status']:8}] {row.file_name:35} risk={risk:.2f} sev={sev:8} cat={cat} ({latency}s)")

    total_time  = round(time.time() - start_total, 2)
    fraud_rate  = flagged / len(claims) if claims else 0
    avg_risk    = sum(r[3] for r in results) / len(results) if results else 0
    avg_conf    = sum(r[16] for r in results) / len(results) if results else 0

    # Log batch-level KPIs to MLflow
    mlflow.log_metric("claims_audited",       len(claims))
    mlflow.log_metric("flagged_count",        flagged)
    mlflow.log_metric("fraud_rate_pct",       round(fraud_rate * 100, 1))
    mlflow.log_metric("error_count",          errors)
    mlflow.log_metric("avg_risk_score",       round(avg_risk, 3))
    mlflow.log_metric("avg_audit_confidence", round(avg_conf, 3))
    mlflow.log_metric("total_estimated_excess_inr", round(total_excess, 2))
    mlflow.log_metric("total_los_violation_days",   total_los_vio)
    mlflow.log_metric("total_audit_seconds",        total_time)

    # Write to Delta
    schema = StructType([
        StructField("claim_id",               StringType()),
        StructField("file_name",              StringType()),
        StructField("status",                 StringType()),
        StructField("risk_score",             FloatType()),
        StructField("severity_level",         StringType()),
        StructField("fraud_category",         StringType()),
        StructField("reason",                 StringType()),
        StructField("cited_rule",             StringType()),
        StructField("package_code",           StringType()),
        StructField("violations",             StringType()),
        StructField("claim_amount_flag",      BooleanType()),
        StructField("estimated_excess_amount",FloatType()),
        StructField("los_violation_days",     IntegerType()),
        StructField("procedure_mismatch",     BooleanType()),
        StructField("pre_auth_required",      BooleanType()),
        StructField("documentation_issues",   StringType()),
        StructField("audit_confidence",       FloatType()),
        StructField("recommended_action",     StringType()),
        StructField("retrieval_k",            IntegerType()),
        StructField("llm_model",              StringType()),
        StructField("audit_latency_sec",      FloatType()),
        StructField("audited_at",             TimestampType()),
        StructField("mlflow_run_id",          StringType()),
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
    print(f"  Total LOS violations:    {total_los_vio} days")
    print(f"  Total time:              {total_time}s")
    print(f"  MLflow run ID:           {run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## Step 6: Review results — all KPIs

# COMMAND ----------

spark.sql(f"""
  SELECT
    r.file_name,
    r.status,
    ROUND(r.risk_score, 2)          AS risk_score,
    r.severity_level,
    r.fraud_category,
    r.claim_amount_flag,
    r.estimated_excess_amount,
    r.los_violation_days,
    r.procedure_mismatch,
    r.pre_auth_required,
    ROUND(r.audit_confidence, 2)    AS confidence,
    r.reason,
    r.cited_rule,
    r.recommended_action
  FROM {CATALOG}.{SCHEMA}.audit_results r
  ORDER BY r.risk_score DESC
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md ## Step 7: Summary statistics

# COMMAND ----------

spark.sql(f"""
  SELECT
    COUNT(*)                                          AS total_claims,
    SUM(CASE WHEN status = 'FLAGGED' THEN 1 ELSE 0 END) AS flagged,
    SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) AS approved,
    ROUND(AVG(risk_score), 3)                         AS avg_risk_score,
    ROUND(AVG(audit_confidence), 3)                   AS avg_confidence,
    SUM(estimated_excess_amount)                      AS total_excess_inr,
    SUM(los_violation_days)                           AS total_los_violation_days,
    SUM(CASE WHEN claim_amount_flag    THEN 1 ELSE 0 END) AS amount_overruns,
    SUM(CASE WHEN procedure_mismatch   THEN 1 ELSE 0 END) AS procedure_mismatches,
    SUM(CASE WHEN pre_auth_required    THEN 1 ELSE 0 END) AS pre_auth_flags
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
