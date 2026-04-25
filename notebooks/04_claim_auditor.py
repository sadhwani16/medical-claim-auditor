# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 04 — Agentic Claim Auditor (RAG + LLM + MLflow)
# MAGIC For every translated claim, retrieve rules via FAISS, prompt LLaMA 3, log results to Delta & MLflow.

# COMMAND ----------

# MAGIC %pip install mlflow langchain langchain-community sentence-transformers faiss-cpu openai -q

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/your-repo/medical-claim-auditor")

import mlflow
import mlflow.pyfunc
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, udf
from pyspark.sql.types import StringType, StructType, StructField, FloatType, TimestampType, ArrayType
import json, os, requests, re, time
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

CATALOG           = "hive_metastore"
SCHEMA            = "pmjay_audit"
VECTOR_STORE_PATH = "/dbfs/pmjay_audit/vector_store"
DATABRICKS_HOST   = dbutils.secrets.get(scope="pmjay", key="databricks_host")   # noqa
DATABRICKS_TOKEN  = dbutils.secrets.get(scope="pmjay", key="databricks_token")  # noqa
LLM_MODEL         = "databricks-meta-llama-3-1-70b-instruct"

mlflow.set_experiment("/pmjay-audit/claim-auditor")

# COMMAND ----------

# MAGIC %md ## Step 1: Load vector store and define RAG retriever

# COMMAND ----------

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = FAISS.load_local(
    os.path.join(VECTOR_STORE_PATH, "pmjay_faiss_index"),
    embeddings,
    allow_dangerous_deserialization=True,
)
print("Vector store loaded.")


def retrieve_rules(query: str, k: int = 5) -> list[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [d.page_content for d in docs]

# COMMAND ----------

# MAGIC %md ## Step 2: Define the LLM call (Databricks Model Serving)

# COMMAND ----------

AUDIT_PROMPT = """\
You are a senior medical claim auditor for Ayushman Bharat PM-JAY.

=== CLAIM ===
{claim_text}

=== RELEVANT PM-JAY RULES ===
{rules_text}

Audit the claim. Respond ONLY with valid JSON:
{{
  "status": "APPROVED" or "FLAGGED",
  "risk_score": <0.0-1.0>,
  "reason": "<one sentence>",
  "cited_rule": "<specific PM-JAY package/rule>",
  "violations": ["<issue1>", "<issue2>"],
  "recommended_action": "<next step>"
}}"""


def call_llm(claim_text: str, rules: list[str]) -> dict:
    rules_text = "\n\n---\n\n".join(rules)
    prompt = AUDIT_PROMPT.format(claim_text=claim_text, rules_text=rules_text)

    resp = requests.post(
        f"{DATABRICKS_HOST}/serving-endpoints/{LLM_MODEL}/invocations",
        headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "status": "ERROR", "risk_score": 0.0, "reason": raw[:200],
        "cited_rule": "", "violations": [], "recommended_action": "Manual review",
    }

# COMMAND ----------

# MAGIC %md ## Step 3: Create audit results table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.audit_results (
  claim_id            STRING,
  status              STRING,
  risk_score          FLOAT,
  reason              STRING,
  cited_rule          STRING,
  violations          STRING,
  recommended_action  STRING,
  audited_at          TIMESTAMP,
  llm_model           STRING,
  mlflow_run_id       STRING
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md ## Step 4: Run audit pipeline with MLflow tracking

# COMMAND ----------

with mlflow.start_run(run_name=f"audit-batch-{datetime.utcnow().strftime('%Y%m%d-%H%M')}") as run:
    mlflow.log_param("llm_model", LLM_MODEL)
    mlflow.log_param("embed_model", "all-MiniLM-L6-v2")
    mlflow.log_param("retrieval_k", 5)
    mlflow.log_param("prompt_version", "v1")

    claims_df = (
        spark.table(f"{CATALOG}.{SCHEMA}.claims_translated")
        .join(
            spark.table(f"{CATALOG}.{SCHEMA}.audit_results").select("claim_id"),
            on="claim_id", how="left_anti",
        )
    )

    claims = claims_df.collect()
    print(f"Auditing {len(claims)} claims…")

    results, flagged = [], 0
    start_total = time.time()

    for row in claims:
        t0 = time.time()
        rules = retrieve_rules(row.translated_text)
        audit = call_llm(row.translated_text, rules)
        latency = round(time.time() - t0, 2)

        if audit["status"] == "FLAGGED":
            flagged += 1

        results.append((
            row.claim_id,
            audit["status"],
            float(audit.get("risk_score", 0.0)),
            audit.get("reason", ""),
            audit.get("cited_rule", ""),
            json.dumps(audit.get("violations", [])),
            audit.get("recommended_action", ""),
            datetime.utcnow(),
            LLM_MODEL,
            run.info.run_id,
        ))
        print(f"  [{audit['status']}] {row.claim_id[:8]}… risk={audit.get('risk_score', 0):.2f} ({latency}s)")

    total_time = round(time.time() - start_total, 2)
    fraud_rate  = flagged / len(claims) if claims else 0

    mlflow.log_metric("claims_audited", len(claims))
    mlflow.log_metric("fraud_rate", fraud_rate)
    mlflow.log_metric("flagged_count", flagged)
    mlflow.log_metric("total_audit_seconds", total_time)

    schema = StructType([
        StructField("claim_id",           StringType()),
        StructField("status",             StringType()),
        StructField("risk_score",         FloatType()),
        StructField("reason",             StringType()),
        StructField("cited_rule",         StringType()),
        StructField("violations",         StringType()),
        StructField("recommended_action", StringType()),
        StructField("audited_at",         TimestampType()),
        StructField("llm_model",          StringType()),
        StructField("mlflow_run_id",      StringType()),
    ])

    spark.createDataFrame(results, schema).write.format("delta").mode("append").saveAsTable(
        f"{CATALOG}.{SCHEMA}.audit_results"
    )

    print(f"\n✅ Batch complete: {len(claims)} claims, {flagged} flagged ({fraud_rate:.0%}), {total_time}s")
    print(f"MLflow run: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## Step 5: Review results

# COMMAND ----------

spark.sql(f"""
SELECT c.file_name, r.status, r.risk_score, r.reason, r.cited_rule
FROM {CATALOG}.{SCHEMA}.audit_results r
JOIN {CATALOG}.{SCHEMA}.claims_raw c ON r.claim_id = c.claim_id
ORDER BY r.risk_score DESC
""").show(truncate=False)
