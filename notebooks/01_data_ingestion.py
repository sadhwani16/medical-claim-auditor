# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01 — Data Ingestion into Delta Lake
# MAGIC Upload claim PDFs and PM-JAY rulebook PDFs to DBFS, extract text, store in Delta tables.

# COMMAND ----------

# MAGIC %pip install pdfplumber PyMuPDF langdetect -q

# COMMAND ----------

import pdfplumber
import fitz
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType
from pyspark.sql.functions import lit, current_timestamp
import os, io, hashlib, uuid
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# ── Config ────────────────────────────────────
CLAIMS_DBFS_PATH = "/dbfs/pmjay_audit/claims"
RULES_DBFS_PATH  = "/dbfs/pmjay_audit/rules"
CATALOG          = "hive_metastore"
SCHEMA           = "pmjay_audit"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## Step 1: Create Delta Tables

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.claims_raw (
  claim_id          STRING,
  file_name         STRING,
  raw_text          STRING,
  detected_language STRING,
  file_type         STRING,
  upload_timestamp  TIMESTAMP
) USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.pmjay_rules_raw (
  rule_id     STRING,
  source_file STRING,
  raw_text    STRING,
  ingest_ts   TIMESTAMP
) USING DELTA
""")

print("Delta tables created.")

# COMMAND ----------

# MAGIC %md ## Step 2: Extract text from PDFs

# COMMAND ----------

def extract_pdf_text(path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception:
        pass
    if len(text.strip()) < 50:
        doc = fitz.open(path)
        text = "\n".join(p.get_text() for p in doc)
        doc.close()
    return text.strip()


def detect_lang(text: str) -> str:
    from langdetect import detect, LangDetectException
    try:
        return detect(text[:500])
    except LangDetectException:
        return "en"


def ingest_claims_from_dbfs(dbfs_path: str):
    rows = []
    os.makedirs(dbfs_path, exist_ok=True)
    for fname in os.listdir(dbfs_path):
        fpath = os.path.join(dbfs_path, fname)
        if fname.endswith(".pdf"):
            text = extract_pdf_text(fpath)
        elif fname.endswith(".txt"):
            with open(fpath) as f:
                text = f.read()
        else:
            continue
        rows.append((
            str(uuid.uuid4()),
            fname,
            text,
            detect_lang(text),
            fname.rsplit(".", 1)[-1],
            datetime.utcnow(),
        ))
        print(f"  Ingested: {fname} ({len(text)} chars)")

    if rows:
        schema = StructType([
            StructField("claim_id",          StringType()),
            StructField("file_name",         StringType()),
            StructField("raw_text",          StringType()),
            StructField("detected_language", StringType()),
            StructField("file_type",         StringType()),
            StructField("upload_timestamp",  TimestampType()),
        ])
        df = spark.createDataFrame(rows, schema)
        df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.claims_raw")
        print(f"Wrote {len(rows)} claims to Delta.")
    else:
        print("No files found. Upload PDFs/TXTs to DBFS path:", dbfs_path)

# COMMAND ----------

ingest_claims_from_dbfs(CLAIMS_DBFS_PATH)

# COMMAND ----------

# MAGIC %md ## Step 3: Ingest PM-JAY Rulebook

# COMMAND ----------

def ingest_rules_from_dbfs(dbfs_path: str):
    rows = []
    os.makedirs(dbfs_path, exist_ok=True)
    for fname in os.listdir(dbfs_path):
        fpath = os.path.join(dbfs_path, fname)
        if fname.endswith(".pdf"):
            text = extract_pdf_text(fpath)
        elif fname.endswith(".txt"):
            with open(fpath) as f:
                text = f.read()
        else:
            continue
        rows.append((str(uuid.uuid4()), fname, text, datetime.utcnow()))

    if rows:
        schema = StructType([
            StructField("rule_id",     StringType()),
            StructField("source_file", StringType()),
            StructField("raw_text",    StringType()),
            StructField("ingest_ts",   TimestampType()),
        ])
        df = spark.createDataFrame(rows, schema)
        df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.pmjay_rules_raw")
        print(f"Wrote {len(rows)} rule files to Delta.")

ingest_rules_from_dbfs(RULES_DBFS_PATH)

# COMMAND ----------

# MAGIC %md ## Step 4: Verify

# COMMAND ----------

spark.sql(f"SELECT claim_id, file_name, detected_language, LEFT(raw_text, 100) as preview FROM {CATALOG}.{SCHEMA}.claims_raw").show(truncate=False)
spark.sql(f"SELECT COUNT(*) as rule_docs FROM {CATALOG}.{SCHEMA}.pmjay_rules_raw").show()
