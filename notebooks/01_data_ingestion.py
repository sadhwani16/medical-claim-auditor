# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01 — Data Ingestion into Delta Lake
# MAGIC
# MAGIC ## How to upload your government guideline PDFs
# MAGIC 1. In Databricks left sidebar → **Catalog** → click the DBFS icon (top right of Catalog page) → **Upload**
# MAGIC 2. Navigate to `/FileStore/pmjay_audit/rules/`
# MAGIC 3. Upload ALL your government guideline PDFs here
# MAGIC 4. For claims: upload PDFs/TXTs to `/FileStore/pmjay_audit/claims/`
# MAGIC
# MAGIC Then run this notebook — it will extract text from every PDF automatically.

# COMMAND ----------

# MAGIC %pip install pdfplumber PyMuPDF langdetect -q

# COMMAND ----------

import pdfplumber
import fitz
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType
from pyspark.sql.functions import lit, current_timestamp
import os, uuid
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

# ── Paths (Community Edition uses FileStore) ──────────────────────────────────
CLAIMS_DBFS_PATH = "/dbfs/FileStore/pmjay_audit/claims"
RULES_DBFS_PATH  = "/dbfs/FileStore/pmjay_audit/rules"
CATALOG          = "hive_metastore"
SCHEMA           = "pmjay_audit"

os.makedirs(CLAIMS_DBFS_PATH, exist_ok=True)
os.makedirs(RULES_DBFS_PATH, exist_ok=True)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")
print(f"Rules path:   {RULES_DBFS_PATH}")
print(f"Claims path:  {CLAIMS_DBFS_PATH}")

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
  file_size_chars   LONG,
  upload_timestamp  TIMESTAMP
) USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.pmjay_rules_raw (
  rule_id     STRING,
  source_file STRING,
  raw_text    STRING,
  file_size_chars LONG,
  ingest_ts   TIMESTAMP
) USING DELTA
""")

print("Delta tables created.")

# COMMAND ----------

# MAGIC %md ## Step 2: PDF text extraction helpers

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
    # Fallback to PyMuPDF for scanned/image PDFs
    if len(text.strip()) < 50:
        try:
            doc = fitz.open(path)
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
        except Exception:
            pass
    return text.strip()


def detect_lang(text: str) -> str:
    from langdetect import detect, LangDetectException
    try:
        return detect(text[:500])
    except LangDetectException:
        return "en"


def read_file(fpath: str, fname: str) -> str:
    if fname.lower().endswith(".pdf"):
        return extract_pdf_text(fpath)
    elif fname.lower().endswith((".txt", ".md")):
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

# COMMAND ----------

# MAGIC %md ## Step 3: Ingest Government Guideline PDFs (Rules)

# COMMAND ----------

def ingest_rules(dbfs_path: str):
    files = [f for f in os.listdir(dbfs_path) if f.lower().endswith((".pdf", ".txt"))]
    if not files:
        print(f"No files found in {dbfs_path}")
        print("Please upload your government guideline PDFs to that DBFS path first.")
        return

    print(f"Found {len(files)} rule files: {files}")
    rows = []
    for fname in files:
        fpath = os.path.join(dbfs_path, fname)
        text  = read_file(fpath, fname)
        if not text:
            print(f"  WARNING: Could not extract text from {fname} (may be scanned image)")
            continue
        rows.append((str(uuid.uuid4()), fname, text, len(text), datetime.utcnow()))
        print(f"  Ingested: {fname} — {len(text):,} chars extracted")

    if rows:
        schema = StructType([
            StructField("rule_id",         StringType()),
            StructField("source_file",     StringType()),
            StructField("raw_text",        StringType()),
            StructField("file_size_chars", LongType()),
            StructField("ingest_ts",       TimestampType()),
        ])
        df = spark.createDataFrame(rows, schema)
        df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.pmjay_rules_raw")
        print(f"\nWrote {len(rows)} rule documents to Delta table pmjay_rules_raw.")

ingest_rules(RULES_DBFS_PATH)

# COMMAND ----------

# MAGIC %md ## Step 4: Ingest Claim Files

# COMMAND ----------

def ingest_claims(dbfs_path: str):
    files = [f for f in os.listdir(dbfs_path) if f.lower().endswith((".pdf", ".txt"))]
    if not files:
        print(f"No claim files found in {dbfs_path}")
        return

    print(f"Found {len(files)} claim files")
    existing = {
        row.file_name
        for row in spark.table(f"{CATALOG}.{SCHEMA}.claims_raw").select("file_name").collect()
    }

    rows = []
    for fname in files:
        if fname in existing:
            print(f"  Skipping (already ingested): {fname}")
            continue
        fpath = os.path.join(dbfs_path, fname)
        text  = read_file(fpath, fname)
        if not text:
            print(f"  WARNING: Could not extract text from {fname}")
            continue
        rows.append((
            str(uuid.uuid4()), fname, text, detect_lang(text),
            fname.rsplit(".", 1)[-1], len(text), datetime.utcnow(),
        ))
        print(f"  Ingested: {fname} ({len(text):,} chars)")

    if rows:
        schema = StructType([
            StructField("claim_id",          StringType()),
            StructField("file_name",         StringType()),
            StructField("raw_text",          StringType()),
            StructField("detected_language", StringType()),
            StructField("file_type",         StringType()),
            StructField("file_size_chars",   LongType()),
            StructField("upload_timestamp",  TimestampType()),
        ])
        df = spark.createDataFrame(rows, schema)
        df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.claims_raw")
        print(f"Wrote {len(rows)} new claims to Delta.")

ingest_claims(CLAIMS_DBFS_PATH)

# COMMAND ----------

# MAGIC %md ## Step 5: Verify ingestion

# COMMAND ----------

print("=== RULES ===")
spark.sql(f"""
  SELECT source_file, file_size_chars, LEFT(raw_text, 120) AS preview
  FROM {CATALOG}.{SCHEMA}.pmjay_rules_raw
""").show(truncate=False)

print("=== CLAIMS ===")
spark.sql(f"""
  SELECT claim_id, file_name, detected_language, file_size_chars,
         LEFT(raw_text, 100) AS preview
  FROM {CATALOG}.{SCHEMA}.claims_raw
""").show(truncate=False)
