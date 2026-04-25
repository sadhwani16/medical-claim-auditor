# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 — Multilingual Translation (Sarvam AI)
# MAGIC Read non-English claims from Delta, translate via Sarvam AI, write back.

# COMMAND ----------

# MAGIC %pip install langdetect requests deep-translator -q

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/your-repo/medical-claim-auditor")  # adjust path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import os, requests, re

spark = SparkSession.builder.getOrCreate()

CATALOG  = "hive_metastore"
SCHEMA   = "pmjay_audit"
SARVAM_API_KEY = dbutils.secrets.get(scope="pmjay", key="sarvam_api_key")  # noqa

SARVAM_LANG_MAP = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN",
    "gu": "gu-IN", "mr": "mr-IN", "pa": "pa-IN",
}

# COMMAND ----------

# MAGIC %md ## Step 1: Create translated claims table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.claims_translated (
  claim_id           STRING,
  translated_text    STRING,
  source_language    STRING,
  translation_method STRING,
  translated_at      TIMESTAMP
) USING DELTA
""")

# COMMAND ----------

# MAGIC %md ## Step 2: UDF for translation

# COMMAND ----------

def translate_text(text: str, src_lang: str) -> str:
    if src_lang == "en" or src_lang is None:
        return text
    sarvam_code = SARVAM_LANG_MAP.get(src_lang)
    if sarvam_code and SARVAM_API_KEY:
        try:
            chunks = [text[i:i+900] for i in range(0, len(text), 900)]
            translated_parts = []
            for chunk in chunks:
                resp = requests.post(
                    "https://api.sarvam.ai/translate",
                    headers={"api-subscription-key": SARVAM_API_KEY},
                    json={
                        "input": chunk,
                        "source_language_code": sarvam_code,
                        "target_language_code": "en-IN",
                        "mode": "formal",
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    translated_parts.append(resp.json().get("translated_text", chunk))
                else:
                    translated_parts.append(chunk)
            return " ".join(translated_parts)
        except Exception:
            pass
    # Fallback: deep-translator
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except Exception:
        return text

translate_udf = udf(translate_text, StringType())

# COMMAND ----------

# MAGIC %md ## Step 3: Translate all non-English claims

# COMMAND ----------

raw_df = spark.table(f"{CATALOG}.{SCHEMA}.claims_raw")
already_translated = spark.table(f"{CATALOG}.{SCHEMA}.claims_translated").select("claim_id")

pending = raw_df.join(already_translated, on="claim_id", how="left_anti")
print(f"Claims pending translation: {pending.count()}")

translated_df = (
    pending
    .withColumn(
        "translated_text",
        when(col("detected_language") == "en", col("raw_text"))
        .otherwise(translate_udf(col("raw_text"), col("detected_language")))
    )
    .withColumn("translation_method",
        when(col("detected_language") == "en", lit("none"))
        .otherwise(lit("sarvam-ai")))
    .withColumn("translated_at", current_timestamp())
    .select("claim_id", "translated_text", col("detected_language").alias("source_language"),
            "translation_method", "translated_at")
)

translated_df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.claims_translated")
print("Translation complete.")

# COMMAND ----------

# MAGIC %md ## Step 4: Preview

# COMMAND ----------

spark.sql(f"""
SELECT c.file_name, c.detected_language, LEFT(c.raw_text, 100) as original, LEFT(t.translated_text, 100) as translated
FROM {CATALOG}.{SCHEMA}.claims_raw c
JOIN {CATALOG}.{SCHEMA}.claims_translated t ON c.claim_id = t.claim_id
""").show(truncate=False)
