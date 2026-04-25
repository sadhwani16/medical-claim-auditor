# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 — Multilingual Translation
# MAGIC Translates non-English claims (Hindi, Tamil, Telugu, etc.) to English.
# MAGIC Falls back to Google Translate (free) if no Sarvam key is set.

# COMMAND ----------

# MAGIC %pip install langdetect requests deep-translator -q

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, current_timestamp, lit
from pyspark.sql.types import StringType
import requests

spark = SparkSession.builder.getOrCreate()

CATALOG        = "hive_metastore"
SCHEMA         = "pmjay_audit"

# ── API Keys ──────────────────────────────────────────────────────────────────
# If you have a Sarvam AI key (free at sarvam.ai), paste it below.
# Otherwise leave as "" and it will use free Google Translate as fallback.
SARVAM_API_KEY = ""   # e.g. "your-sarvam-key-here"

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

# MAGIC %md ## Step 2: Translation function

# COMMAND ----------

def translate_text(text: str, src_lang: str) -> str:
    if not text or src_lang == "en" or src_lang is None:
        return text

    # Try Sarvam AI first (better for Indian languages)
    sarvam_code = SARVAM_LANG_MAP.get(src_lang)
    if sarvam_code and SARVAM_API_KEY:
        try:
            chunks = [text[i:i+900] for i in range(0, len(text), 900)]
            parts  = []
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
                parts.append(resp.json().get("translated_text", chunk) if resp.status_code == 200 else chunk)
            return " ".join(parts)
        except Exception:
            pass

    # Fallback: free Google Translate via deep-translator
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src_lang, target="en").translate(text[:4999])
    except Exception:
        return text


translate_udf = udf(translate_text, StringType())

# COMMAND ----------

# MAGIC %md ## Step 3: Translate all pending claims

# COMMAND ----------

raw_df            = spark.table(f"{CATALOG}.{SCHEMA}.claims_raw")
already_done      = spark.table(f"{CATALOG}.{SCHEMA}.claims_translated").select("claim_id")
pending           = raw_df.join(already_done, on="claim_id", how="left_anti")

print(f"Claims pending translation: {pending.count()}")

translated_df = (
    pending
    .withColumn(
        "translated_text",
        when(col("detected_language") == "en", col("raw_text"))
        .otherwise(translate_udf(col("raw_text"), col("detected_language")))
    )
    .withColumn(
        "translation_method",
        when(col("detected_language") == "en", lit("none"))
        .when(lit(bool(SARVAM_API_KEY)), lit("sarvam-ai"))
        .otherwise(lit("google-translate"))
    )
    .withColumn("translated_at", current_timestamp())
    .select(
        "claim_id",
        "translated_text",
        col("detected_language").alias("source_language"),
        "translation_method",
        "translated_at",
    )
)

translated_df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.claims_translated")
print("Translation complete.")

# COMMAND ----------

# MAGIC %md ## Step 4: Preview

# COMMAND ----------

spark.sql(f"""
  SELECT c.file_name, c.detected_language, t.translation_method,
         LEFT(c.raw_text, 80) AS original,
         LEFT(t.translated_text, 80) AS translated
  FROM {CATALOG}.{SCHEMA}.claims_raw c
  JOIN {CATALOG}.{SCHEMA}.claims_translated t ON c.claim_id = t.claim_id
""").show(truncate=False)
