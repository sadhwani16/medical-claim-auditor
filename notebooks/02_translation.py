# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 — Multilingual Translation (IndicTrans2 + Sarvam AI)
# MAGIC
# MAGIC Translation priority:
# MAGIC 1. **IndicTrans2** (AI4Bharat) — best quality for 22 Indian languages, free via HuggingFace
# MAGIC 2. **Sarvam AI** — good for 10 Indian languages, needs API key from sarvam.ai
# MAGIC 3. **Google Translate** — free fallback via deep-translator

# COMMAND ----------

# MAGIC %pip install langdetect requests deep-translator sentencepiece sacremoses -q

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when, current_timestamp, lit
from pyspark.sql.types import StringType
import requests

spark = SparkSession.builder.getOrCreate()

CATALOG        = "hive_metastore"
SCHEMA         = "pmjay_audit"

# ── API Keys ──────────────────────────────────────────────────────────────────
# Get free key at: https://sarvam.ai  → Dashboard → API Keys
SARVAM_API_KEY = ""   # paste here if you have one, otherwise leave blank

# HuggingFace token (free at huggingface.co → Settings → Access Tokens)
# Only needed if IndicTrans2 model is gated — usually not required
HF_TOKEN = ""

# ── Language maps ─────────────────────────────────────────────────────────────
# IndicTrans2 language codes (Flores-101 format)
INDICTRANS2_LANG_MAP = {
    "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "bn": "ben_Beng",
    "gu": "guj_Gujr", "mr": "mar_Deva", "pa": "pan_Guru",
    "or": "ory_Orya", "as": "asm_Beng", "ur": "urd_Arab",
}

# Sarvam AI language codes
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

# MAGIC %md ## Step 2: IndicTrans2 translation (AI4Bharat — best quality for Indian languages)

# COMMAND ----------

# IndicTrans2 is loaded once and reused — ~1.2GB model, takes 3-4 min on first run
_indictrans2_pipe = None

def get_indictrans2():
    global _indictrans2_pipe
    if _indictrans2_pipe is not None:
        return _indictrans2_pipe
    try:
        from transformers import pipeline
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        # 200M parameter model — lighter than 1B, fits Community Edition
        _indictrans2_pipe = pipeline(
            "translation",
            model="ai4bharat/indictrans2-indic-en-dist-200M",
            trust_remote_code=True,
        )
        print("IndicTrans2 model loaded.")
    except Exception as e:
        print(f"IndicTrans2 load failed: {e} — will fall back to Sarvam/Google")
        _indictrans2_pipe = None
    return _indictrans2_pipe


def translate_indictrans2(text: str, src_lang: str) -> str | None:
    flores_code = INDICTRANS2_LANG_MAP.get(src_lang)
    if not flores_code:
        return None
    pipe = get_indictrans2()
    if not pipe:
        return None
    try:
        # Split into chunks — model max ~512 tokens
        chunks = [text[i:i+800] for i in range(0, len(text), 800)]
        parts  = []
        for chunk in chunks:
            result = pipe(chunk, src_lang=flores_code, tgt_lang="eng_Latn")
            parts.append(result[0]["translation_text"])
        return " ".join(parts)
    except Exception as e:
        print(f"IndicTrans2 translation error: {e}")
        return None


def translate_sarvam(text: str, src_lang: str) -> str | None:
    if not SARVAM_API_KEY:
        return None
    sarvam_code = SARVAM_LANG_MAP.get(src_lang)
    if not sarvam_code:
        return None
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
        return None


def translate_google_fallback(text: str, src_lang: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=src_lang, target="en").translate(text[:4999])
    except Exception:
        return text


def translate_text(text: str, src_lang: str) -> str:
    if not text or src_lang == "en" or src_lang is None:
        return text

    # 1st choice: IndicTrans2 (best for Indian languages)
    result = translate_indictrans2(text, src_lang)
    if result:
        return result

    # 2nd choice: Sarvam AI
    result = translate_sarvam(text, src_lang)
    if result:
        return result

    # Final fallback: Google Translate (free)
    return translate_google_fallback(text, src_lang)


def get_method(src_lang: str) -> str:
    if src_lang == "en":
        return "none"
    if INDICTRANS2_LANG_MAP.get(src_lang) and get_indictrans2():
        return "indictrans2"
    if SARVAM_API_KEY and SARVAM_LANG_MAP.get(src_lang):
        return "sarvam-ai"
    return "google-translate"


translate_udf = udf(translate_text, StringType())

# COMMAND ----------

# MAGIC %md ## Step 3: Translate all pending claims

# COMMAND ----------

raw_df       = spark.table(f"{CATALOG}.{SCHEMA}.claims_raw")
already_done = spark.table(f"{CATALOG}.{SCHEMA}.claims_translated").select("claim_id")
pending      = raw_df.join(already_done, on="claim_id", how="left_anti")
count        = pending.count()

print(f"Claims pending translation: {count}")
if count == 0:
    print("All claims already translated.")
else:
    translated_df = (
        pending
        .withColumn(
            "translated_text",
            when(col("detected_language") == "en", col("raw_text"))
            .otherwise(translate_udf(col("raw_text"), col("detected_language")))
        )
        .withColumn("translation_method",
            when(col("detected_language") == "en", lit("none"))
            .otherwise(lit(get_method("hi")))   # representative method label
        )
        .withColumn("translated_at", current_timestamp())
        .select(
            "claim_id", "translated_text",
            col("detected_language").alias("source_language"),
            "translation_method", "translated_at",
        )
    )
    translated_df.write.format("delta").mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.claims_translated")
    print("Translation complete.")

# COMMAND ----------

# MAGIC %md ## Step 4: Preview results

# COMMAND ----------

spark.sql(f"""
  SELECT c.file_name, c.detected_language, t.translation_method,
         LEFT(c.raw_text, 80) AS original,
         LEFT(t.translated_text, 80) AS translated
  FROM {CATALOG}.{SCHEMA}.claims_raw c
  JOIN {CATALOG}.{SCHEMA}.claims_translated t ON c.claim_id = t.claim_id
""").show(truncate=False)
