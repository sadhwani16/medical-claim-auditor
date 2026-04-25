import os
from dotenv import load_dotenv

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "databricks")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "databricks-meta-llama-3-1-70b-instruct")

DELTA_CATALOG = os.getenv("DELTA_CATALOG", "hive_metastore")
DELTA_SCHEMA = os.getenv("DELTA_SCHEMA", "pmjay_audit")

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "/tmp/pmjay_audit/vector_store")
CLAIMS_UPLOAD_PATH = os.getenv("CLAIMS_UPLOAD_PATH", "/tmp/pmjay_audit/claims")
RULES_PATH = os.getenv("RULES_PATH", "/tmp/pmjay_audit/rules")

FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.55"))

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
}

SARVAM_LANG_MAP = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
}
