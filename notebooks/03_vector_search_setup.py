# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 — Build PM-JAY Rules Vector Index (FAISS on DBFS)
# MAGIC Chunks the PM-JAY rulebook, embeds with sentence-transformers, saves FAISS index to DBFS.

# COMMAND ----------

# MAGIC %pip install langchain langchain-community sentence-transformers faiss-cpu -q

# COMMAND ----------

import sys
sys.path.insert(0, "/Workspace/Repos/your-repo/medical-claim-auditor")

from pyspark.sql import SparkSession
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

spark = SparkSession.builder.getOrCreate()

CATALOG           = "hive_metastore"
SCHEMA            = "pmjay_audit"
VECTOR_STORE_PATH = "/dbfs/pmjay_audit/vector_store"
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"

# COMMAND ----------

# MAGIC %md ## Step 1: Load all PM-JAY rules text from Delta

# COMMAND ----------

rules_df = spark.table(f"{CATALOG}.{SCHEMA}.pmjay_rules_raw")
all_rules_text = "\n\n".join(
    [row.raw_text for row in rules_df.select("raw_text").collect()]
)
print(f"Total rules text: {len(all_rules_text):,} characters")

# COMMAND ----------

# MAGIC %md ## Step 2: Chunk the rules

# COMMAND ----------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", "—", "-", " "],
)
chunks = splitter.split_text(all_rules_text)
print(f"Created {len(chunks)} rule chunks")

# Preview first 3
for i, c in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{c[:200]}")

# COMMAND ----------

# MAGIC %md ## Step 3: Build embeddings and FAISS index

# COMMAND ----------

print("Loading embedding model (may take 1-2 min on first run)…")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Building FAISS index…")
vector_store = FAISS.from_texts(chunks, embeddings)

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store.save_local(os.path.join(VECTOR_STORE_PATH, "pmjay_faiss_index"))
print(f"Index saved to {VECTOR_STORE_PATH}")

# COMMAND ----------

# MAGIC %md ## Step 4: Test retrieval

# COMMAND ----------

test_queries = [
    "cataract surgery maximum days of stay",
    "knee replacement package rate",
    "coronary artery bypass cost limit",
]

for q in test_queries:
    docs = vector_store.similarity_search(q, k=3)
    print(f"\nQuery: {q}")
    for d in docs:
        print(f"  → {d.page_content[:120]}")

# COMMAND ----------

# MAGIC %md ## Step 5: Also store chunk metadata in Delta for audit trail

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import current_timestamp
import uuid
from datetime import datetime

rows = [(str(uuid.uuid4()), i, c, datetime.utcnow()) for i, c in enumerate(chunks)]
schema = StructType([
    StructField("chunk_id",   StringType()),
    StructField("chunk_index", IntegerType()),
    StructField("chunk_text",  StringType()),
    StructField("indexed_at",  TimestampType()),
])

spark.createDataFrame(rows, schema).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.pmjay_rule_chunks"
)
print(f"Stored {len(rows)} chunks in Delta table pmjay_rule_chunks.")
