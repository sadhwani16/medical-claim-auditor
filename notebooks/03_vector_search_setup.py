# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 — Build PM-JAY Rules Vector Index (FAISS)
# MAGIC Chunks all government guideline PDFs ingested in Notebook 01,
# MAGIC embeds them with sentence-transformers, and saves a FAISS index to DBFS.

# COMMAND ----------

# MAGIC %pip install langchain langchain-community sentence-transformers faiss-cpu -q

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.sql.functions import current_timestamp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os, uuid
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

CATALOG           = "hive_metastore"
SCHEMA            = "pmjay_audit"
VECTOR_STORE_PATH = "/tmp/pmjay_audit/vector_store"
EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# COMMAND ----------

# MAGIC %md ## Step 1: Load all government guideline text from Delta

# COMMAND ----------

rules_df = spark.table(f"{CATALOG}.{SCHEMA}.pmjay_rules_raw")
rule_count = rules_df.count()
print(f"Loaded {rule_count} guideline documents from Delta")

if rule_count == 0:
    raise Exception("No rules found. Run Notebook 01 first and upload your guideline PDFs.")

# Build (text, source_file) pairs to preserve which PDF each chunk came from
rule_rows = rules_df.select("raw_text", "source_file").collect()
all_texts   = [r.raw_text for r in rule_rows]
all_sources = [r.source_file for r in rule_rows]

total_chars = sum(len(t) for t in all_texts)
print(f"Total guideline text: {total_chars:,} characters across {rule_count} PDFs")
for src, txt in zip(all_sources, all_texts):
    print(f"  {src}: {len(txt):,} chars")

# COMMAND ----------

# MAGIC %md ## Step 2: Chunk the guideline text

# COMMAND ----------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "— ", "- ", " "],
)

all_chunks   = []
chunk_metas  = []   # track which source PDF each chunk came from

for source_file, text in zip(all_sources, all_texts):
    chunks = splitter.split_text(text)
    all_chunks.extend(chunks)
    chunk_metas.extend([{"source": source_file}] * len(chunks))

print(f"Created {len(all_chunks)} chunks from {rule_count} guideline PDFs")
print(f"Average chunk size: {sum(len(c) for c in all_chunks) // len(all_chunks)} chars")

# Preview first 3 chunks
for i, c in enumerate(all_chunks[:3]):
    print(f"\n--- Chunk {i+1} ({chunk_metas[i]['source']}) ---\n{c[:200]}")

# COMMAND ----------

# MAGIC %md ## Step 3: Build FAISS index with embeddings

# COMMAND ----------

print("Loading embedding model (first run downloads ~90MB, takes 1-2 min)…")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print(f"Building FAISS index for {len(all_chunks)} chunks…")
vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=chunk_metas)

index_path = os.path.join(VECTOR_STORE_PATH, "pmjay_faiss_index")
vector_store.save_local(index_path)
print(f"FAISS index saved to {index_path}")

# COMMAND ----------

# MAGIC %md ## Step 4: Test retrieval

# COMMAND ----------

test_queries = [
    "cataract surgery maximum days of stay",
    "knee replacement package rate",
    "coronary artery bypass cost limit",
    "fraud overstay hospitalization",
    "pre-authorization required procedure",
]

print("=== Retrieval Test ===")
for q in test_queries:
    docs = vector_store.similarity_search(q, k=2)
    print(f"\nQuery: {q}")
    for d in docs:
        print(f"  [{d.metadata.get('source', '?')}] {d.page_content[:120]}")

# COMMAND ----------

# MAGIC %md ## Step 5: Store chunk metadata in Delta for audit trail

# COMMAND ----------

rows = [
    (str(uuid.uuid4()), i, c, chunk_metas[i].get("source", ""), datetime.utcnow())
    for i, c in enumerate(all_chunks)
]
schema = StructType([
    StructField("chunk_id",    StringType()),
    StructField("chunk_index", IntegerType()),
    StructField("chunk_text",  StringType()),
    StructField("source_file", StringType()),
    StructField("indexed_at",  TimestampType()),
])

spark.createDataFrame(rows, schema).write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.pmjay_rule_chunks"
)
print(f"Stored {len(rows)} chunks in Delta table pmjay_rule_chunks.")

spark.sql(f"""
  SELECT source_file, COUNT(*) AS chunks
  FROM {CATALOG}.{SCHEMA}.pmjay_rule_chunks
  GROUP BY source_file ORDER BY chunks DESC
""").show()
