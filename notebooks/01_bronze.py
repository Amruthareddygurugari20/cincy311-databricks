# Databricks notebook source
# MAGIC %md
# MAGIC # 01_bronze — Raw ingestion of Cincinnati 311 requests
# MAGIC
# MAGIC **Bronze** is the raw landing layer: data as it arrived, plus lineage metadata.
# MAGIC No business cleaning happens here — that's silver's job.
# MAGIC
# MAGIC - Source: Cincinnati 311 (Non-Emergency) Service Requests CSV in `/Volumes/workspace/cincy311/raw/`
# MAGIC - Target: Delta table `workspace.cincy311.bronze_311`
# MAGIC - Re-run safe: the table is **overwritten** on every run, never appended.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Settings and check the raw files
# MAGIC All names live in one place. We create the schema if missing and list the files in the Volume.

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "cincy311"
RAW_PATH = "/Volumes/workspace/cincy311/raw/"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_311"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
files = dbutils.fs.ls(RAW_PATH)
for f in files:
    print(f"{f.name:60} {f.size / 1024 / 1024:10.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Read the CSV as all strings
# MAGIC `inferSchema=false` keeps every column a string so nothing is silently turned into null.
# MAGIC `multiLine` + `escape` handle free-text fields with line breaks and quotes.
# MAGIC `_metadata.file_path` is a hidden column Databricks adds to file reads — we keep it for lineage.

# COMMAND ----------

from pyspark.sql import functions as F

raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")   # keep everything as strings
    .option("multiLine", "true")      # descriptions can contain line breaks
    .option("escape", '"')            # handle quotes inside text fields
    .option("pathGlobFilter", "*.csv")
    .csv(RAW_PATH)
    .withColumn("_source_file", F.col("_metadata.file_path"))
)

print("Number of columns:", len(raw_df.columns))
print(raw_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Clean column names to lowercase_with_underscores
# MAGIC Delta does not allow spaces/symbols in column names, and clean names are easier in SQL.
# MAGIC The assert stops the run if two headers would collapse into the same name.

# COMMAND ----------

import re

def clean_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)   # spaces/symbols -> underscore
    name = name.strip("_")
    return name or "unnamed"

csv_cols = [c for c in raw_df.columns if c != "_source_file"]
new_names = [clean_name(c) for c in csv_cols]

dupes = {n for n in new_names if new_names.count(n) > 1}
assert not dupes, f"Two columns became the same name after cleaning: {dupes}"

bronze_df = raw_df.select(
    *[F.col(f"`{old}`").alias(new) for old, new in zip(csv_cols, new_names)],
    F.col("_source_file"),
    F.current_timestamp().alias("_ingested_at"),
)

for old, new in zip(csv_cols, new_names):
    print(f"{old!r:45} -> {new}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Write the Delta table (safe to re-run)
# MAGIC `overwrite` replaces the table each run, so re-running never duplicates rows.
# MAGIC `overwriteSchema` lets the table follow the file if the city adds or renames a column.

# COMMAND ----------

(
    bronze_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)
print("Wrote", BRONZE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Verify what landed
# MAGIC Read the table back from the catalog: row count, schema, and rows per source file.

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)
row_count = bronze.count()
print("Rows in bronze:", row_count)
print("Columns:", len(bronze.columns))
bronze.printSchema()

per_file = bronze.groupBy("_source_file").count().collect()
for r in per_file:
    print(r["_source_file"], r["count"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Delta history
# MAGIC Every write is a numbered version. Run the notebook twice: two versions, same row count.

# COMMAND ----------

history = spark.sql(f"DESCRIBE HISTORY {BRONZE_TABLE}").select("version", "timestamp", "operation").limit(5)
for r in history.collect():
    print(r["version"], r["timestamp"], r["operation"])

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "table": BRONZE_TABLE,
    "rows": row_count,
    "columns": len(bronze.columns),
    "files": [f.name for f in files],
}))
