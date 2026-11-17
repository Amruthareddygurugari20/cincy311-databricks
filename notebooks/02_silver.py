# Databricks notebook source
# MAGIC %md
# MAGIC # 02_silver — Clean, type, de-duplicate and quality-check
# MAGIC
# MAGIC **Silver** is the trusted, cleaned version of bronze: real data types, consistent text,
# MAGIC one row per service request, and bad rows split off into a quarantine table with a reason.
# MAGIC
# MAGIC - Input: `workspace.cincy311.bronze_311`
# MAGIC - Output: `workspace.cincy311.silver_311` (good rows) and `workspace.cincy311.silver_311_quarantine` (bad rows + reason)
# MAGIC - Re-run safe: both tables are fully rebuilt from bronze with `overwrite`.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

BRONZE_TABLE = "workspace.cincy311.bronze_311"
SILVER_TABLE = "workspace.cincy311.silver_311"
QUARANTINE_TABLE = "workspace.cincy311.silver_311_quarantine"

bronze = spark.table(BRONZE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Look before you clean: schema and 5 sample rows
# MAGIC We check the real column names and value formats instead of guessing them.

# COMMAND ----------

bronze.printSchema()
bronze.select(
    "sr_number", "sr_type_desc", "dept_name", "neighborhood",
    "date_created", "time_received", "date_closed", "sr_status", "sr_status_flag",
).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Trim all text and turn empty strings into nulls
# MAGIC Stray spaces make "WESTWOOD" and "WESTWOOD " look like two neighborhoods.
# MAGIC Empty strings become real nulls so null checks catch them.

# COMMAND ----------

string_cols = [c for c, t in bronze.dtypes if t == "string"]
trimmed = bronze.select(
    *[
        F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c))).alias(c)
        if c in string_cols else F.col(c)
        for c in bronze.columns
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Parse dates, times and numbers
# MAGIC Dates arrive as text like `11/07/2026 12:00:00 AM` (date only), and `time_received` comes in several
# MAGIC formats (`1:03 PM`, `13:03`, `1:03:22 PM`). `try_to_timestamp` returns null instead of crashing on a bad
# MAGIC value, so bad dates can be caught by the quality checks below.

# COMMAND ----------

DATE_FMT = "MM/dd/yyyy hh:mm:ss a"

def parse_date(col):
    return F.to_date(F.expr(f"try_to_timestamp({col}, '{DATE_FMT}')"))

typed = (
    trimmed
    .withColumn("created_date", parse_date("date_created"))
    .withColumn("closed_date", parse_date("date_closed"))
    .withColumn("last_update_date", parse_date("date_last_update"))
    .withColumn("_created_str", F.date_format("created_date", "yyyy-MM-dd"))
    .withColumn(
        "created_ts",
        F.expr("""coalesce(
            try_to_timestamp(concat(_created_str, ' ', time_received), 'yyyy-MM-dd h:mm a'),
            try_to_timestamp(concat(_created_str, ' ', time_received), 'yyyy-MM-dd h:mm:ss a'),
            try_to_timestamp(concat(_created_str, ' ', time_received), 'yyyy-MM-dd H:mm')
        )"""),
    )
    .withColumn(
        "last_update_ts",
        F.expr("""coalesce(
            try_to_timestamp(concat(date_format(last_update_date, 'yyyy-MM-dd'), ' ', time_last_update), 'yyyy-MM-dd h:mm a'),
            cast(last_update_date as timestamp)
        )"""),
    )
    .withColumn("latitude", F.expr("try_cast(latitude as double)"))
    .withColumn("longitude", F.expr("try_cast(longitude as double)"))
    .withColumn("planned_completion_days", F.expr("try_cast(planned_completion_days as int)"))
    .drop("_created_str")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Standardize neighborhood names
# MAGIC Upper-case, collapse double spaces, spell out `MT.` as `MOUNT` (matching the city's community-council names),
# MAGIC and label `N/A` or missing as `UNKNOWN` so these rows still count in totals.

# COMMAND ----------

nbhd = F.upper(F.regexp_replace(F.col("neighborhood"), r"\s+", " "))
nbhd = F.regexp_replace(nbhd, r"^MT\.?\s+", "MOUNT ")
typed = typed.withColumn(
    "neighborhood",
    F.when(nbhd.isNull() | nbhd.isin("N/A", "NA", "NONE"), "UNKNOWN").otherwise(nbhd),
)

before = trimmed.select("neighborhood").distinct().count()
after = typed.select("neighborhood").distinct().count()
print(f"Distinct neighborhoods: before={before}, after={after}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Remove duplicate service request numbers (keep the latest record)
# MAGIC Some request numbers appear more than once. We rank each request's rows by last update (newest first)
# MAGIC and keep only rank 1, so every request appears exactly once.

# COMMAND ----------

latest_first = Window.partitionBy("sr_number").orderBy(
    F.col("last_update_ts").desc_nulls_last(),
    F.col("closed_date").desc_nulls_last(),
    F.col("_ingested_at").desc(),
)
deduped = (
    typed
    .withColumn("_rn", F.row_number().over(latest_first))
    .filter("_rn = 1")
    .drop("_rn")
)

rows_before = typed.count()
rows_after = deduped.count()
print(f"Rows before de-dup: {rows_before:,}")
print(f"Rows after de-dup:  {rows_after:,}")
print(f"Duplicates removed: {rows_before - rows_after:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Add days_to_close and a few useful columns
# MAGIC `days_to_close` is closed date minus created date, in whole days (the source dates have no time part).
# MAGIC It stays null for open requests. `is_closed` comes from the city's own status flag.

# COMMAND ----------

enriched = (
    deduped
    .withColumn("days_to_close", F.datediff("closed_date", "created_date"))
    .withColumn("is_closed", F.col("sr_status_flag") == "CLOSED")
    .withColumn("created_month", F.trunc("created_date", "month"))
    .withColumn("created_day_of_week", F.date_format("created_date", "EEEE"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Data quality check: null counts on key columns
# MAGIC These are the columns later steps depend on. `closed_date` is expected to be null for open requests.

# COMMAND ----------

key_cols = ["sr_number", "created_date", "sr_type_desc", "dept_name", "neighborhood", "closed_date"]
null_counts = enriched.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in key_cols]
).collect()[0].asDict()
for c in key_cols:
    print(f"{c:15} nulls: {null_counts[c]:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Quality rules → a reason for every bad row
# MAGIC Each rule adds a text reason when it fails. A row with at least one reason goes to quarantine,
# MAGIC so bad data is kept for review instead of silently deleted.

# COMMAND ----------

today = F.current_date()
reason = F.concat_ws(
    "; ",
    F.when(F.col("sr_number").isNull(), "missing sr_number"),
    F.when(F.col("created_date").isNull(), "missing or unparseable created date"),
    F.when(F.col("sr_type_desc").isNull(), "missing request type"),
    F.when(F.col("dept_name").isNull(), "missing department"),
    F.when(F.col("closed_date") < F.col("created_date"), "closed before created"),
    F.when(F.col("created_date") > today, "created date in the future"),
    F.when(F.col("closed_date") > today, "closed date in the future"),
)
checked = enriched.withColumn("dq_reason", reason)

good = checked.filter(F.col("dq_reason") == "").drop("dq_reason")
bad = checked.filter(F.col("dq_reason") != "").withColumn("_quarantined_at", F.current_timestamp())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9. Write silver and quarantine tables (safe to re-run)
# MAGIC Silver keeps a focused set of typed columns. Quarantine keeps the same columns plus the reason.
# MAGIC Both are overwritten each run, so re-running never duplicates rows.

# COMMAND ----------

SILVER_COLS = [
    "sr_number", "sr_type", "sr_type_desc", "priority", "dept_code", "dept_name", "dept_division",
    "neighborhood", "community_council_neighborhood", "zipcode", "address", "method_received",
    "sr_status", "sr_status_flag", "is_closed", "created_date", "created_ts", "closed_date",
    "last_update_ts", "days_to_close", "created_month", "created_day_of_week",
    "planned_completion_days", "latitude", "longitude", "police_district",
    "_source_file", "_ingested_at",
]

(good.select(SILVER_COLS)
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE))

(bad.select(*SILVER_COLS, "dq_reason", "_quarantined_at")
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(QUARANTINE_TABLE))

print("Wrote", SILVER_TABLE, "and", QUARANTINE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10. Results: how many passed, how many were quarantined, and why

# COMMAND ----------

silver = spark.table(SILVER_TABLE)
quarantine = spark.table(QUARANTINE_TABLE)
passed, quarantined = silver.count(), quarantine.count()

print(f"Rows passed to silver:     {passed:,}")
print(f"Rows sent to quarantine:   {quarantined:,}")
print(f"Total (should equal de-duped rows {rows_after:,}): {passed + quarantined:,}")

print("\nQuarantine reasons:")
for r in quarantine.groupBy("dq_reason").count().orderBy(F.desc("count")).collect():
    print(f"  {r['count']:>8,}  {r['dq_reason']}")

print("\ndays_to_close on closed silver rows:")
silver.filter("days_to_close IS NOT NULL").select(
    F.count("*").alias("closed_rows"),
    F.round(F.avg("days_to_close"), 2).alias("avg_days"),
    F.expr("percentile_approx(days_to_close, 0.5)").alias("median_days"),
    F.min("days_to_close").alias("min_days"),
    F.max("days_to_close").alias("max_days"),
).show()

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "bronze_rows": rows_before, "deduped_rows": rows_after,
    "duplicates_removed": rows_before - rows_after,
    "silver_rows": passed, "quarantine_rows": quarantined,
}))
