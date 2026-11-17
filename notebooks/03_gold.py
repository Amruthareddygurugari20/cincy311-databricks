# Databricks notebook source
# MAGIC %md
# MAGIC # 03_gold — Business-ready aggregate tables
# MAGIC
# MAGIC **Gold** tables answer specific business questions and are small enough for dashboards.
# MAGIC Each one is built from `workspace.cincy311.silver_311` with `CREATE OR REPLACE TABLE ... AS SELECT`,
# MAGIC so every run fully rebuilds it — re-run safe, never duplicated.
# MAGIC
# MAGIC | Table | Question it answers |
# MAGIC |---|---|
# MAGIC | `gold_requests_by_neighborhood_month` | Where and when do residents ask for help? |
# MAGIC | `gold_close_time_by_type` | How long does each request type take, per department? |
# MAGIC | `gold_open_backlog` | What is still open, and how old is it? |

# COMMAND ----------

SILVER = "workspace.cincy311.silver_311"
GOLD = "workspace.cincy311"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Requests per neighborhood per month
# MAGIC One row per neighborhood and month, with total, closed and still-open counts.
# MAGIC This is the shape a trend chart or map needs.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.gold_requests_by_neighborhood_month
COMMENT 'Count of 311 requests per neighborhood per month (from silver_311)'
AS
SELECT
  neighborhood,
  created_month                                   AS month,
  COUNT(*)                                        AS request_count,
  SUM(CASE WHEN is_closed THEN 1 ELSE 0 END)      AS closed_count,
  SUM(CASE WHEN NOT is_closed THEN 1 ELSE 0 END)  AS open_count
FROM {SILVER}
GROUP BY neighborhood, created_month
""")

t1 = spark.table(f"{GOLD}.gold_requests_by_neighborhood_month")
print("Rows:", t1.count())
t1.selectExpr("min(month) AS first_month", "max(month) AS last_month",
              "count(DISTINCT neighborhood) AS neighborhoods", "sum(request_count) AS total_requests").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Close time by request type and department
# MAGIC Average **and** median days to close. The data is skewed by a few very old requests,
# MAGIC so the median is the "typical" experience and the average shows the long tail.
# MAGIC Only requests the city marks CLOSED are used, so no request is counted here and in the open backlog.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.gold_close_time_by_type
COMMENT 'Average and median days_to_close per request type and department (closed requests only)'
AS
SELECT
  sr_type_desc                         AS request_type,
  dept_name                            AS department,
  COUNT(*)                             AS closed_requests,
  ROUND(AVG(days_to_close), 2)         AS avg_days_to_close,
  MEDIAN(days_to_close)                AS median_days_to_close
FROM {SILVER}
WHERE is_closed AND days_to_close IS NOT NULL   -- 217 OPEN rows carry a close date (likely reopened); exclude them
GROUP BY sr_type_desc, dept_name
""")

t2 = spark.table(f"{GOLD}.gold_close_time_by_type")
print("Rows:", t2.count())
print("Top 10 request types by volume:")
t2.orderBy("closed_requests", ascending=False).show(10, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Open backlog by request type and age bucket
# MAGIC Age = days from created date to today. Buckets make it easy to spot work that has been waiting too long.
# MAGIC `snapshot_date` records which day the backlog was measured.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {GOLD}.gold_open_backlog
COMMENT 'Open 311 requests by type, department and age bucket, as of snapshot_date'
AS
WITH open_requests AS (
  SELECT
    sr_type_desc AS request_type,
    dept_name    AS department,
    DATEDIFF(CURRENT_DATE(), created_date) AS age_days
  FROM {SILVER}
  WHERE NOT is_closed
)
SELECT
  request_type,
  department,
  CASE
    WHEN age_days <= 7   THEN '1) 0-7 days'
    WHEN age_days <= 30  THEN '2) 8-30 days'
    WHEN age_days <= 90  THEN '3) 31-90 days'
    WHEN age_days <= 365 THEN '4) 91-365 days'
    ELSE                      '5) over 1 year'
  END                       AS age_bucket,
  COUNT(*)                  AS open_requests,
  CURRENT_DATE()            AS snapshot_date
FROM open_requests
GROUP BY ALL
""")

t3 = spark.table(f"{GOLD}.gold_open_backlog")
print("Rows:", t3.count())
print("Open requests by age bucket:")
t3.groupBy("age_bucket").sum("open_requests").withColumnRenamed("sum(open_requests)", "open_requests") \
  .orderBy("age_bucket").show(truncate=False)
print("Top 10 request types in the backlog:")
t3.groupBy("request_type").sum("open_requests").withColumnRenamed("sum(open_requests)", "open_requests") \
  .orderBy("open_requests", ascending=False).show(10, truncate=40)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Cross-check gold against silver
# MAGIC Totals in gold must match silver exactly; if they don't, an aggregation dropped rows.

# COMMAND ----------

silver = spark.table(SILVER)
checks = {
    "neighborhood_month total = silver rows":
        (t1.agg({"request_count": "sum"}).first()[0], silver.count()),
    "close_time total = closed silver rows":
        (t2.agg({"closed_requests": "sum"}).first()[0], silver.filter("is_closed AND days_to_close IS NOT NULL").count()),
    "backlog total = open silver rows":
        (t3.agg({"open_requests": "sum"}).first()[0], silver.filter("NOT is_closed").count()),
}
for name, (gold_v, silver_v) in checks.items():
    print(f"{'PASS' if gold_v == silver_v else 'FAIL'}  {name}: gold={gold_v:,} silver={silver_v:,}")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "gold_requests_by_neighborhood_month": t1.count(),
    "gold_close_time_by_type": t2.count(),
    "gold_open_backlog": t3.count(),
    "checks_passed": all(g == s for g, s in checks.values()),
}))
