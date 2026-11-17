# Databricks notebook source
# MAGIC %md
# MAGIC # 04_model — Predict days_to_close, tracked in MLflow
# MAGIC
# MAGIC **Question:** when a resident files a request, how many days will it take to close?
# MAGIC
# MAGIC - Data: closed requests in `workspace.cincy311.silver_311`
# MAGIC - Features: request type, department, neighborhood, month, day of week
# MAGIC - Metric: **MAE** (mean absolute error, in days) on a held-out test set
# MAGIC - Baseline: median days per request type → then a scikit-learn gradient-boosted model
# MAGIC - Tracking: MLflow runs (params, metrics, model) + registration in Unity Catalog

# COMMAND ----------

# MAGIC %md
# MAGIC ### 0. Install ML libraries
# MAGIC Serverless notebooks start from a slim environment without MLflow and with an old scikit-learn (1.1.1),
# MAGIC so we install MLflow and pin scikit-learn 1.5.2 with `%pip`, then restart Python to pick them up.

# COMMAND ----------

# MAGIC %pip install -q "mlflow-skinny[databricks]" "scikit-learn==1.5.2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow, sklearn, numpy as np, pandas as pd
print("mlflow", mlflow.__version__, "| scikit-learn", sklearn.__version__)

SILVER = "workspace.cincy311.silver_311"
UC_MODEL_NAME = "workspace.cincy311.days_to_close_model"
FEATURES = ["request_type", "department", "neighborhood", "month", "day_of_week"]
TARGET = "days_to_close"
SEED = 42

me = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{me}/cincy311_days_to_close")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Build the training data
# MAGIC Only requests the city marks CLOSED have a real answer to learn from. We take a fixed-seed sample of about
# MAGIC 1 million rows so scikit-learn fits comfortably in memory on one machine; the seed makes it repeatable.

# COMMAND ----------

closed = spark.sql(f"""
  SELECT sr_type_desc            AS request_type,
         dept_name               AS department,
         neighborhood,
         month(created_date)     AS month,
         created_day_of_week     AS day_of_week,
         days_to_close
  FROM {SILVER}
  WHERE is_closed AND days_to_close IS NOT NULL
""")
total_closed = closed.count()
pdf = closed.sample(fraction=0.45, seed=SEED).toPandas()
print(f"Closed requests available: {total_closed:,}")
print(f"Sampled for modeling:      {len(pdf):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Hold out a test set
# MAGIC 80% of rows train the models and 20% are hidden until the end. Both models are scored on the
# MAGIC same unseen rows, so the comparison is fair.

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(pdf, test_size=0.2, random_state=SEED)
X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]
print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
print(f"Test set target: median={y_test.median():.0f} days, mean={y_test.mean():.2f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Baseline: median days per request type
# MAGIC The simplest sensible guess: "this kind of request usually takes N days."
# MAGIC Any real model must beat this, or it is not worth the complexity. Unseen types fall back to the overall median.

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, median_absolute_error

type_median = train.groupby("request_type")[TARGET].median()
global_median = float(y_train.median())
baseline_pred = X_test["request_type"].map(type_median).fillna(global_median)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_medae = median_absolute_error(y_test, baseline_pred)

with mlflow.start_run(run_name="baseline_median_by_type"):
    mlflow.log_params({"model": "median_per_request_type", "train_rows": len(train),
                       "test_rows": len(test), "fallback_median": global_median})
    mlflow.log_metrics({"test_mae": baseline_mae, "test_median_ae": baseline_medae})

print(f"Baseline MAE:              {baseline_mae:.2f} days")
print(f"Baseline median abs error: {baseline_medae:.2f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. scikit-learn model: gradient-boosted trees
# MAGIC `OrdinalEncoder` turns text categories into numbers; there are 1,000+ request types, so rare ones are
# MAGIC grouped (`max_categories=250`) because the trees handle at most 255 categories per feature.
# MAGIC `loss="absolute_error"` makes the model optimise MAE directly, which suits this skewed target.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

CAT = ["request_type", "department", "neighborhood", "day_of_week"]
NUM = ["month"]
params = {"loss": "absolute_error", "max_iter": 300, "learning_rate": 0.1,
          "max_leaf_nodes": 63, "max_categories": 250, "random_state": SEED}

pre = ColumnTransformer([
    ("cat", OrdinalEncoder(max_categories=params["max_categories"], handle_unknown="use_encoded_value",
                           unknown_value=np.nan, encoded_missing_value=np.nan), CAT),
    ("num", "passthrough", NUM),
])
model = Pipeline([
    ("prep", pre),
    ("gbr", HistGradientBoostingRegressor(
        loss=params["loss"], max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], random_state=SEED,
        categorical_features=list(range(len(CAT))),
    )),
])
model.fit(X_train, y_train)
pred = np.clip(model.predict(X_test), 0, None)   # days can't be negative

model_mae = mean_absolute_error(y_test, pred)
model_medae = median_absolute_error(y_test, pred)
print(f"Model MAE:              {model_mae:.2f} days")
print(f"Model median abs error: {model_medae:.2f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Log the model run in MLflow
# MAGIC MLflow stores the parameters, the test metrics, and the trained pipeline with an input/output signature,
# MAGIC so anyone can later reload exactly this model and see how it scored.

# COMMAND ----------

from mlflow.models import infer_signature

improvement = (baseline_mae - model_mae) / baseline_mae * 100
signature = infer_signature(X_test.head(1000), pred[:1000])

with mlflow.start_run(run_name="hist_gradient_boosting") as run:
    mlflow.log_params({**params, "features": ",".join(FEATURES),
                       "train_rows": len(train), "test_rows": len(test)})
    mlflow.log_metrics({"test_mae": model_mae, "test_median_ae": model_medae,
                        "baseline_test_mae": baseline_mae, "mae_improvement_pct": improvement})
    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                             input_example=X_test.head(5))
    model_run_id = run.info.run_id

print(f"MLflow run id: {model_run_id}")
print(f"MAE improvement over baseline: {improvement:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Improvement: give the model each request type's typical close time
# MAGIC The first model lost information because the 1,000+ request types were squeezed into 250 groups.
# MAGIC **Target encoding** adds one numeric column: the median days-to-close of that request type in the *training* data.
# MAGIC Rare types are smoothed toward the overall median so a type seen once can't leak its own answer.
# MAGIC The encoder lives inside the pipeline, so the saved model still takes the same raw inputs.

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin

class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    """Replace a category with its smoothed median target: (n*median + m*global) / (n + m)."""

    def __init__(self, smoothing=20):
        self.smoothing = smoothing

    def fit(self, X, y):
        col = pd.Series(np.asarray(X).ravel())
        y = pd.Series(np.asarray(y, dtype=float))
        self.global_median_ = float(y.median())
        stats = y.groupby(col.values).agg(["median", "count"])
        m = self.smoothing
        self.mapping_ = ((stats["count"] * stats["median"] + m * self.global_median_) / (stats["count"] + m)).to_dict()
        return self

    def transform(self, X):
        col = pd.Series(np.asarray(X).ravel())
        return col.map(self.mapping_).fillna(self.global_median_).to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["request_type_median_days"])

te_params = {**params, "target_encoding": "median_per_request_type", "te_smoothing": 20}
pre_te = ColumnTransformer([
    ("te", MedianTargetEncoder(smoothing=te_params["te_smoothing"]), ["request_type"]),
    ("cat", OrdinalEncoder(max_categories=params["max_categories"], handle_unknown="use_encoded_value",
                           unknown_value=np.nan, encoded_missing_value=np.nan), CAT),
    ("num", "passthrough", NUM),
])
model_te = Pipeline([
    ("prep", pre_te),
    ("gbr", HistGradientBoostingRegressor(
        loss=params["loss"], max_iter=params["max_iter"], learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"], random_state=SEED,
        categorical_features=list(range(1, 1 + len(CAT))),   # column 0 is the numeric encoding
    )),
])
model_te.fit(X_train, y_train)
pred_te = np.clip(model_te.predict(X_test), 0, None)

te_mae = mean_absolute_error(y_test, pred_te)
te_medae = median_absolute_error(y_test, pred_te)
te_improvement = (baseline_mae - te_mae) / baseline_mae * 100

with mlflow.start_run(run_name="hgb_with_type_median_encoding") as run:
    mlflow.log_params({**te_params, "features": ",".join(FEATURES),
                       "train_rows": len(train), "test_rows": len(test)})
    mlflow.log_metrics({"test_mae": te_mae, "test_median_ae": te_medae,
                        "baseline_test_mae": baseline_mae, "mae_improvement_pct": te_improvement})
    mlflow.sklearn.log_model(model_te, artifact_path="model",
                             signature=infer_signature(X_test.head(1000), pred_te[:1000]),
                             input_example=X_test.head(5))
    te_run_id = run.info.run_id

print(f"Target-encoded model MAE:              {te_mae:.2f} days")
print(f"Target-encoded model median abs error: {te_medae:.2f} days")
print(f"MAE improvement over baseline:         {te_improvement:.1f}%")
print(f"MLflow run id: {te_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Compare all three on the same test set

# COMMAND ----------

comparison = pd.DataFrame({
    "model": ["baseline: median per request type", "HistGradientBoosting",
              "HistGradientBoosting + type-median encoding"],
    "test_MAE_days": [round(baseline_mae, 2), round(model_mae, 2), round(te_mae, 2)],
    "test_median_abs_error_days": [round(baseline_medae, 2), round(model_medae, 2), round(te_medae, 2)],
})
print(comparison.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8. Register the better scikit-learn model in Unity Catalog
# MAGIC Registering gives the model a governed name (`catalog.schema.model`) with versions, just like a table.
# MAGIC We register whichever of the two trained models had the lower test MAE.
# MAGIC If the workspace does not allow it, we print the error instead of failing the whole notebook.

# COMMAND ----------

best_run_id, best_name = (te_run_id, "hgb_with_type_median_encoding") if te_mae < model_mae \
    else (model_run_id, "hist_gradient_boosting")
print("Registering:", best_name)

mlflow.set_registry_uri("databricks-uc")
try:
    mv = mlflow.register_model(f"runs:/{best_run_id}/model", UC_MODEL_NAME)
    registration = f"registered {best_name} as {UC_MODEL_NAME} version {mv.version}"
except Exception as e:
    registration = f"NOT registered: {type(e).__name__}: {str(e)[:300]}"
print(registration)

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "closed_rows": total_closed, "sampled_rows": len(pdf), "train_rows": len(train), "test_rows": len(test),
    "baseline_mae": round(baseline_mae, 2), "model_mae": round(model_mae, 2),
    "baseline_median_ae": round(baseline_medae, 2), "model_median_ae": round(model_medae, 2),
    "improvement_pct": round(improvement, 1),
    "te_mae": round(te_mae, 2), "te_median_ae": round(te_medae, 2), "te_improvement_pct": round(te_improvement, 1),
    "registration": registration,
}))
