"""
src/profiler/spark_profiler.py
-------------------------------
Distributed dataset profiling using Apache Spark.

Extracts statistical meta-features from a raw dataset in a single
distributed pass. Never exposes raw data to downstream components —
returns a compact profile dict only.

Usage:
    from src.profiler.spark_profiler import DatasetProfiler
    profiler = DatasetProfiler(spark)
    profile  = profiler.profile("data/raw/adult_income.csv",
                                 target_col="income",
                                 task_type="classification",
                                 dataset_name="adult_income")
"""

import json
import datetime
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType, StringType, IntegerType, LongType, DoubleType, FloatType

logger = logging.getLogger(__name__)


class DatasetProfiler:
    """
    Extracts statistical meta-features from a dataset using Apache Spark.
    Returns a profile dict matching Contract 1 schema (see docs/contracts.md).

    All computations use Spark DataFrame operations so they scale to
    datasets larger than single-machine memory.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    # ── Public API ──────────────────────────────────────────────────────────

    def profile(self,
                path: str,
                target_col: str,
                task_type: str,
                dataset_name: str) -> dict:
        """
        Main entry point. Reads dataset, computes all meta-features,
        returns a profile dict.

        Args:
            path:         Path to CSV or Parquet file.
            target_col:   Name of the target / label column.
            task_type:    'classification' or 'regression'.
            dataset_name: Human-readable identifier (used in result logs).

        Returns:
            dict matching Contract 1 schema.
        """
        logger.info(f"Profiling [{dataset_name}] from {path}")
        df = self._load(path)

        # Drop rows where target is null (would corrupt labels)
        df = df.dropna(subset=[target_col])

        feature_cols = [c for c in df.columns if c != target_col]

        profile = {
            "dataset_name":       dataset_name,
            "task_type":          task_type,
            "n_rows":             self._count_rows(df),
            "n_features":         len(feature_cols),
            "missing_ratio":      self._missing_ratio(df, feature_cols),
            "categorical_ratio":  self._categorical_ratio(df, feature_cols),
            "class_imbalance":    self._class_imbalance(df, target_col, task_type),
            "feature_variance":   self._mean_feature_variance(df, feature_cols),
            "n_classes":          self._n_classes(df, target_col, task_type),
            "timestamp":          datetime.datetime.utcnow().isoformat() + "Z",
            "spark_version":      self.spark.version,
        }

        logger.info(f"Profile computed: {json.dumps(profile, indent=2)}")
        return profile

    def save_profile(self, profile: dict, output_path: str) -> None:
        """Writes profile JSON to disk for reproducibility logging."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Profile saved → {output_path}")

    # ── Private methods ──────────────────────────────────────────────────────

    def _load(self, path: str) -> DataFrame:
        """Infers format from extension. Supports .csv and .parquet."""
        if path.endswith(".parquet"):
            return self.spark.read.parquet(path)
        return self.spark.read.csv(path, header=True, inferSchema=True)

    def _count_rows(self, df: DataFrame) -> int:
        return df.count()

    def _missing_ratio(self, df: DataFrame, feature_cols: list) -> float:
        """
        Fraction of all feature cells that contain null / NaN.
        Uses a single distributed aggregation pass over feature columns only
        (excludes the target column).
        """
        if not feature_cols:
            return 0.0

        total_cells = df.count() * len(feature_cols)
        if total_cells == 0:
            return 0.0

        null_counts = df.select([
            F.count(F.when(F.col(c).isNull(), c)).alias(c)
            for c in feature_cols
        ]).collect()[0]

        total_nulls = sum(null_counts.asDict().values())
        return round(total_nulls / total_cells, 4)

    def _categorical_ratio(self, df: DataFrame, feature_cols: list) -> float:
        """
        Fraction of feature columns whose Spark inferred type is StringType.
        Numeric columns encoded as integers (e.g., binary 0/1) are NOT
        counted as categorical — use this ratio as a proxy, not an exact count.
        """
        if not feature_cols:
            return 0.0

        feature_types = {f.name: f.dataType for f in df.schema.fields
                         if f.name in feature_cols}
        n_categorical = sum(
            1 for dtype in feature_types.values()
            if isinstance(dtype, StringType)
        )
        return round(n_categorical / len(feature_cols), 4)

    def _class_imbalance(self, df: DataFrame,
                         target_col: str,
                         task_type: str) -> float:
        """
        Minority class fraction for classification tasks.
        Returns None for regression tasks.
        Value close to 0.5 = balanced; close to 0.0 = very imbalanced.
        """
        if task_type != "classification":
            return None

        counts = (
            df.groupBy(target_col)
              .count()
              .orderBy("count", ascending=True)
              .collect()
        )
        if not counts:
            return None

        minority_count = counts[0]["count"]
        total          = df.count()
        return round(minority_count / total, 4)

    def _mean_feature_variance(self, df: DataFrame, feature_cols: list) -> float:
        """
        Mean variance across all NUMERIC feature columns.
        Non-numeric (String) columns are excluded.
        Runs as a single Spark aggregation pass.
        """
        numeric_types = (NumericType, IntegerType, LongType,
                         DoubleType, FloatType)
        numeric_cols = [
            f.name for f in df.schema.fields
            if f.name in feature_cols and isinstance(f.dataType, numeric_types)
        ]
        if not numeric_cols:
            return 0.0

        agg_exprs = [F.variance(F.col(c)).alias(c) for c in numeric_cols]
        variances  = df.select(agg_exprs).collect()[0].asDict()
        vals = [v for v in variances.values() if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _n_classes(self, df: DataFrame,
                   target_col: str,
                   task_type: str):
        """Number of distinct target classes (None for regression)."""
        if task_type != "classification":
            return None
        return df.select(target_col).distinct().count()
