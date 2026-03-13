"""
experiments/covtype_experiment.py
Covtype (581K satir) icin Spark profiling + ML pipeline deneyi
"""
import sys, os, time, csv, datetime, uuid, traceback, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.utils import load_config, get_logger, load_dataset_pandas, prepare_X_y
from src.agent.pipeline_agent import PipelineAgent
from src.pipeline.pipeline_builder import PipelineBuilder
from src.evaluation.metrics import Evaluator

logger   = get_logger(__name__)
RESULTS  = "experiments/results/results.csv"
PROFILES = "experiments/results/profiles"
FIELDS   = [
    "run_id","dataset","method","fold",
    "f1_macro","accuracy","precision","recall","auc_roc",
    "rmse","mae","r2","train_time_s","n_pipeline_stages","timestamp",
]

def append_result(row):
    exists = os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})

def spark_profile(csv_path):
    """Spark ile covtype profili cikar. Spark yoksa pandas fallback."""
    try:
        from pyspark.sql import SparkSession, functions as F
        t0 = time.perf_counter()
        spark = (SparkSession.builder
                 .appName("covtype_profile")
                 .config("spark.driver.memory", "3g")
                 .config("spark.sql.shuffle.partitions", "8")
                 .config("spark.ui.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")

        df    = spark.read.csv(csv_path, header=True, inferSchema=True)
        n_rows = df.count()
        feat_cols = [c for c in df.columns if c != "cover_type"]
        n_feat    = len(feat_cols)

        # Eksik deger
        total = n_rows * n_feat
        miss  = sum(df.filter(F.col(c).isNull()).count() for c in feat_cols)
        miss_ratio = miss / total if total else 0

        # Class imbalance
        vc = df.groupBy("cover_type").count().orderBy("count").collect()
        total_y = sum(r["count"] for r in vc)
        ci = vc[0]["count"] / total_y if total_y else 0.5

        # Feature variance (numerik kolonlar)
        num_cols = [f.name for f in df.schema.fields
                    if str(f.dataType) in
                    ("DoubleType()","IntegerType()","LongType()","FloatType()")
                    and f.name != "cover_type"]
        desc = df.select(num_cols).describe().collect()
        stds = []
        for row in desc:
            if row[0] == "stddev":
                for c in num_cols:
                    try: stds.append(float(row[c] or 0))
                    except: pass
        fvar = float(np.mean([s**2 for s in stds])) if stds else 0.0

        t_spark = round(time.perf_counter() - t0, 3)
        profile = {
            "dataset_name":      "covtype",
            "task_type":         "classification",
            "n_rows":            n_rows,
            "n_features":        n_feat,
            "missing_ratio":     round(miss_ratio, 4),
            "categorical_ratio": 0.0,
            "class_imbalance":   round(ci, 4),
            "feature_variance":  round(fvar, 4),
            "n_classes":         len(vc),
        }
        spark.stop()
        logger.info(f"  Spark profiling: {t_spark}s")
        return t_spark, profile

    except Exception as e:
        logger.warning(f"  Spark hatasi ({e}), pandas fallback")
        import pandas as pd
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path)
        feat = df.drop(columns=["cover_type"])
        vc   = df["cover_type"].value_counts(normalize=True)
        profile = {
            "dataset_name":      "covtype",
            "task_type":         "classification",
            "n_rows":            len(df),
            "n_features":        len(feat.columns),
            "missing_ratio":     round(float(feat.isnull().mean().mean()), 4),
            "categorical_ratio": 0.0,
            "class_imbalance":   round(float(vc.min()), 4),
            "feature_variance":  round(float(feat.var().mean()), 4),
            "n_classes":         int(df["cover_type"].nunique()),
        }
        return round(time.perf_counter()-t0, 3), profile

def main():
    logger.info("="*55)
    logger.info("Covtype Deneyi (581K satirlik gercek buyuk veri)")
    logger.info("="*55)

    CSV_PATH = "data/raw/covtype.csv"

    # 1. Spark profiling
    logger.info("\nAdim 1: Spark profiling...")
    t_spark, profile = spark_profile(CSV_PATH)
    profile_path = os.path.join(PROFILES, "covtype_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logger.info(f"  Profil kaydedildi: {profile_path}")
    logger.info(f"  {profile}")

    # 2. Veri yukle
    logger.info("\nAdim 2: Veri yukleniyor...")
    df   = load_dataset_pandas(CSV_PATH)
    X, y = prepare_X_y(df, "cover_type")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    evaluator = Evaluator()
    cfg       = load_config("configs/experiment_config.yaml")
    agent     = PipelineAgent.from_env(
        model_name=cfg.get("agent",{}).get("model","llama-3.1-8b-instant"),
        log_path=cfg["output"]["agent_log"])
    builder = PipelineBuilder()

    # 3. Metodlar
    logger.info("\nAdim 3: ML deneyleri...")

    # B3 icin 50K sample — 581K * GradientBoosting cok yavash
    sample_idx = np.random.default_rng(42).choice(
        len(X_tr), size=min(50_000, len(X_tr)), replace=False)
    X_tr_sample = X_tr[sample_idx]
    y_tr_sample = y_tr[sample_idx]

    methods = [
        ("B1_manual", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", LogisticRegression(max_iter=500, random_state=42))])),
        ("B2_default", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1))])),
        ("B3_heuristic", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", GradientBoostingClassifier(
                n_estimators=100, random_state=42))])),
    ]

    for method_name, pipeline in methods:
        use_sample = method_name == "B3_heuristic"
        Xtr = X_tr_sample if use_sample else X_tr
        ytr = y_tr_sample if use_sample else y_tr
        note = "(50K sample)" if use_sample else "(full 465K)"
        logger.info(f"  {method_name} {note}...")
        try:
            t0 = time.perf_counter()
            pipeline.fit(Xtr, ytr)
            tt = round(time.perf_counter()-t0, 3)
            m  = evaluator.evaluate(pipeline, X_te, y_te, "classification")
            logger.info(f"  F1={m.get('f1_macro')}  train={tt}s")
            append_result({
                "run_id": str(uuid.uuid4()),
                "dataset": "covtype",
                "method": method_name,
                "fold": None,
                "train_time_s": tt,
                "n_pipeline_stages": len(pipeline.steps),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **m
            })
        except Exception as e:
            logger.error(f"  HATA {method_name}: {e}")
            traceback.print_exc()

    # Agent — full profile
    logger.info("  agent_full...")
    try:
        spec     = agent.generate_pipeline(profile, "full_profile")
        pipeline = builder.build(spec)
        logger.info(f"  Agent modeli: {spec['model']}")
        # Buyuk veri icin sample
        use_s = spec["model"] in ("GradientBoostingClassifier",
                                   "GradientBoostingRegressor")
        Xtr = X_tr_sample if use_s else X_tr
        ytr = y_tr_sample if use_s else y_tr
        note = "(50K sample)" if use_s else "(full 465K)"
        logger.info(f"  Egitim {note}...")
        t0 = time.perf_counter()
        pipeline.fit(Xtr, ytr)
        tt = round(time.perf_counter()-t0, 3)
        m  = evaluator.evaluate(pipeline, X_te, y_te, "classification")
        logger.info(f"  F1={m.get('f1_macro')}  train={tt}s")
        append_result({
            "run_id": str(uuid.uuid4()),
            "dataset": "covtype",
            "method": "agent_full",
            "fold": None,
            "train_time_s": tt,
            "n_pipeline_stages": len(pipeline.steps),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **m
        })
    except Exception as e:
        logger.error(f"  HATA agent_full: {e}")
        traceback.print_exc()

    logger.info(f"\nTamamlandi. Sonuclar -> {RESULTS}")
    logger.info(f"Spark profiling suresi: {t_spark}s")

if __name__ == "__main__":
    main()