#!/usr/bin/env python3
"""
experiments/experiment_runner.py — v2
8 dataset, 4 method, XGB/LGBM destekli
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, json, time, uuid, traceback, datetime
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from src.utils.utils import load_config, get_logger, load_dataset_pandas, prepare_X_y
from src.agent.pipeline_agent import PipelineAgent
from src.pipeline.pipeline_builder import PipelineBuilder
from src.evaluation.metrics import Evaluator

# Spark profiler — opsiyonel, yoksa fallback
try:
    from src.profiler.spark_profiler import SparkProfiler
    HAS_SPARK = True
except Exception:
    HAS_SPARK = False

logger = get_logger(__name__)
CFG_PATH = "configs/experiment_config.yaml"

RESULT_FIELDS = [
    "run_id","dataset","method","fold",
    "f1_macro","accuracy","precision","recall","auc_roc",
    "rmse","mae","r2","train_time_s","n_pipeline_stages","timestamp",
]

def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path,"w",newline="",encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()

def append_result(path, row):
    full = {k: row.get(k) for k in RESULT_FIELDS}
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDS).writerow(full)

def make_pandas_profile(df, ds_cfg):
    """Spark olmadan pandas ile profil uret"""
    import pandas as pd
    target = ds_cfg["target_col"]
    feat_df = df.drop(columns=[target])
    n_rows, n_features = feat_df.shape
    cat_cols = feat_df.select_dtypes(include=["object","category"]).columns
    categorical_ratio = len(cat_cols) / n_features if n_features > 0 else 0
    missing_ratio = feat_df.isnull().mean().mean()
    num_df = feat_df.select_dtypes(include=["number"])
    feature_variance = float(num_df.var().mean()) if not num_df.empty else 0.0

    task = ds_cfg["task_type"]
    y = df[target]
    if task == "classification":
        vc = y.value_counts(normalize=True)
        class_imbalance = float(vc.min()) if len(vc) > 0 else 0.5
        n_classes = int(y.nunique())
    else:
        class_imbalance = None
        n_classes = None

    return {
        "dataset_name":        ds_cfg["name"],
        "task_type":           task,
        "n_rows":              int(n_rows),
        "n_features":          int(n_features),
        "missing_ratio":       round(float(missing_ratio), 4),
        "categorical_ratio":   round(float(categorical_ratio), 4),
        "class_imbalance":     round(float(class_imbalance), 4) if class_imbalance is not None else None,
        "feature_variance":    round(float(feature_variance), 4),
        "n_classes":           n_classes,
    }

def get_or_create_profile(ds_cfg, df, profiles_dir):
    path = os.path.join(profiles_dir, f"{ds_cfg['name']}_profile.json")
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            profile = json.load(f)
        logger.info(f"  Profil yuklendi: {path}")
        return profile

    # Spark varsa kullan, yoksa pandas fallback
    if HAS_SPARK:
        try:
            profiler = SparkProfiler()
            profile  = profiler.profile(ds_cfg["path"], ds_cfg["target_col"],
                                        ds_cfg["task_type"], ds_cfg["name"])
            profiler.stop()
        except Exception as e:
            logger.warning(f"  Spark profil hatasi ({e}), pandas fallback")
            profile = make_pandas_profile(df, ds_cfg)
    else:
        profile = make_pandas_profile(df, ds_cfg)

    os.makedirs(profiles_dir, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    logger.info(f"  Profil olusturuldu: {path}")
    return profile

# ── Baseline builder'lar ──────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_manual(task):
    if task == "classification":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scl", StandardScaler()),
                         ("mdl", LogisticRegression(max_iter=1000, random_state=42))])
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()),
                     ("mdl", Ridge())])

def build_default(task):
    if task == "classification":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("mdl", RandomForestClassifier(random_state=42))])
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("mdl", RandomForestRegressor(random_state=42))])

def build_heuristic(task, profile):
    n_rows = profile.get("n_rows", 0)
    cat_r  = profile.get("categorical_ratio", 0)
    ci     = profile.get("class_imbalance", 0.5) or 0.5

    if task == "classification":
        if n_rows < 2000:
            mdl = LogisticRegression(max_iter=1000, random_state=42)
        elif cat_r > 0.4 or ci < 0.2:
            mdl = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        if n_rows < 2000:
            mdl = Ridge()
        else:
            mdl = GradientBoostingRegressor(n_estimators=100, random_state=42)

    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()),
                     ("mdl", mdl)])

# ── run_one ──────────────────────────────────────────────────────────────────
def run_one(ds_cfg, method, pipeline_or_spec, X_tr, X_te, y_tr, y_te,
            evaluator, builder=None, fold=None):
    run_id = str(uuid.uuid4())
    try:
        if isinstance(pipeline_or_spec, dict):
            pipeline = builder.build(pipeline_or_spec)
        else:
            pipeline = pipeline_or_spec

        t0 = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        train_time = round(time.perf_counter() - t0, 4)
        metrics    = evaluator.evaluate(pipeline, X_te, y_te, ds_cfg["task_type"])
        primary    = (f"F1={metrics.get('f1_macro')}"
                      if ds_cfg["task_type"]=="classification"
                      else f"RMSE={metrics.get('rmse')}")
        logger.info(f"  [{ds_cfg['name']}] {method} fold={fold} {primary} {train_time}s")
        return {"run_id":run_id,"dataset":ds_cfg["name"],"method":method,"fold":fold,
                "train_time_s":train_time,
                "n_pipeline_stages":len(pipeline.steps),
                "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                **metrics}
    except Exception as e:
        logger.error(f"  HATA [{ds_cfg['name']}] {method}: {e}")
        traceback.print_exc()
        return {"run_id":run_id,"dataset":ds_cfg["name"],"method":method,"fold":fold,
                "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat()}

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    cfg          = load_config(CFG_PATH)
    seed         = cfg["split"]["random_seed"]
    profiles_dir = cfg["output"]["profiles_dir"]
    results_path = cfg["output"]["results_csv"]

    builder   = PipelineBuilder()
    evaluator = Evaluator()
    agent     = PipelineAgent.from_env(
        model_name = cfg.get("agent",{}).get("model","llama-3.1-8b-instant"),
        log_path   = cfg["output"]["agent_log"],
    )

    init_csv(results_path)

    for ds_cfg in cfg["datasets"]:
        name = ds_cfg["name"]
        logger.info(f"\n{'='*55}\nDataset: {name}\n{'='*55}")

        df      = load_dataset_pandas(ds_cfg["path"])
        X, y    = prepare_X_y(df, ds_cfg["target_col"])
        profile = get_or_create_profile(ds_cfg, df, profiles_dir)
        task    = ds_cfg["task_type"]
        strat   = ds_cfg["eval_strategy"]

        methods = {
            "B1_manual":    lambda: build_manual(task),
            "B2_default":   lambda: build_default(task),
            "B3_heuristic": lambda: build_heuristic(task, profile),
            "agent_full":   lambda: agent.generate_pipeline(profile, "full_profile"),
        }

        for method_name, method_fn in methods.items():
            logger.info(f"\n  Method: {method_name}")

            if strat == "single_split":
                strat_y = y if task == "classification" else None
                X_tr,X_te,y_tr,y_te = train_test_split(
                    X, y, test_size=cfg["split"]["test_size"],
                    random_state=seed, stratify=strat_y)
                obj = method_fn()
                append_result(results_path,
                    run_one(ds_cfg, method_name, obj, X_tr,X_te,y_tr,y_te,
                            evaluator, builder))

            elif strat == "cv5":
                kf = (StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                      if task == "classification"
                      else KFold(n_splits=5, shuffle=True, random_state=seed))
                for fi,(tr,te) in enumerate(kf.split(X,y), 1):
                    obj = method_fn()
                    append_result(results_path,
                        run_one(ds_cfg, method_name, obj,
                                X[tr],X[te],y[tr],y[te],
                                evaluator, builder, fold=fi))

    logger.info(f"\nTamamlandi -> {results_path}")

if __name__ == "__main__":
    main()