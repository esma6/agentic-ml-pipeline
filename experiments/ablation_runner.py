#!/usr/bin/env python3
"""
experiments/ablation_runner.py — Groq versiyonu
-------------------------------------------------
A1: no_profile  — sadece task_type bilgisiyle pipeline oluştur
A2: model_only  — profil var ama sadece model sec, preprocessing yok

Calistirmadan once experiment_runner.py tamamlanmis olmali
(profiles klasorunde JSON dosyalari olmali).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv, json, time, uuid, traceback, datetime
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from src.utils.utils import load_config, get_logger, load_dataset_pandas, prepare_X_y
from src.agent.pipeline_agent import PipelineAgent
from src.pipeline.pipeline_builder import PipelineBuilder
from src.evaluation.metrics import Evaluator

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

def load_profile(profiles_dir, dataset_name):
    path = os.path.join(profiles_dir, f"{dataset_name}_profile.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Profil bulunamadi: {path}\n"
            "Once experiment_runner.py calistirmaniz gerekiyor."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_one(ds_cfg, method, condition, X_tr, X_te, y_tr, y_te,
            profile, agent, builder, evaluator, fold=None):
    run_id = str(uuid.uuid4())
    try:
        spec     = agent.generate_pipeline(profile, condition=condition)
        pipeline = builder.build(spec)
        t0       = time.perf_counter()
        pipeline.fit(X_tr, y_tr)
        train_time = round(time.perf_counter() - t0, 4)
        metrics    = evaluator.evaluate(pipeline, X_te, y_te, ds_cfg["task_type"])
        primary    = f"F1={metrics.get('f1_macro')}" if ds_cfg["task_type"]=="classification" else f"RMSE={metrics.get('rmse')}"
        logger.info(f"  [{ds_cfg['name']}] {method} fold={fold} {primary} {train_time}s")
        return {"run_id":run_id,"dataset":ds_cfg["name"],"method":method,"fold":fold,
                "train_time_s":train_time,"n_pipeline_stages":len(pipeline.steps),
                "timestamp":datetime.datetime.utcnow().isoformat()+"Z", **metrics}
    except Exception as e:
        logger.error(f"  HATA [{ds_cfg['name']}] {method}: {e}")
        traceback.print_exc()
        return {"run_id":run_id,"dataset":ds_cfg["name"],"method":method,"fold":fold,
                "timestamp":datetime.datetime.utcnow().isoformat()+"Z"}

def main():
    cfg          = load_config(CFG_PATH)
    seed         = cfg["split"]["random_seed"]
    profiles_dir = cfg["output"]["profiles_dir"]
    results_path = cfg["output"]["results_csv"]

    builder   = PipelineBuilder()
    evaluator = Evaluator()

    # Groq agent
    agent = PipelineAgent.from_env(
        model_name = cfg.get("agent", {}).get("model", "llama-3.1-8b-instant"),
        log_path   = cfg["output"]["agent_log"],
    )

    init_csv(results_path)

    # Ablasyon kosullari
    ablation_conditions = [
        ("agent_no_profile", "no_profile"),   # A1: profil yok
        ("agent_model_only", "model_only"),   # A2: sadece model
    ]

    for ds_cfg in cfg["datasets"]:
        name = ds_cfg["name"]
        logger.info(f"\n{'='*55}\nDataset: {name}\n{'='*55}")

        # Kaydedilmis profili yukle (Spark tekrar calistirmadan)
        try:
            profile = load_profile(profiles_dir, name)
            logger.info(f"  Profil yuklendi: {profiles_dir}/{name}_profile.json")
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        df   = load_dataset_pandas(ds_cfg["path"])
        X, y = prepare_X_y(df, ds_cfg["target_col"])

        for method, condition in ablation_conditions:
            logger.info(f"\n  Method: {method} (condition={condition})")
            strategy = ds_cfg["eval_strategy"]

            if strategy == "single_split":
                strat = y if ds_cfg["task_type"]=="classification" else None
                X_tr,X_te,y_tr,y_te = train_test_split(
                    X,y,test_size=cfg["split"]["test_size"],
                    random_state=seed,stratify=strat)
                append_result(results_path,
                    run_one(ds_cfg,method,condition,X_tr,X_te,y_tr,y_te,
                            profile,agent,builder,evaluator))
            elif strategy == "cv5":
                kf = (StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
                      if ds_cfg["task_type"]=="classification"
                      else KFold(n_splits=5,shuffle=True,random_state=seed))
                for fi,(tr,te) in enumerate(kf.split(X,y),1):
                    append_result(results_path,
                        run_one(ds_cfg,method,condition,X[tr],X[te],y[tr],y[te],
                                profile,agent,builder,evaluator,fold=fi))

    logger.info(f"\nAblasyon tamamlandi -> {results_path}")

if __name__ == "__main__":
    main()