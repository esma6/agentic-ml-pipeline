"""
experiments/covtype_agent_full.py
Agent'i tam 465K covtype verisi uzerinde calistirir.
Onceki agent_full satirini results.csv'den siler, yenisini ekler.
"""
import sys, os, time, csv, datetime, uuid, json, warnings, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.utils import load_config, get_logger, load_dataset_pandas, prepare_X_y
from src.agent.pipeline_agent import PipelineAgent
from src.pipeline.pipeline_builder import PipelineBuilder
from src.evaluation.metrics import Evaluator

logger  = get_logger(__name__)
RESULTS = "experiments/results/results.csv"
FIELDS  = [
    "run_id","dataset","method","fold",
    "f1_macro","accuracy","precision","recall","auc_roc",
    "rmse","mae","r2","train_time_s","n_pipeline_stages","timestamp",
]

def main():
    logger.info("Covtype — Agent full 465K")

    # Onceki covtype agent_full satirini sil
    if os.path.exists(RESULTS):
        with open(RESULTS, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows
                if not (r["dataset"]=="covtype" and r["method"]=="agent_full")]
        removed = len(rows) - len(kept)
        with open(RESULTS, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(kept)
        logger.info(f"  Eski agent_full satiri silindi ({removed} satir)")

    # Profil yukle
    profile_path = "experiments/results/profiles/covtype_profile.json"
    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)
    logger.info(f"  Profil yuklendi: {profile}")

    # Veri yukle
    logger.info("  Veri yukleniyor...")
    df = load_dataset_pandas("data/raw/covtype.csv")
    X, y = prepare_X_y(df, "cover_type")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"  Train: {len(X_tr):,}  Test: {len(X_te):,}")

    # Agent pipeline uret
    cfg     = load_config("configs/experiment_config.yaml")
    agent   = PipelineAgent.from_env(
        model_name=cfg.get("agent",{}).get("model","llama-3.1-8b-instant"),
        log_path=cfg["output"]["agent_log"])
    builder   = PipelineBuilder()
    evaluator = Evaluator()

    logger.info("  Agent pipeline uretiliyor...")
    spec     = agent.generate_pipeline(profile, "full_profile")
    pipeline = builder.build(spec)
    logger.info(f"  Agent modeli: {spec['model']}")
    logger.info(f"  Hiperparametreler: {spec.get('hyperparameters',{})}")
    logger.info(f"  Reasoning: {spec.get('reasoning','')}")

    # Tam 465K uzerinde egit
    logger.info("  Egitim tam 465K uzerinde basliyor...")
    t0 = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    tt = round(time.perf_counter()-t0, 3)
    m  = evaluator.evaluate(pipeline, X_te, y_te, "classification")

    logger.info(f"  F1={m.get('f1_macro')}  train={tt}s")
    logger.info(f"  Accuracy={m.get('accuracy')}  AUC={m.get('auc_roc')}")

    # Kaydet
    row = {
        "run_id": str(uuid.uuid4()),
        "dataset": "covtype",
        "method": "agent_full",
        "fold": None,
        "train_time_s": tt,
        "n_pipeline_stages": len(pipeline.steps),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **m
    }
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writerow({k: row.get(k) for k in FIELDS})

    logger.info(f"  Kaydedildi -> {RESULTS}")

    # Ozet
    logger.info("\n" + "="*50)
    logger.info("COVTYPE OZET:")
    logger.info(f"  B2_default (RF, 465K):   F1=0.9241")
    logger.info(f"  agent_full ({spec['model']}, 465K): F1={m.get('f1_macro')}")
    logger.info("="*50)

if __name__ == "__main__":
    main()