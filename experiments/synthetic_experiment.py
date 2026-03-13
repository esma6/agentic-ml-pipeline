"""
experiments/synthetic_experiment.py
-------------------------------------
5 farkli boyutta sentetik veri uretir:
  - Spark ile profil cikarir
  - Agent ile pipeline uretir
  - Baseline ile karsilastirir
  - Profiling suresi + ML suresi + F1 olcer

Ciktisi:
  experiments/results/synthetic_results.csv
  experiments/results/figures/fig_synthetic.png
"""
import sys, os, time, csv, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.utils import load_config, get_logger, prepare_X_y
from src.agent.pipeline_agent import PipelineAgent
from src.pipeline.pipeline_builder import PipelineBuilder
from src.evaluation.metrics import Evaluator

logger = get_logger(__name__)

OUT_CSV = "experiments/results/synthetic_results.csv"
OUT_FIG = "experiments/results/figures/fig_synthetic.png"
os.makedirs("experiments/results/figures", exist_ok=True)
os.makedirs("data/synthetic", exist_ok=True)

SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
SEED  = 42

RESULT_FIELDS = [
    "dataset", "n_rows", "size_mb",
    "spark_profiling_time_s",
    "method", "f1_macro", "train_time_s",
]

# ── Sentetik veri uret ───────────────────────────────────────────────────────
def generate_synthetic(n_rows, n_features=20, n_cat=4, seed=42):
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_features - n_cat):
        data[f"num_{i}"] = rng.normal(0, 1, n_rows)
    cats = ["A","B","C","D","E"]
    for i in range(n_cat):
        data[f"cat_{i}"] = rng.choice(cats, n_rows)
    # Gürültülü ama öğrenilebilir hedef
    nums = np.column_stack([rng.normal(0,1,n_rows)
                            for _ in range(n_features - n_cat)])
    score = nums[:, 0] * 0.5 + nums[:, 1] * 0.3 + rng.normal(0, 0.5, n_rows)
    data["target"] = (score > score.mean()).astype(int)
    df = pd.DataFrame(data)
    # %3 eksik deger
    num_cols = [c for c in df.columns if c.startswith("num_")]
    mask = rng.random((n_rows, len(num_cols))) < 0.03
    df[num_cols] = df[num_cols].where(~mask, other=np.nan)
    return df

# ── Spark profil ─────────────────────────────────────────────────────────────
def spark_profile(csv_path, dataset_name):
    try:
        from pyspark.sql import SparkSession, functions as F
        t0 = time.perf_counter()
        spark = (SparkSession.builder
                 .appName(f"profile_{dataset_name}")
                 .config("spark.driver.memory", "2g")
                 .config("spark.sql.shuffle.partitions", "4")
                 .config("spark.ui.enabled", "false")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")

        df = spark.read.csv(csv_path, header=True, inferSchema=True)
        n_rows    = df.count()
        feat_cols = [c for c in df.columns if c != "target"]
        n_feat    = len(feat_cols)
        cat_cols  = [f.name for f in df.schema.fields
                     if str(f.dataType)=="StringType()" and f.name!="target"]
        cat_ratio = len(cat_cols) / n_feat if n_feat else 0

        total = n_rows * n_feat
        miss  = sum(df.filter(F.col(c).isNull()).count() for c in feat_cols)
        miss_ratio = miss / total if total else 0

        vc = df.groupBy("target").count().orderBy("count").collect()
        ci = vc[0]["count"] / sum(r["count"] for r in vc) if len(vc)>=2 else 0.5

        num_cols = [f.name for f in df.schema.fields
                    if str(f.dataType) in
                    ("DoubleType()","IntegerType()","LongType()","FloatType()")
                    and f.name != "target"]
        desc = df.select(num_cols).describe().collect() if num_cols else []
        stds = []
        for row in desc:
            if row[0] == "stddev":
                for c in num_cols:
                    try: stds.append(float(row[c] or 0))
                    except: pass
        fvar = float(np.mean([s**2 for s in stds])) if stds else 0.0

        t_spark = round(time.perf_counter() - t0, 3)
        profile = {
            "dataset_name":      dataset_name,
            "task_type":         "classification",
            "n_rows":            n_rows,
            "n_features":        n_feat,
            "missing_ratio":     round(miss_ratio, 4),
            "categorical_ratio": round(cat_ratio, 4),
            "class_imbalance":   round(ci, 4),
            "feature_variance":  round(fvar, 4),
            "n_classes":         len(vc),
        }
        spark.stop()
        return t_spark, profile

    except Exception as e:
        logger.warning(f"Spark hatasi: {e} — pandas fallback")
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path)
        feat = df.drop(columns=["target"])
        cat_cols = feat.select_dtypes(include=["object"]).columns
        n_feat = len(feat.columns)
        vc = df["target"].value_counts()
        profile = {
            "dataset_name":      dataset_name,
            "task_type":         "classification",
            "n_rows":            len(df),
            "n_features":        n_feat,
            "missing_ratio":     round(feat.isnull().mean().mean(), 4),
            "categorical_ratio": round(len(cat_cols)/n_feat, 4),
            "class_imbalance":   round(float(vc.min()/vc.sum()), 4),
            "feature_variance":  round(float(feat.select_dtypes("number").var().mean()), 4),
            "n_classes":         int(df["target"].nunique()),
        }
        return round(time.perf_counter()-t0, 3), profile

# ── ML pipeline deneyi ───────────────────────────────────────────────────────
def run_ml(df, method, profile, agent, builder, evaluator):
    X, y = prepare_X_y(df, "target")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    if method == "B1_manual":
        pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", LogisticRegression(max_iter=500, random_state=SEED))])
    elif method == "B2_default":
        pipeline = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", RandomForestClassifier(n_estimators=100, random_state=SEED))])
    elif method == "agent_full":
        spec = agent.generate_pipeline(profile, condition="full_profile")
        pipeline = builder.build(spec)

    t0 = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    train_time = round(time.perf_counter() - t0, 3)
    metrics = evaluator.evaluate(pipeline, X_te, y_te, "classification")
    return metrics.get("f1_macro", None), train_time

# ── Ana ──────────────────────────────────────────────────────────────────────
def main():
    cfg     = load_config("configs/experiment_config.yaml")
    agent   = PipelineAgent.from_env(
        model_name=cfg.get("agent",{}).get("model","llama-3.1-8b-instant"),
        log_path=cfg["output"]["agent_log"])
    builder   = PipelineBuilder()
    evaluator = Evaluator()

    rows = []

    for n in SIZES:
        name     = f"synthetic_{n//1000}K"
        csv_path = f"data/synthetic/{name}.csv"
        logger.info(f"\n{'='*50}\n{name} ({n:,} satir)\n{'='*50}")

        # Veri uret
        logger.info("  Veri uretiliyor...")
        df = generate_synthetic(n, seed=SEED)
        df.to_csv(csv_path, index=False)
        size_mb = round(os.path.getsize(csv_path) / 1024**2, 2)
        logger.info(f"  {size_mb} MB")

        # Spark profil
        logger.info("  Spark profiling...")
        t_spark, profile = spark_profile(csv_path, name)
        logger.info(f"  Profiling: {t_spark}s")

        # Profili kaydet
        ppath = f"experiments/results/profiles/{name}_profile.json"
        with open(ppath, "w") as f:
            json.dump(profile, f, indent=2)

        # ML deneyleri (500K ve 1M icin sadece 20% sample kullan — RAM)
        df_ml = df.sample(min(n, 50_000), random_state=SEED) if n > 100_000 else df
        logger.info(f"  ML deney boyutu: {len(df_ml):,} satir")

        for method in ["B1_manual", "B2_default", "agent_full"]:
            logger.info(f"    {method}...")
            try:
                f1, t_train = run_ml(df_ml, method, profile, agent, builder, evaluator)
                logger.info(f"    F1={f1:.4f} train={t_train}s")
            except Exception as e:
                logger.error(f"    HATA: {e}")
                f1, t_train = None, None

            rows.append({
                "dataset":               name,
                "n_rows":                n,
                "size_mb":               size_mb,
                "spark_profiling_time_s": t_spark,
                "method":                method,
                "f1_macro":              round(f1, 4) if f1 else None,
                "train_time_s":          t_train,
            })

        # Gecici CSV sil
        os.remove(csv_path)

    # Kaydet
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"\nSonuclar: {OUT_CSV}")

    # Grafik
    plot(rows)

def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,
        "figure.dpi":150,
    })

    df = pd.DataFrame(rows)
    sizes = sorted(df["n_rows"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # Sol: Spark profiling suresi
    spark_times = df.groupby("n_rows")["spark_profiling_time_s"].first()
    ax1.plot(sizes, [spark_times[s] for s in sizes],
             "o-", color="#BA7517", lw=2, markersize=8,
             markerfacecolor="white", markeredgewidth=2)
    for s in sizes:
        t = spark_times[s]
        ax1.annotate(f"{t:.1f}s", (s, t),
                     textcoords="offset points", xytext=(6,4),
                     fontsize=8.5, color="#555")
    ax1.set_xlabel("Dataset Size (rows)", fontsize=10)
    ax1.set_ylabel("Spark Profiling Time (s)", fontsize=10)
    ax1.set_title("(a) Spark Profiling Scalability", fontsize=10, fontweight="bold")
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f"{int(x/1000)}K"))
    ax1.grid(True, alpha=0.3, axis="y")

    # Sag: F1 karsilastirma
    colors = {"B1_manual":"#888780","B2_default":"#1D9E75","agent_full":"#185FA5"}
    labels = {"B1_manual":"B1 Manual","B2_default":"B2 Default RF","agent_full":"Agent (ours)"}
    x = np.arange(len(sizes))
    w = 0.25
    offsets = [-w, 0, w]
    for i, method in enumerate(["B1_manual","B2_default","agent_full"]):
        sub = df[df["method"]==method].set_index("n_rows")
        vals = [sub.loc[s,"f1_macro"] if s in sub.index else 0 for s in sizes]
        ax2.bar(x + offsets[i], vals, w, color=colors[method],
                label=labels[method], zorder=3, edgecolor="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s//1000}K" for s in sizes])
    ax2.set_xlabel("Dataset Size", fontsize=10)
    ax2.set_ylabel("F1 Macro", fontsize=10)
    ax2.set_ylim(0.5, 1.0)
    ax2.set_title("(b) ML Performance on Synthetic Data\n(sampled 50K for 500K+)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Figure 5. Spark Scalability & Agent Performance on Synthetic Big Data",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Grafik: {OUT_FIG}")

if __name__ == "__main__":
    main()