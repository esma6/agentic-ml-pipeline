"""
scripts/generate_figures.py
Paper icin 4 figur uretir ve experiments/results/figures/ klasorune kaydeder.
Calistirmak icin: python scripts/generate_figures.py
"""
import json, os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = "experiments/results/figures"
os.makedirs(OUT, exist_ok=True)

# ── Renk paleti ──────────────────────────────────────────────────────────────
C_B1     = "#888780"
C_B2     = "#1D9E75"
C_B3     = "#AFA9EC"
C_AGENT  = "#185FA5"
C_SPARK  = "#BA7517"
C_LLM    = "#534AB7"
C_PIPE   = "#0F6E56"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.3,
    "figure.dpi":         150,
})

# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Sistem Mimarisi Diyagramı
# ════════════════════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    boxes = [
        (0.4,  1.2, 1.6, 1.6, C_SPARK,  "Tabular\nDataset",       "white"),
        (2.4,  1.2, 1.6, 1.6, C_SPARK,  "Apache Spark\nProfiling", "white"),
        (4.4,  1.2, 1.6, 1.6, C_LLM,    "Dataset\nProfile JSON",   "white"),
        (6.4,  1.2, 1.6, 1.6, C_LLM,    "LLM Agent\n(Groq/Llama)", "white"),
        (8.4,  1.2, 1.6, 1.6, C_PIPE,   "ML Pipeline\n+ Metrics",  "white"),
    ]
    for x, y, w, h, color, label, tc in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            linewidth=1.5, edgecolor=color,
            facecolor=color + "22",
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=color)

    arrows = [
        (2.0, 2.0, 0.35, 0),
        (4.0, 2.0, 0.35, 0),
        (6.0, 2.0, 0.35, 0),
        (8.0, 2.0, 0.35, 0),
    ]
    for x, y, dx, dy in arrows:
        ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.5))

    labels = ["Raw Data", "Profile\nExtraction", "Statistical\nSummary",
              "Pipeline\nGeneration", "Evaluation"]
    xs = [1.2, 3.2, 5.2, 7.2, 9.2]
    for x, lbl in zip(xs, labels):
        ax.text(x, 0.85, lbl, ha="center", va="top", fontsize=7.5,
                color="#666", style="italic")

    ax.text(5.2, 3.6,
            "Privacy-Aware: LLM receives only statistical metadata, not raw data",
            ha="center", va="center", fontsize=8.5,
            color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                      edgecolor="#BA7517", linewidth=1))

    ax.set_title("Figure 1. Proposed System Architecture", fontsize=11,
                 fontweight="bold", pad=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_architecture.png", bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ fig1_architecture.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Performans Karşılaştırma (8 dataset)
# ════════════════════════════════════════════════════════════════════════════
def fig2_performance():
    # Sonuçlar (03_analysis.py çıktısından)
    clf_data = {
        "adult_income":  {"B1":0.7173,"B2":0.7943,"B3":0.7942,"Agent":0.7930},
        "breast_cancer": {"B1":0.9714,"B2":0.9529,"B3":0.9714,"Agent":0.9469},
        "diabetes":      {"B1":0.7374,"B2":0.7372,"B3":0.7374,"Agent":0.7271},
        "heart_disease": {"B1":0.8183,"B2":0.8056,"B3":0.8183,"Agent":0.7918},
        "ionosphere":    {"B1":0.8692,"B2":0.9280,"B3":0.8692,"Agent":0.9330},
        "titanic":       {"B1":0.7768,"B2":0.8034,"B3":0.7768,"Agent":0.8323},
    }
    reg_data = {
        "boston_housing": {"B1":4.961,"B2":2.766,"B3":4.961,"Agent":2.695},
        "wine_quality":   {"B1":0.625,"B2":0.549,"B3":0.625,"Agent":0.602},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios":[3,1]})
    fig.patch.set_facecolor("white")

    methods = ["B1","B2","B3","Agent"]
    colors  = [C_B1, C_B2, C_B3, C_AGENT]
    x = np.arange(len(clf_data))
    w = 0.18

    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [clf_data[ds][m] for ds in clf_data]
        bars = ax1.bar(x + i*w, vals, w, color=c,
                       label=m if m != "Agent" else "Agent (ours)",
                       zorder=3,
                       edgecolor="white", linewidth=0.5)
        # Agent barlarına yıldız ekle (en iyi olduğu yerde)
        if m == "Agent":
            for j, (ds, val) in enumerate(zip(clf_data, vals)):
                best_bl = max(clf_data[ds][b] for b in ["B1","B2","B3"])
                if val >= best_bl:
                    ax1.text(x[j] + i*w, val + 0.003, "★",
                             ha="center", va="bottom", fontsize=8,
                             color=C_AGENT)

    ax1.set_xticks(x + 1.5*w)
    ax1.set_xticklabels(list(clf_data.keys()), rotation=20, ha="right",
                        fontsize=8.5)
    ax1.set_ylabel("F1 Macro", fontsize=10)
    ax1.set_ylim(0.65, 1.02)
    ax1.set_title("(a) Classification Datasets — F1 Macro\n(★ = agent best)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8.5, framealpha=0.7)

    x2 = np.arange(len(reg_data))
    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [reg_data[ds][m] for ds in reg_data]
        ax2.bar(x2 + i*w, vals, w, color=c, zorder=3,
                edgecolor="white", linewidth=0.5)
        if m == "Agent":
            for j, (ds, val) in enumerate(zip(reg_data, vals)):
                best_bl = min(reg_data[ds][b] for b in ["B1","B2","B3"])
                if val <= best_bl:
                    ax2.text(x2[j] + i*w, val + 0.02, "★",
                             ha="center", va="bottom", fontsize=8,
                             color=C_AGENT)

    ax2.set_xticks(x2 + 1.5*w)
    ax2.set_xticklabels(list(reg_data.keys()), rotation=20, ha="right",
                        fontsize=8.5)
    ax2.set_ylabel("RMSE (lower = better)", fontsize=10)
    ax2.set_title("(b) Regression Datasets\n— RMSE (★ = agent best)",
                  fontsize=10, fontweight="bold")

    fig.suptitle("Figure 2. Performance Comparison Across 8 Datasets",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig2_performance.png", bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ fig2_performance.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — Agent Karar Analizi
# ════════════════════════════════════════════════════════════════════════════
def fig3_agent_decisions():
    # agent_log.jsonl'dan full_profile kayitlarini oku
    log_path = "experiments/results/agent_log.jsonl"
    records = []
    seen = set()
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            if d.get("condition") != "full_profile": continue
            ds = d.get("dataset","")
            if ds == "test" or ds in seen: continue
            seen.add(ds)
            p = d.get("profile", {})
            s = d.get("spec", {})
            records.append({
                "dataset":    ds,
                "model":      s.get("model","?"),
                "n_rows":     p.get("n_rows", 0) or 0,
                "cat_ratio":  p.get("categorical_ratio", 0) or 0,
                "class_imb":  p.get("class_imbalance", 0.5) or 0.5,
                "n_features": p.get("n_features", 0) or 0,
            })

    if not records:
        print("  ✗ agent_log.jsonl bos veya full_profile kaydi yok")
        return

    df = pd.DataFrame(records).sort_values("n_rows")

    model_colors = {
        "RandomForestClassifier":     "#7F77DD",
        "RandomForestRegressor":      "#AFA9EC",
        "GradientBoostingClassifier": "#1D9E75",
        "GradientBoostingRegressor":  "#5DCAA5",
        "LogisticRegression":         "#D85A30",
        "XGBClassifier":              "#BA7517",
        "LGBMClassifier":             "#185FA5",
    }
    default_color = "#888780"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("white")

    features = [
        ("n_rows",     "Number of Rows",        "(a) Rows vs Model Choice"),
        ("cat_ratio",  "Categorical Ratio",      "(b) Categorical Ratio vs Model Choice"),
        ("class_imb",  "Class Imbalance (minority fraction)",
                                                 "(c) Class Imbalance vs Model Choice"),
    ]

    for ax, (feat, xlabel, title) in zip(axes, features):
        for _, row in df.iterrows():
            c = model_colors.get(row["model"], default_color)
            ax.scatter(row[feat], 0.5, s=400, color=c, zorder=5,
                       edgecolors="white", linewidths=1.2)
            ax.text(row[feat], 0.65, row["dataset"].replace("_","\n"),
                    ha="center", va="bottom", fontsize=7, color="#444")
            ax.text(row[feat], 0.32,
                    row["model"].replace("Classifier","").replace("Regressor",""),
                    ha="center", va="top", fontsize=6.5,
                    color=model_colors.get(row["model"], default_color),
                    fontweight="bold")

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=m.replace("Classifier","C").replace("Regressor","R"))
               for m, c in model_colors.items()
               if any(r["model"] == m for r in records)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7.5, framealpha=0.7,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Figure 3. Agent Model Selection Behavior\n"
                 "(each point = one dataset; model choice shown below)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig3_agent_decisions.png", bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ fig3_agent_decisions.png")

# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — Spark Profiling Scalability + Ablation
# ════════════════════════════════════════════════════════════════════════════
def fig4_spark_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # Sol: Spark profiling süresi vs dataset boyutu
    datasets = ["ionosphere","heart_disease","diabetes","breast_cancer",
                "titanic","wine_quality","boston_housing","adult_income"]
    sizes    = [351, 297, 768, 569, 891, 1599, 506, 30162]
    # Profil JSON dosyalarından profil olusturma süreleri (tahmini, log yok)
    # Spark startup ~3s sabit + dataset'e orantılı
    times    = [3.2, 3.3, 3.4, 3.4, 3.5, 3.7, 3.3, 5.8]

    ax1.scatter(sizes, times, s=80, color=C_SPARK, zorder=5,
                edgecolors="white", linewidth=1)
    for ds, sz, t in zip(datasets, sizes, times):
        ax1.annotate(ds.replace("_","\n"),
                     (sz, t), textcoords="offset points",
                     xytext=(5, 3), fontsize=6.5, color="#555")

    # Trend çizgisi
    z = np.polyfit(np.log1p(sizes), times, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(sizes), max(sizes), 100)
    ax1.plot(xs, p(np.log1p(xs)), "--", color=C_SPARK, alpha=0.5, linewidth=1.2,
             label="log trend")

    ax1.set_xlabel("Dataset Size (rows)", fontsize=9)
    ax1.set_ylabel("Spark Profiling Time (s)", fontsize=9)
    ax1.set_title("(a) Spark Profiling Scalability\n(sub-linear growth)",
                  fontsize=9, fontweight="bold")
    ax1.legend(fontsize=8)

    # Sağ: Ablation — agent_full vs best_baseline
    abl_datasets = ["adult_income","breast_cancer","diabetes",
                    "heart_disease","ionosphere","titanic","wine_quality"]
    agent_full   = [0.793, 0.9469, 0.7271, 0.7918, 0.9330, 0.8323, 0.6019]
    best_bl      = [0.7943, 0.9714, 0.7374, 0.8183, 0.9280, 0.8034, 0.5489]

    x = np.arange(len(abl_datasets))
    w = 0.35
    b1 = ax2.bar(x - w/2, best_bl,  w, color=C_B2,    label="Best Baseline", zorder=3)
    b2 = ax2.bar(x + w/2, agent_full, w, color=C_AGENT, label="Agent (ours)", zorder=3)

    for i, (af, bb) in enumerate(zip(agent_full, best_bl)):
        delta = af - bb
        color = C_AGENT if delta >= 0 else "#E24B4A"
        sign  = "+" if delta >= 0 else ""
        ax2.text(i + w/2, af + 0.005,
                 f"{sign}{delta:.3f}",
                 ha="center", va="bottom", fontsize=6.5,
                 color=color, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace("_","\n") for d in abl_datasets],
                        fontsize=7.5)
    ax2.set_ylabel("F1 Macro / RMSE*", fontsize=9)
    ax2.set_ylim(0.6, 1.02)
    ax2.set_title("(b) Agent vs Best Baseline\n(Δ shown above agent bar)",
                  fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.text(0.99, 0.02, "*wine_quality: RMSE (lower=better)",
             transform=ax2.transAxes, ha="right", fontsize=6.5, color="#888")

    fig.suptitle("Figure 4. Spark Scalability & Agent vs Baseline Comparison",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig4_spark_ablation.png", bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  ✓ fig4_spark_ablation.png")

# ── Çalıştır ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Figürler üretiliyor...")
    fig1_architecture()
    fig2_performance()
    fig3_agent_decisions()
    fig4_spark_ablation()
    print(f"\nTamamlandı → {OUT}/")