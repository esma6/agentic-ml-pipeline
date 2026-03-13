#!/usr/bin/env python3
"""
notebooks/03_analysis.py
--------------------------
Complete analysis pipeline:
  1. Load results.csv
  2. Compute summary table (Table 3 in the paper)
  3. Wilcoxon signed-rank tests vs each baseline
  4. Load agent_log.jsonl → Agent Decision Table (Table 4)
  5. Ablation results table (Table 5)
  6. Save all tables as CSV + print to console

Run after both experiment_runner.py and ablation_runner.py complete.

Usage:
    python notebooks/03_analysis.py
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os

RESULTS_PATH  = "experiments/results/results.csv"
LOG_PATH      = "experiments/results/agent_log.jsonl"
OUTPUT_DIR    = "experiments/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and preview results
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 1: Loading results")
print("="*70)

results = pd.read_csv(RESULTS_PATH)
print(f"Loaded {len(results)} rows from {RESULTS_PATH}")
print(f"Methods found: {sorted(results['method'].unique())}")
print(f"Datasets found: {sorted(results['dataset'].unique())}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Summary Table (Table 3)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 2: Main Results Table (Table 3)")
print("="*70)

def primary_metric(row):
    """Returns the primary metric value for this row (F1 or RMSE)."""
    if row["dataset"] == "wine_quality":
        return row["rmse"]
    return row["f1_macro"]

results["primary"] = results.apply(primary_metric, axis=1)

# Aggregate: mean (and std for CV datasets)
summary = (
    results
    .groupby(["dataset", "method"])["primary"]
    .agg(["mean", "std", "count"])
    .round(4)
    .reset_index()
)
summary.columns = ["dataset", "method", "mean", "std", "n_folds"]

# Pivot to wide format
pivot = summary.pivot(index="dataset", columns="method", values="mean")

# Compute delta of agent vs best baseline
baseline_cols = [c for c in pivot.columns if c.startswith("B")]
if "agent_full" in pivot.columns and baseline_cols:
    best_baseline = pivot[baseline_cols].max(axis=1)
    pivot["delta_vs_best_BL"] = (pivot["agent_full"] - best_baseline).round(4)

print("\nMain Results (mean primary metric):")
print(pivot.to_string())

pivot.to_csv(os.path.join(OUTPUT_DIR, "table3_main_results.csv"))
print(f"\nSaved → {OUTPUT_DIR}/table3_main_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Wilcoxon Signed-Rank Tests
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 3: Statistical Significance — Wilcoxon Signed-Rank Tests")
print("="*70)

classification_datasets = [
    d for d in results["dataset"].unique()
    if d != "wine_quality"
]

agent_results = results[results["method"] == "agent_full"].copy()
wilcoxon_rows = []

for baseline in ["B1_manual", "B2_default", "B3_heuristic"]:
    bl_results = results[results["method"] == baseline].copy()
    
    # Collect per-dataset (or per-fold) primary metric pairs
    agent_scores = []
    bl_scores    = []

    for dataset in classification_datasets:
        a_subset = agent_results[agent_results["dataset"] == dataset]["f1_macro"].dropna().values
        b_subset = bl_results[bl_results["dataset"] == dataset]["f1_macro"].dropna().values

        # Use per-fold scores for CV datasets; single score otherwise
        n = min(len(a_subset), len(b_subset))
        if n > 0:
            agent_scores.extend(a_subset[:n].tolist())
            bl_scores.extend(b_subset[:n].tolist())

    if len(agent_scores) < 2:
        print(f"  {baseline}: Not enough data points for Wilcoxon test.")
        continue

    differences = np.array(agent_scores) - np.array(bl_scores)

    # Wilcoxon requires non-zero differences
    nonzero_diffs = differences[differences != 0]
    if len(nonzero_diffs) < 2:
        print(f"  {baseline}: All differences are zero — cannot run Wilcoxon.")
        continue

    stat, p = wilcoxon(nonzero_diffs, alternative="two-sided")

    # Cohen's d on the differences
    cohen_d = differences.mean() / (differences.std() + 1e-10)

    sig = "*** p<0.001" if p < 0.001 else ("** p<0.01" if p < 0.01 else
          ("* p<0.05" if p < 0.05 else "ns"))

    print(f"  Agent vs {baseline:15s}: W={stat:.1f}, p={p:.4f} {sig}, "
          f"d={cohen_d:.3f}, n={len(nonzero_diffs)}")

    wilcoxon_rows.append({
        "comparison":   f"agent_full vs {baseline}",
        "W":            stat,
        "p_value":      round(p, 4),
        "significance": sig,
        "cohens_d":     round(cohen_d, 3),
        "n":            len(nonzero_diffs),
    })

if wilcoxon_rows:
    pd.DataFrame(wilcoxon_rows).to_csv(
        os.path.join(OUTPUT_DIR, "wilcoxon_tests.csv"), index=False
    )
    print(f"\nSaved → {OUTPUT_DIR}/wilcoxon_tests.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Agent Decision Table (Table 4 — tests H2)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 4: Agent Decision Log Analysis (Table 4 — H2 test)")
print("="*70)

entries = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
else:
    print(f"  WARNING: {LOG_PATH} not found — skipping agent decision analysis.")
    entries = []

if entries:
    log_df = pd.json_normalize(entries, sep="_")

    # Filter to full_profile condition only
    full_cond = log_df[log_df.get("condition", log_df.get("spec_prompt_condition", "")) == "full_profile"].copy() if "condition" in log_df.columns else log_df.copy()

    if len(full_cond) > 0:
        decision_cols = [c for c in [
            "dataset",
            "profile_categorical_ratio",
            "profile_class_imbalance",
            "profile_n_features",
            "spec_model",
            "spec_preprocessing",
            "spec_reasoning",
        ] if c in full_cond.columns]

        decision_table = full_cond[decision_cols].drop_duplicates(subset=["dataset"]) if "dataset" in decision_cols else full_cond[decision_cols]
        print("\nAgent Decision Table (full_profile condition):")
        print(decision_table.to_string(index=False))

        # Model diversity index
        if "spec_model" in full_cond.columns:
            n_distinct = full_cond["spec_model"].nunique()
            print(f"\nModel diversity index: {n_distinct}/{len(full_cond['dataset'].unique())} distinct model families")
            print("H2 support: STRONG" if n_distinct >= 3 else "H2 support: WEAK")

        decision_table.to_csv(
            os.path.join(OUTPUT_DIR, "table4_agent_decisions.csv"), index=False
        )
        print(f"\nSaved → {OUTPUT_DIR}/table4_agent_decisions.csv")
    else:
        print("  No full_profile entries found in log.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ablation Results Table (Table 5 — tests H3)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("STEP 5: Ablation Results Table (Table 5 — H3 test)")
print("="*70)

ablation_methods = ["agent_full", "agent_no_profile", "agent_model_only"]
ablation_data = results[results["method"].isin(ablation_methods)].copy()

if len(ablation_data) > 0:
    ablation_summary = (
        ablation_data
        .groupby(["dataset", "method"])["primary"]
        .mean()
        .round(4)
        .unstack()
    )

    # Compute deltas
    if "agent_full" in ablation_summary.columns:
        if "agent_no_profile" in ablation_summary.columns:
            ablation_summary["delta_A1"] = (
                ablation_summary["agent_full"] - ablation_summary["agent_no_profile"]
            ).round(4)
        if "agent_model_only" in ablation_summary.columns:
            ablation_summary["delta_A2"] = (
                ablation_summary["agent_full"] - ablation_summary["agent_model_only"]
            ).round(4)

    print("\nAblation Results:")
    print(ablation_summary.to_string())

    ablation_summary.to_csv(os.path.join(OUTPUT_DIR, "table5_ablation.csv"))
    print(f"\nSaved → {OUTPUT_DIR}/table5_ablation.csv")

    # Quick H3 verdict
    if "delta_A1" in ablation_summary.columns:
        mean_delta_a1 = ablation_summary["delta_A1"].mean()
        print(f"\nMean Delta A1 (profile value): {mean_delta_a1:.4f}")
        print("H3 support: CONFIRMED" if mean_delta_a1 > 0.005 else
              "H3 support: WEAK or REJECTED")
else:
    print("  No ablation data found — run ablation_runner.py first.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary printout
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("Output files:")
for fname in ["table3_main_results.csv", "wilcoxon_tests.csv",
              "table4_agent_decisions.csv", "table5_ablation.csv"]:
    fpath = os.path.join(OUTPUT_DIR, fname)
    status = "✓" if os.path.exists(fpath) else "✗ (not generated)"
    print(f"  {status}  {fpath}")
