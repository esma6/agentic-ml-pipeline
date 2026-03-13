"""
src/baselines/
--------------
B1ManualPipeline    — fixed non-adaptive pipeline (manual_pipeline.py)
B2DefaultPipeline   — default Random Forest, no tuning (default_pipeline.py)
B3HeuristicPipeline — deterministic rule-based selector (heuristic_pipeline.py)

All three expose the same interface:
    pipeline = BaselineClass().build(task_type, profile=None)

This uniform interface lets the experiment runner treat all methods identically.
"""

# ─────────────────────────────────────────────────────────────────────────────
# manual_pipeline.py  (B1)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.decomposition import PCA


class B1ManualPipeline:
    """
    Baseline 1: Manually designed fixed pipeline.

    Represents a competent practitioner applying a sensible generic solution
    without inspecting dataset characteristics. Applied identically to every
    dataset — there is NO adaptation.

    Classification : Imputer → StandardScaler → LogisticRegression(C=1.0)
    Regression     : Imputer → StandardScaler → Ridge(alpha=1.0)
    """

    def build(self, task_type: str, profile: dict = None) -> Pipeline:
        if task_type == "classification":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   LogisticRegression(
                                C=1.0,
                                max_iter=1000,
                                random_state=42,
                            )),
            ])
        else:  # regression
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   Ridge(alpha=1.0)),
            ])


# ─────────────────────────────────────────────────────────────────────────────
# default_pipeline.py  (B2)
# ─────────────────────────────────────────────────────────────────────────────

class B2DefaultPipeline:
    """
    Baseline 2: Default Random Forest with no tuning.

    Mirrors the behaviour of Spark MLlib RandomForestClassifier with
    all-default parameters. Represents the 'press run and hope' baseline.
    Applied identically to every dataset — no adaptation.

    Classification : Imputer → RandomForestClassifier(n_estimators=100, defaults)
    Regression     : Imputer → RandomForestRegressor(n_estimators=100, defaults)
    """

    def build(self, task_type: str, profile: dict = None) -> Pipeline:
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   model),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# heuristic_pipeline.py  (B3)
# ─────────────────────────────────────────────────────────────────────────────

class B3HeuristicPipeline:
    """
    Baseline 3: Rule-based deterministic pipeline selector.

    Uses the SAME profile statistics available to the LLM agent, but
    applies explicit if-then rules instead of LLM reasoning.

    SCIENTIFIC ROLE: This is the most important baseline.
    If the LLM agent does not outperform B3, its value reduces to
    zero — a lookup table would suffice.

    Rules applied in order (classification):
      1. categorical_ratio > 0.30  → RF + median imputer
         (many categorical features → tree model, skip scaling)
      2. n_features > 20           → PCA(0.95) + LR + StandardScaler
         (high-dimensional → dimensionality reduction)
      3. class_imbalance < 0.10    → GBT + StandardScaler
         (severe imbalance → gradient boosting handles it better)
      4. else                      → LR + StandardScaler
         (generic safe choice)

    Regression: always Ridge + StandardScaler (no profile-based branching
    needed for the single regression dataset in this study).
    """

    def build(self, task_type: str, profile: dict) -> Pipeline:
        if profile is None:
            raise ValueError("B3HeuristicPipeline requires a profile dict.")

        if task_type == "regression":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   Ridge(alpha=1.0)),
            ])

        # ── Classification rule tree ──────────────────────────────────────────
        cat_ratio      = float(profile.get("categorical_ratio", 0.0))
        n_features     = int(profile.get("n_features", 0))
        class_imbalance= float(profile.get("class_imbalance") or 0.5)

        if cat_ratio > 0.30:
            # Many categorical features → tree model handles mixed types
            return Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("model",   RandomForestClassifier(
                                n_estimators=200, random_state=42
                            )),
            ])

        elif n_features > 20:
            # High-dimensional numeric → reduce then linear model
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("pca",     PCA(n_components=0.95)),
                ("model",   LogisticRegression(
                                C=1.0, max_iter=1000, random_state=42
                            )),
            ])

        elif class_imbalance < 0.10:
            # Severe minority class → gradient boosting is more robust
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   GradientBoostingClassifier(
                                n_estimators=100, random_state=42
                            )),
            ])

        else:
            # Generic: logistic regression is a strong linear baseline
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   LogisticRegression(
                                C=1.0, max_iter=1000, random_state=42
                            )),
            ])
