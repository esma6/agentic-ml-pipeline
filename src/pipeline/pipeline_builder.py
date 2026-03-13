"""
src/pipeline/pipeline_builder.py — v2
XGBoost ve LightGBM destekli, label_encoder pipeline'dan kaldirildi.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

PREPROCESSING_MAP = {
    "imputer":            lambda: SimpleImputer(strategy="median"),
    "standard_scaler":    lambda: StandardScaler(),
    "min_max_scaler":     lambda: MinMaxScaler(),
    "one_hot_encoder":    lambda: SimpleImputer(strategy="most_frequent"),  # already encoded in utils
    "pca":                lambda: PCA(n_components=0.95),
    "variance_threshold": lambda: VarianceThreshold(threshold=0.01),
}

def _xgb_cls(**kw):
    defaults = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="logloss",
                    verbosity=0, random_state=42)
    defaults.update(kw)
    return XGBClassifier(**defaults)

def _xgb_reg(**kw):
    defaults = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                    verbosity=0, random_state=42)
    defaults.update(kw)
    return XGBRegressor(**defaults)

def _lgbm_cls(**kw):
    defaults = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                    verbose=-1, random_state=42)
    defaults.update(kw)
    return LGBMClassifier(**defaults)

def _lgbm_reg(**kw):
    defaults = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                    verbose=-1, random_state=42)
    defaults.update(kw)
    return LGBMRegressor(**defaults)

MODEL_MAP = {
    # Classification
    "LogisticRegression":          lambda **kw: LogisticRegression(max_iter=1000, random_state=42, **kw),
    "RandomForestClassifier":      lambda **kw: RandomForestClassifier(random_state=42, **kw),
    "GradientBoostingClassifier":  lambda **kw: GradientBoostingClassifier(random_state=42, **kw),
    "SVC":                         lambda **kw: SVC(probability=True, random_state=42, **kw),
    "KNeighborsClassifier":        lambda **kw: KNeighborsClassifier(**kw),
    "XGBClassifier":               _xgb_cls,
    "LGBMClassifier":              _lgbm_cls,
    # Regression
    "Ridge":                       lambda **kw: Ridge(**kw),
    "RandomForestRegressor":       lambda **kw: RandomForestRegressor(random_state=42, **kw),
    "GradientBoostingRegressor":   lambda **kw: GradientBoostingRegressor(random_state=42, **kw),
    "SVR":                         lambda **kw: SVR(**kw),
    "XGBRegressor":                _xgb_reg,
    "LGBMRegressor":               _lgbm_reg,
}

# XGB/LGBM yoksa fallback
if not HAS_XGB:
    MODEL_MAP["XGBClassifier"]  = lambda **kw: GradientBoostingClassifier(random_state=42)
    MODEL_MAP["XGBRegressor"]   = lambda **kw: GradientBoostingRegressor(random_state=42)
if not HAS_LGBM:
    MODEL_MAP["LGBMClassifier"] = lambda **kw: GradientBoostingClassifier(random_state=42)
    MODEL_MAP["LGBMRegressor"]  = lambda **kw: GradientBoostingRegressor(random_state=42)

VALID_HP = {
    "LogisticRegression":         {"C", "max_iter"},
    "RandomForestClassifier":     {"n_estimators", "max_depth"},
    "RandomForestRegressor":      {"n_estimators", "max_depth"},
    "GradientBoostingClassifier": {"n_estimators", "max_depth", "learning_rate", "subsample"},
    "GradientBoostingRegressor":  {"n_estimators", "max_depth", "learning_rate"},
    "XGBClassifier":              {"n_estimators", "max_depth", "learning_rate"},
    "XGBRegressor":               {"n_estimators", "max_depth", "learning_rate"},
    "LGBMClassifier":             {"n_estimators", "max_depth", "learning_rate", "num_leaves"},
    "LGBMRegressor":              {"n_estimators", "max_depth", "learning_rate", "num_leaves"},
    "Ridge":                      {"alpha"},
    "SVC":                        {"C"},
    "SVR":                        {"C"},
    "KNeighborsClassifier":       {"n_neighbors"},
}

class PipelineBuilder:
    def build(self, spec: dict) -> Pipeline:
        steps = []

        # Preprocessing — label_encoder'i atla (utils'de zaten encode ediliyor)
        skip = {"label_encoder"}
        for name in spec.get("preprocessing", []):
            if name in skip or name not in PREPROCESSING_MAP:
                continue
            steps.append((name, PREPROCESSING_MAP[name]()))

        # En az imputer olsun
        if not any(s[0] == "imputer" for s in steps):
            steps.insert(0, ("imputer", SimpleImputer(strategy="median")))

        # Model
        model_name = spec.get("model", "RandomForestClassifier")
        if model_name not in MODEL_MAP:
            model_name = "RandomForestClassifier"

        hp = spec.get("hyperparameters", {}) or {}
        allowed = VALID_HP.get(model_name, set())
        clean_hp = {k: v for k, v in hp.items()
                    if k in allowed and isinstance(v, (int, float, str, bool))}

        steps.append(("model", MODEL_MAP[model_name](**clean_hp)))
        return Pipeline(steps)

    def build_fixed_preprocessing(self, spec: dict) -> Pipeline:
        """A2 ablasyon: preprocessing sabit, sadece model LLM'den"""
        model_name = spec.get("model", "RandomForestClassifier")
        if model_name not in MODEL_MAP:
            model_name = "RandomForestClassifier"
        hp = spec.get("hyperparameters", {}) or {}
        allowed = VALID_HP.get(model_name, set())
        clean_hp = {k: v for k, v in hp.items() if k in allowed}
        return Pipeline([
            ("imputer",        SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
            ("model",          MODEL_MAP[model_name](**clean_hp)),
        ])