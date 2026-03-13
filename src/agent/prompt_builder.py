"""
src/agent/prompt_builder.py — v2
---------------------------------
Model cesitliligini zorlayan, dataset profile'a gore karar veren prompt.
"""

class PromptBuilder:

    FULL_PROFILE_TEMPLATE = """You are an expert machine learning engineer designing pipelines for tabular datasets.

Dataset profile:
- Name: {dataset_name}
- Task: {task_type}
- Rows: {n_rows}
- Features: {n_features}
- Missing ratio: {missing_ratio:.2%}
- Categorical ratio: {categorical_ratio:.2%}
- Class imbalance (minority fraction): {class_imbalance}
- Mean feature variance: {feature_variance}
- Classes: {n_classes}

STRICT MODEL SELECTION RULES — you MUST follow these:
1. n_rows < 2000 AND n_features < 20  →  use LogisticRegression or GradientBoostingClassifier/GradientBoostingRegressor
2. categorical_ratio > 0.40            →  use LGBMClassifier or LGBMRegressor
3. n_rows > 20000                      →  use XGBClassifier or XGBRegressor
4. class_imbalance < 0.15 (severe)     →  use GradientBoostingClassifier with subsample
5. n_features > 25 AND missing_ratio > 0.05 →  use RandomForestClassifier or RandomForestRegressor
6. Otherwise                           →  choose the most appropriate model

ALLOWED MODELS:
Classification: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier
Regression:     Ridge, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor

ALLOWED PREPROCESSING steps (use only these exact strings):
imputer, standard_scaler, min_max_scaler, one_hot_encoder, pca, variance_threshold

IMPORTANT: Do NOT include label_encoder in preprocessing list.

Return ONLY a valid JSON object with no extra text:
{{
  "preprocessing": ["imputer", "standard_scaler"],
  "model": "ModelName",
  "hyperparameters": {{"param": value}},
  "reasoning": "Explain which rule you applied and why this model fits this dataset."
}}"""

    NO_PROFILE_TEMPLATE = """You are an expert ML engineer.
Task type: {task_type}

Choose a preprocessing pipeline and model. You must NOT always pick RandomForest.
Consider LogisticRegression for simple tasks, GradientBoosting for medium complexity, XGB/LGBM for large data.

ALLOWED MODELS:
Classification: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier
Regression:     Ridge, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor

ALLOWED PREPROCESSING: imputer, standard_scaler, min_max_scaler, one_hot_encoder, pca, variance_threshold

Return ONLY valid JSON:
{{
  "preprocessing": ["imputer", "standard_scaler"],
  "model": "ModelName",
  "hyperparameters": {{}},
  "reasoning": "Brief justification."
}}"""

    MODEL_ONLY_TEMPLATE = """You are an expert ML engineer.

Dataset profile:
- Task: {task_type}
- Rows: {n_rows}
- Features: {n_features}
- Categorical ratio: {categorical_ratio:.2%}
- Class imbalance: {class_imbalance}

Choose ONLY the model (preprocessing will be fixed). Do NOT always pick RandomForest.
Apply the selection rules:
- n_rows < 2000 → LogisticRegression or GradientBoosting
- categorical > 0.40 → LGBM
- n_rows > 20000 → XGB
- Otherwise → best fit

ALLOWED MODELS:
Classification: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier
Regression:     Ridge, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor

Return ONLY valid JSON:
{{
  "preprocessing": [],
  "model": "ModelName",
  "hyperparameters": {{}},
  "reasoning": "Which rule applied and why."
}}"""

    def build(self, profile: dict, condition: str = "full_profile") -> str:
        p = profile
        ci = p.get("class_imbalance", 0.5)
        if ci is None: ci = 0.5

        if condition == "full_profile":
            return self.FULL_PROFILE_TEMPLATE.format(
                dataset_name       = p.get("dataset_name", "unknown"),
                task_type          = p.get("task_type", "classification"),
                n_rows             = p.get("n_rows", 0),
                n_features         = p.get("n_features", 0),
                missing_ratio      = p.get("missing_ratio", 0),
                categorical_ratio  = p.get("categorical_ratio", 0),
                class_imbalance    = round(float(ci), 4),
                feature_variance   = round(float(p.get("feature_variance", 0)), 4),
                n_classes          = p.get("n_classes", 2),
            )
        elif condition == "no_profile":
            return self.NO_PROFILE_TEMPLATE.format(
                task_type = p.get("task_type", "classification"),
            )
        elif condition == "model_only":
            return self.MODEL_ONLY_TEMPLATE.format(
                task_type         = p.get("task_type", "classification"),
                n_rows            = p.get("n_rows", 0),
                n_features        = p.get("n_features", 0),
                categorical_ratio = p.get("categorical_ratio", 0),
                class_imbalance   = round(float(ci), 4),
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")