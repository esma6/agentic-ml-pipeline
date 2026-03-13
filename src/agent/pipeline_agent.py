"""
src/agent/pipeline_agent.py — v2 (Groq + model diversity)
"""

import json, uuid, datetime, logging, os, time
from groq import Groq
from src.agent.prompt_builder import PromptBuilder
from src.agent.response_parser import ResponseParser

logger = logging.getLogger(__name__)

ALLOWED_PREPROCESSING = {
    "imputer","standard_scaler","min_max_scaler",
    "one_hot_encoder","pca","variance_threshold",
}
ALLOWED_MODELS = {
    # Classification
    "LogisticRegression","RandomForestClassifier","GradientBoostingClassifier",
    "SVC","KNeighborsClassifier","XGBClassifier","LGBMClassifier",
    # Regression
    "Ridge","RandomForestRegressor","GradientBoostingRegressor",
    "SVR","XGBRegressor","LGBMRegressor",
}
HYPERPARAMETER_CLAMPS = {
    "n_estimators":(50,500),"max_depth":(3,20),"C":(0.001,100),
    "alpha":(0.01,100),"n_neighbors":(3,15),
    "learning_rate":(0.01,0.3),"max_iter":(100,2000),
    "num_leaves":(20,200),"subsample":(0.5,1.0),
}
DEFAULT_SPEC = {
    "preprocessing":["imputer","standard_scaler"],
    "model":"GradientBoostingClassifier",
    "hyperparameters":{"n_estimators":100},
    "reasoning":"Fallback.",
}
DEFAULT_SPEC_REG = {
    "preprocessing":["imputer","standard_scaler"],
    "model":"GradientBoostingRegressor",
    "hyperparameters":{"n_estimators":100},
    "reasoning":"Fallback.",
}

class PipelineAgent:
    def __init__(self, client, model_name, prompt_builder, response_parser,
                 log_path="experiments/results/agent_log.jsonl"):
        self.client         = client
        self.model_name     = model_name
        self.prompt_builder = prompt_builder
        self.parser         = response_parser
        self.log_path       = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    @classmethod
    def from_env(cls, model_name="llama-3.1-8b-instant",
                 log_path="experiments/results/agent_log.jsonl"):
        api_key = os.environ.get("GROQ_API_KEY","")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY bulunamadi!")
        return cls(Groq(api_key=api_key), model_name,
                   PromptBuilder(), ResponseParser(), log_path)

    def generate_pipeline(self, profile, condition="full_profile"):
        run_id = str(uuid.uuid4())
        prompt = self.prompt_builder.build(profile, condition=condition)
        raw    = self._call_groq(prompt, run_id)
        spec   = self.parser.parse(raw) if raw else None
        spec   = self._validate(spec, profile.get("task_type","classification"))
        spec.update({
            "agent_model":      self.model_name,
            "prompt_condition": condition,
            "run_id":           run_id,
            "timestamp":        datetime.datetime.utcnow().isoformat()+"Z",
        })
        self._log(run_id, profile, prompt, raw, spec)
        logger.info(f"[{run_id[:8]}] model={spec['model']} cond={condition}")
        return spec

    def _call_groq(self, prompt, run_id, max_attempts=3):
        for attempt in range(1, max_attempts+1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                )
                return r.choices[0].message.content
            except Exception as e:
                err = str(e).lower()
                wait = 65 if ("429" in err or "rate" in err) else 2**attempt
                logger.warning(f"[{run_id[:8]}] Hata {attempt}/{max_attempts}: {str(e)[:60]} — {wait}s")
                if attempt == max_attempts: return None
                time.sleep(wait)

    def _validate(self, spec, task_type):
        fb = DEFAULT_SPEC if task_type=="classification" else DEFAULT_SPEC_REG
        if not spec: return dict(fb)
        # Model: string olmali ve ALLOWED_MODELS'da olmali
        model = spec.get("model")
        if not isinstance(model, str) or model not in ALLOWED_MODELS:
            spec["model"] = fb["model"]
        # Preprocessing: sadece string listesi, label_encoder yok
        steps = spec.get("preprocessing", [])
        spec["preprocessing"] = [
            s for s in (steps if isinstance(steps, list) else [])
            if isinstance(s, str) and s in ALLOWED_PREPROCESSING
        ]
        # Hyperparameters: clamp
        hp = spec.get("hyperparameters", {})
        clamped = {}
        for k, v in (hp if isinstance(hp, dict) else {}).items():
            if k in HYPERPARAMETER_CLAMPS and isinstance(v, (int, float)):
                lo, hi = HYPERPARAMETER_CLAMPS[k]
                clamped[k] = max(lo, min(hi, v))
            elif isinstance(v, (int, float, str, bool)):
                clamped[k] = v
        spec["hyperparameters"] = clamped
        if not spec.get("reasoning"): spec["reasoning"] = "Gerekce belirtilmedi."
        return spec

    def _log(self, run_id, profile, prompt, raw, spec):
        entry = {"run_id":run_id,"dataset":profile.get("dataset_name"),
                 "condition":spec.get("prompt_condition"),"profile":profile,
                 "prompt":prompt,"raw_response":raw or "","spec":spec,
                 "timestamp":spec.get("timestamp")}
        with open(self.log_path,"a",encoding="utf-8") as f:
            f.write(json.dumps(entry,ensure_ascii=False)+"\n")