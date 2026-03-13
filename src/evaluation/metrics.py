"""
src/evaluation/metrics.py
--------------------------
Evaluator class: runs a fitted sklearn Pipeline on a test split
and returns the full metrics dict for logging.
"""

import numpy as np
import logging
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a fitted pipeline on a held-out test set.

    Returns a dict of metrics appropriate to the task type.
    All values are rounded to 4 decimal places.
    Missing metrics for the other task type are returned as None.
    """

    def evaluate(self, pipeline, X_test, y_test, task_type: str) -> dict:
        """
        Args:
            pipeline:  A fitted sklearn Pipeline.
            X_test:    Test features (numpy array or pandas DataFrame).
            y_test:    True labels / targets.
            task_type: 'classification' or 'regression'.

        Returns:
            dict with keys: f1_macro, accuracy, precision, recall,
                            auc_roc, rmse, mae, r2
            Non-applicable metrics are None.
        """
        y_pred = pipeline.predict(X_test)

        if task_type == "classification":
            # AUC-ROC: requires probability estimates
            auc = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_prob = pipeline.predict_proba(X_test)
                    n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 1
                    if n_classes == 2:
                        auc = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
                    else:
                        auc = round(roc_auc_score(
                            y_test, y_prob, multi_class="ovr", average="macro"
                        ), 4)
                except Exception as e:
                    logger.warning(f"AUC-ROC computation failed: {e}")

            return {
                "f1_macro":  round(f1_score(y_test, y_pred, average="macro",
                                            zero_division=0), 4),
                "accuracy":  round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred,
                                                   average="macro",
                                                   zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred,
                                                average="macro",
                                                zero_division=0), 4),
                "auc_roc":   auc,
                "rmse":      None,
                "mae":       None,
                "r2":        None,
            }

        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            return {
                "f1_macro":  None,
                "accuracy":  None,
                "precision": None,
                "recall":    None,
                "auc_roc":   None,
                "rmse":      round(float(np.sqrt(mse)), 4),
                "mae":       round(float(mean_absolute_error(y_test, y_pred)), 4),
                "r2":        round(float(r2_score(y_test, y_pred)), 4),
            }
