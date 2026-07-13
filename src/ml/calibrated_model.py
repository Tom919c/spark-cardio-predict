"""Serializable calibrated classifier used by the two risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CalibratedRiskModel:
    """Combines a random forest with an isotonic probability calibrator."""

    def __init__(self, estimator, calibrator, feature_columns, decision_threshold=0.5):
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_columns = list(feature_columns)
        self.decision_threshold = float(decision_threshold)

    def predict_proba(self, features):
        frame = self._feature_frame(features)
        raw_probability = self.estimator.predict_proba(frame)[:, 1]
        calibrated_probability = self.calibrator.predict(raw_probability)
        return np.column_stack((1 - calibrated_probability, calibrated_probability))

    def predict(self, features, threshold=None):
        selected_threshold = (
            self.decision_threshold if threshold is None else float(threshold)
        )
        return (self.predict_proba(features)[:, 1] >= selected_threshold).astype(int)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def _feature_frame(self, features):
        frame = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        return frame.loc[:, self.feature_columns]
