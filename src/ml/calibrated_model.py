"""Serializable calibrated classifier used by the two risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CalibratedRiskModel:
    """Combines a random forest with an isotonic probability calibrator."""

    def __init__(self, estimator, calibrator, feature_columns):
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_columns = list(feature_columns)

    def predict_proba(self, features):
        frame = self._feature_frame(features)
        raw_probability = self.estimator.predict_proba(frame)[:, 1]
        calibrated_probability = self.calibrator.predict(raw_probability)
        return np.column_stack((1 - calibrated_probability, calibrated_probability))

    def predict(self, features, threshold=0.5):
        return (self.predict_proba(features)[:, 1] >= threshold).astype(int)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def _feature_frame(self, features):
        frame = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        return frame.loc[:, self.feature_columns]
