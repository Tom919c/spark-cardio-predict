"""双模型预测器：加载心脏与卒中模型并输出联合预测结果。"""

import joblib
import pandas as pd
from pathlib import Path
from threading import Lock


class CardioRiskPredictor:
    """加载已训练的双模型，对单样本执行预测。"""

    _model_cache = {}
    _cache_lock = Lock()

    @classmethod
    def load_model(cls, model_path):
        """按文件修改时间缓存模型，避免每次请求重复反序列化。"""
        path = Path(model_path).resolve()
        stat = path.stat()
        cache_key = str(path)
        version = (stat.st_mtime_ns, stat.st_size)
        with cls._cache_lock:
            cached = cls._model_cache.get(cache_key)
            if cached and cached[0] == version:
                return cached[1]
            model = joblib.load(path)
            cls._model_cache[cache_key] = (version, model)
            return model

    def predict(self, model_paths, sample):
        try:
            result = self.predict_dataframe(
                model_paths,
                self._build_sample_frame(sample),
            )
        except (KeyError, OSError, EOFError, ImportError, ValueError, AttributeError) as exc:
            raise ValueError("模型文件无法加载或与当前特征契约不兼容，请重新训练。") from exc

        return {
            key: value[0] if isinstance(value, list) else value
            for key, value in result.items()
        }

    def predict_dataframe(self, model_paths, dataframe):
        """批量预测，返回可直接转为 JSON 的列式结果。"""
        heart_model = self.load_model(model_paths["heart"])
        stroke_model = self.load_model(model_paths["stroke"])
        # Flask 在 Windows 下不适合让单次请求再创建 joblib 进程池。
        # 只限制推理并行度，不改变训练产物、特征契约或概率校准逻辑。
        self._configure_inference_model(heart_model)
        self._configure_inference_model(stroke_model)
        dataframe = self._align_features(dataframe, heart_model)
        heart_probability = self._predict_probability(
            heart_model, dataframe
        )
        stroke_probability = self._predict_probability(
            stroke_model, dataframe
        )
        heart_labels = self._predict_label(heart_model, dataframe, heart_probability)
        stroke_labels = self._predict_label(stroke_model, dataframe, stroke_probability)
        return {
            "heart_predicted_label": heart_labels.astype(int).tolist(),
            "stroke_predicted_label": stroke_labels.astype(int).tolist(),
            "heart_predicted_probability": heart_probability.tolist(),
            "stroke_predicted_probability": stroke_probability.tolist(),
        }

    @staticmethod
    def _configure_inference_model(model):
        """将树模型推理限制为当前 Flask 进程，避免 Windows 子进程权限问题。"""
        estimator = getattr(model, "estimator", model)
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1

    def _predict_probability(self, model, sample_frame):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(sample_frame)[:, 1]
        raise ValueError("模型不支持概率输出。")

    def _predict_label(self, model, sample_frame, probabilities):
        if hasattr(model, "predict"):
            return model.predict(sample_frame)
        threshold = getattr(model, "decision_threshold", 0.5)
        return (probabilities >= threshold).astype(int)

    def _build_sample_frame(self, sample):
        working_sample = dict(sample)
        if "bmi" in working_sample:
            working_sample["bmi"] = round(float(working_sample["bmi"]), 2)
        feature_columns = getattr(sample, "feature_columns", None)
        if feature_columns:
            return pd.DataFrame([working_sample], columns=feature_columns)
        return pd.DataFrame([working_sample])

    @staticmethod
    def _align_features(dataframe, model):
        feature_columns = getattr(model, "feature_columns", None)
        if feature_columns:
            missing = [column for column in feature_columns if column not in dataframe]
            if missing:
                raise KeyError(f"缺少模型特征列: {missing}")
            return dataframe.loc[:, feature_columns]
        return dataframe
