# ml_models.py

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

class CommitRiskModel:
    def __init__(
        self,
        classifier: ClassifierMixin,
        features: Optional[List[str]] = None,
        cluster_model: Optional[KMeans] = None
    ):
        self.classifier = classifier
        self.features = features or [
            'lines_added', 'lines_deleted', 'files_changed',
            'avg_file_history', 'message_length',
            'has_bug_keyword', 'complexity_score'
        ]
        self.cluster_model = cluster_model or KMeans(n_clusters=2, random_state=0, n_init=10)
        self._is_fitted = False
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def _extract_X(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        return np.array([[commit.get(f, 0) for f in self.features] for commit in commits])

    def _generate_pseudo_labels(self, X: np.ndarray) -> np.ndarray:
        labels = self.cluster_model.fit_predict(X)
        centers = self.cluster_model.cluster_centers_
        if centers[0, 0] > centers[1, 0]:
            mapping = {0: 1, 1: 0}
        else:
            mapping = {0: 0, 1: 1}
        return np.vectorize(mapping.get)(labels)

    def fit(self, commits: List[Dict[str, Any]]):
        X = self._extract_X(commits)
        y = self._generate_pseudo_labels(X)
        self.classifier.fit(X, y)
        self._X, self._y = X, y
        self._is_fitted = True
        return self

    def predict(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        assert self._is_fitted, "Модель не обучена"
        X = self._extract_X(commits)
        return self.classifier.predict(X)

    def predict_proba(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        assert self._is_fitted, "Модель не обучена"
        X = self._extract_X(commits)
        return self.classifier.predict_proba(X)[:, 1]

    def feature_importances(self) -> Dict[str, float]:
        if not self._is_fitted or self._X is None or self._y is None:
            raise RuntimeError("Нужно вызвать .fit() перед feature_importances()")
        if hasattr(self.classifier, "feature_importances_"):
            vals = self.classifier.feature_importances_
        else:
            result = permutation_importance(
                self.classifier, self._X, self._y,
                n_repeats=5, random_state=0, n_jobs=-1
            )
            vals = result.importances_mean
        return dict(zip(self.features, vals))
