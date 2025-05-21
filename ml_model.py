# ml_models.py
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


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

        dist0 = np.linalg.norm(X - centers[0], axis=1)
        dist1 = np.linalg.norm(X - centers[1], axis=1)

        prob_cluster1 = dist0 / (dist0 + dist1 + 1e-8)

        if centers[0, 0] > centers[1, 0]:
            prob_risky = 1 - prob_cluster1
        else:
            prob_risky = prob_cluster1

        threshold = 0.3
        labels_soft = (prob_risky >= threshold).astype(int)

        return labels_soft

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

    def evaluate_model(self, commits: List[Dict[str, Any]]) -> Dict[str, float]:
        X = self._extract_X(commits)
        y = self._generate_pseudo_labels(X)
        print("[DEBUG] Метки (y) распределение:", Counter(y))

        if len(set(y)) < 2:
            print("[WARNING] В данных только один класс, метрики классификации не применимы.")
            clf = self.classifier
            clf.fit(X, y)
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred,
                                                                                                     dtype=float)
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc": 0.0
            }

        # Разбиваем на train/test с учётом баланса классов
        stratify_param = y if min(Counter(y).values()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

        if isinstance(self.classifier, CascadeForestClassifier):
            clf = CascadeForestClassifier(random_state=42)
        else:
            from copy import deepcopy
            clf = deepcopy(self.classifier)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("[DEBUG] y_pred распределение:", Counter(y_pred))

        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
            print("[DEBUG] y_proba min/max:", y_proba.min(), y_proba.max())
        else:
            y_proba = np.zeros_like(y_pred, dtype=float)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(set(y_test)) == 2 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }