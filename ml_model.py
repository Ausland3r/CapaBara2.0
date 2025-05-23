from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class CommitRiskModel:
    def __init__(
        self,
        classifier: ClassifierMixin,
        features: Optional[List[str]] = None,
        cluster_model: Optional[KMeans] = None,
        scaling: bool = True,
        sensitivity: float = 1.0
    ):
        self.classifier = classifier
        self.classifier_class = classifier.__class__
        self.classifier_params = (
            classifier.get_params() if hasattr(classifier, "get_params") else {}
        )
        self.features = features or [
            'lines_added', 'lines_deleted', 'files_changed',
            'avg_file_history', 'message_length', 'has_bug_keyword',
            'complexity_score',
            'pylint_warnings', 'pylint_errors', 'bandit_issues',
            'eslint_warnings', 'eslint_errors', 'checkstyle_issues'
        ]
        self.cluster_model = cluster_model or KMeans(n_clusters=2, random_state=0, n_init=10)
        self.scaling = scaling
        self.sensitivity = sensitivity
        self._is_fitted = False
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self.scaler = StandardScaler() if self.scaling else None

    def _extract_X(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        X = np.array([[commit.get(f, 0) for f in self.features] for commit in commits])
        if self.scaling:
            X = self.scaler.fit_transform(X)
        return X

    def _generate_pseudo_labels(self, X: np.ndarray) -> np.ndarray:
        labels = self.cluster_model.fit_predict(X)
        centers = self.cluster_model.cluster_centers_

        # Расстояния от каждого объекта до центра своего кластера
        dists = np.linalg.norm(X - centers[labels], axis=1)

        # Порог риска = среднее расстояние + sensitivity * std
        threshold = dists.mean() + self.sensitivity * dists.std()
        is_risky = (dists > threshold).astype(int)

        print(f"[DEBUG] Расстояния: mean={dists.mean():.4f}, std={dists.std():.4f}, threshold={threshold:.4f}")
        print(f"[DEBUG] Рискованные коммиты: {np.sum(is_risky)} из {len(is_risky)}")

        return is_risky

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
            clf = self.classifier_class(**self.classifier_params)
            clf.fit(X, y)
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") and clf.predict_proba(X).shape[1] > 1 else np.zeros_like(y_pred, dtype=float)
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "auc": 0.0
            }

        stratify_param = y if min(Counter(y).values()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

        clf = self.classifier_class(**self.classifier_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("[DEBUG] y_pred распределение:", Counter(y_pred))

        if hasattr(clf, "predict_proba"):
            if clf.predict_proba(X_test).shape[1] > 1:
                y_proba = clf.predict_proba(X_test)[:, 1]
                print("[DEBUG] y_proba min/max:", y_proba.min(), y_proba.max())
            else:
                y_proba = np.zeros_like(y_pred, dtype=float)
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
