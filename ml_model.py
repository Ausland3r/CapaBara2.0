from __future__ import annotations

from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CommitRiskModel:
    _NUMERIC_FIELDS = {
        "lines_added",
        "lines_deleted",
        "files_changed",
        "avg_file_history",
        "message_length",
        "complexity_score",
        "pylint_warnings",
        "pylint_errors",
        "bandit_issues",
        "eslint_warnings",
        "eslint_errors",
        "checkstyle_issues",
    }

    _OPTIONAL_FIELDS = {"has_bug_keyword"}

    _DEFAULTS: Dict[str, Any] = {f: 0 for f in _OPTIONAL_FIELDS}

    def __init__(
        self,
        classifier: ClassifierMixin,
        features: Optional[List[str]] = None,
        cluster_model: Optional[KMeans] = None,
        *,
        scaling: bool = True,
        sensitivity: float = 1.0,
    ):
        self.classifier = classifier
        self._clf_cls = classifier.__class__
        self._clf_params = classifier.get_params() if hasattr(
            classifier, "get_params"
        ) else {}

        base_feats = [
            "lines_added",
            "lines_deleted",
            "files_changed",
            "avg_file_history",
            "message_length",
            "complexity_score",
            *sorted(self._OPTIONAL_FIELDS),
            "pylint_warnings",
            "pylint_errors",
            "bandit_issues",
            "eslint_warnings",
            "eslint_errors",
            "checkstyle_issues",
        ]
        self.features: List[str] = features or base_feats

        self.cluster_model = cluster_model or KMeans(
            n_clusters=2, random_state=0, n_init=10
        )
        self.scaling = scaling
        self.sensitivity = sensitivity

        self._is_fitted = False
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self.scaler = StandardScaler() if scaling else None

    def _fill_defaults(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {**self._DEFAULTS, **commit}
            for commit in commits
        ]

    def _validate_commits(self, commits: List[Dict[str, Any]]) -> None:
        for i, commit in enumerate(commits):
            missing = [f for f in self.features if f not in commit]
            if missing:
                raise KeyError(
                    f"Отсутствует(ют) поле(я) {missing} в коммите {i}"
                )

            for f in self._NUMERIC_FIELDS & commit.keys():
                v = commit[f]
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Неверный тип поля '{f}' в коммите {i}: {type(v).__name__}"
                    )
                if v < 0:
                    raise ValueError(
                        f"Отрицательное значение поля '{f}' в коммите {i}: {v}"
                    )

    def _extract_X(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        commits = self._fill_defaults(commits)
        self._validate_commits(commits)

        X = np.array([[c[f] for f in self.features] for c in commits], dtype=float)
        return self.scaler.fit_transform(X) if self.scaling else X

    def _generate_pseudo_labels(self, X: np.ndarray) -> np.ndarray:
        labels = self.cluster_model.fit_predict(X)
        centres = self.cluster_model.cluster_centers_
        dists = np.linalg.norm(X - centres[labels], axis=1)

        thr = dists.mean() + self.sensitivity * dists.std()
        is_risky = (dists > thr).astype(int)

        print(
            f"[DEBUG] dist μ={dists.mean():.4f} σ={dists.std():.4f} thr={thr:.4f} "
            f"risky={is_risky.sum()}/{len(is_risky)}"
        )
        return is_risky

    def fit(self, commits: List[Dict[str, Any]]) -> "CommitRiskModel":
        X = self._extract_X(commits)
        y = self._generate_pseudo_labels(X)

        self.classifier.fit(X, y)

        self._X, self._y, self._is_fitted = X, y, True
        return self

    def predict(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        assert self._is_fitted, "Модель не обучена"
        X = self._extract_X(commits)
        return self.classifier.predict(X)

    def predict_proba(self, commits: List[Dict[str, Any]]) -> np.ndarray:
        assert self._is_fitted, "Модель не обучена"
        X = self._extract_X(commits)
        proba = self.classifier.predict_proba(X)
        return proba[:, -1] if proba.ndim == 2 else proba


    def feature_importances(self) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Сначала вызовите .fit()")

        if hasattr(self.classifier, "feature_importances_"):
            values = self.classifier.feature_importances_
        else:
            result = permutation_importance(
                self.classifier, self._X, self._y, n_repeats=5, n_jobs=-1
            )
            values = result.importances_mean

        return dict(zip(self.features, values))

    def evaluate_model(self, commits: List[Dict[str, Any]]) -> Dict[str, float]:
        X = self._extract_X(commits)
        y = self._generate_pseudo_labels(X)

        if len(set(y)) < 2:
            return dict.fromkeys(("precision", "recall", "f1_score", "auc"), 0.0)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = self._clf_cls(**self._clf_params)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = (
            clf.predict_proba(X_te)[:, 1]
            if hasattr(clf, "predict_proba") and clf.predict_proba(X_te).shape[1] > 1
            else np.zeros_like(y_pred, dtype=float)
        )

        return {
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1_score": f1_score(y_te, y_pred, zero_division=0),
            "auc": roc_auc_score(y_te, y_prob) if len(set(y_te)) == 2 else 0.0,
        }
