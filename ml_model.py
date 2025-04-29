from deepforest import CascadeForestClassifier
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
import numpy as np


def train_model(commits, features=None):
    if features is None:
        features = [
            'lines_added', 'lines_deleted', 'files_changed',
            'avg_file_history', 'message_length',
            'has_bug_keyword', 'complexity_score'
        ]

    X = np.array([[commit.get(f, 0) for f in features] for commit in commits])

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    cluster_0_mean = np.mean(X[cluster_labels == 0], axis=0)
    cluster_1_mean = np.mean(X[cluster_labels == 1], axis=0)

    if cluster_0_mean[0] > cluster_1_mean[0]:
        risk_mapping = {0: 1, 1: 0}
    else:
        risk_mapping = {0: 0, 1: 1}

    y = np.array([risk_mapping[label] for label in cluster_labels])

    model = CascadeForestClassifier(
        n_bins=255,
        random_state=0,
        n_estimators=4,
        max_layers=10,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X, y)

    return model


def get_feature_importances(model, X, y):
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = result.importances_mean
    return importances
