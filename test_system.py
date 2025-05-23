import copy
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from app import load_and_analyze_repos, train_and_update_model, update_tabs
from ml_model import CommitRiskModel
from recommendations import generate_recommendations
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# Расширенный набор коммитов, включая экстремумы и шум
COMMITS_EXTENDED = [
    # Экстремальный коммит (много изменений)
    {
        "lines_added": 1000, "lines_deleted": 500, "files_changed": 100,
        "avg_file_history": 10.0, "message_length": 300, "has_bug_keyword": 1,
        "complexity_score": 50, "pylint_warnings": 20, "pylint_errors": 10,
        "bandit_issues": 5, "eslint_warnings": 0, "eslint_errors": 0,
        "checkstyle_issues": 100, "author_name": "Overflow", "minutes_since_previous_commit": 500
    },
    # Почти пустой коммит
    {
        "lines_added": 0, "lines_deleted": 0, "files_changed": 0,
        "avg_file_history": 0.0, "message_length": 0, "has_bug_keyword": 0,
        "complexity_score": 0, "pylint_warnings": 0, "pylint_errors": 0,
        "bandit_issues": 0, "eslint_warnings": 0, "eslint_errors": 0,
        "checkstyle_issues": 0, "author_name": "Ghost", "minutes_since_previous_commit": None
    },
    # Типичный коммит
    {
        "lines_added": 25, "lines_deleted": 10, "files_changed": 3,
        "avg_file_history": 2.5, "message_length": 80, "has_bug_keyword": 0,
        "complexity_score": 4, "pylint_warnings": 1, "pylint_errors": 0,
        "bandit_issues": 0, "eslint_warnings": 1, "eslint_errors": 0,
        "checkstyle_issues": 3, "author_name": "Normal", "minutes_since_previous_commit": 15
    },
    # Частый коммит
    {
        "lines_added": 3, "lines_deleted": 1, "files_changed": 1,
        "avg_file_history": 1.2, "message_length": 12, "has_bug_keyword": 0,
        "complexity_score": 1, "pylint_warnings": 0, "pylint_errors": 0,
        "bandit_issues": 0, "eslint_warnings": 0, "eslint_errors": 0,
        "checkstyle_issues": 0, "author_name": "FastDev", "minutes_since_previous_commit": 0.5
    },
    # Рефакторинг (много изменений без добавлений)
    {
        "lines_added": 10, "lines_deleted": 200, "files_changed": 20,
        "avg_file_history": 4.0, "message_length": 60, "has_bug_keyword": 0,
        "complexity_score": 2, "pylint_warnings": 2, "pylint_errors": 1,
        "bandit_issues": 0, "eslint_warnings": 3, "eslint_errors": 0,
        "checkstyle_issues": 5, "author_name": "Cleaner", "minutes_since_previous_commit": 60
    },
    # Багфикс коммит
    {
        "lines_added": 4, "lines_deleted": 2, "files_changed": 1,
        "avg_file_history": 1.0, "message_length": 25, "has_bug_keyword": 1,
        "complexity_score": 2, "pylint_warnings": 0, "pylint_errors": 0,
        "bandit_issues": 0, "eslint_warnings": 0, "eslint_errors": 0,
        "checkstyle_issues": 0, "author_name": "Fixer", "minutes_since_previous_commit": 120
    },
    # Коммит с неожиданным количеством проблем
    {
        "lines_added": 20, "lines_deleted": 5, "files_changed": 2,
        "avg_file_history": 1.5, "message_length": 10, "has_bug_keyword": 0,
        "complexity_score": 3, "pylint_warnings": 8, "pylint_errors": 5,
        "bandit_issues": 3, "eslint_warnings": 2, "eslint_errors": 1,
        "checkstyle_issues": 20, "author_name": "MessyDev", "minutes_since_previous_commit": 5
    },
]

def test_model_extremes_and_balanced_cases():
    model = CommitRiskModel(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(COMMITS_EXTENDED)
    preds = model.predict(COMMITS_EXTENDED)
    assert len(set(preds)) > 1, "Модель не должна выдавать один и тот же класс на разные входы"
    # Проверка типов и длины
    assert isinstance(preds, (list, np.ndarray))
    assert len(preds) == len(COMMITS_EXTENDED)
    assert all(isinstance(p, (int, np.integer)) for p in preds)

def test_model_predict_proba_output():
    model = CommitRiskModel(RandomForestClassifier(n_estimators=10, random_state=42))
    model.fit(COMMITS_EXTENDED)
    proba = model.predict_proba(COMMITS_EXTENDED)
    assert len(proba) == len(COMMITS_EXTENDED)
    for p in proba:
        # p — скаляр вероятности класса риска, проверяем, что в диапазоне [0,1]
        assert 0.0 <= p <= 1.0

def test_recommendations_extreme_commit():
    risk = 0.99
    commit = COMMITS_EXTENDED[0]
    stats = {
        "total_changes": {"mean": 100, "std": 50},
        "files_changed": {"quantile_95": 10},
        "complexity_score": {"quantile_90": 10},
        "avg_file_history": {"mean": 3, "std": 1},
        "commit_interval": {"median": 30},
        "author_stats": {"Overflow": {"median_lines_added": 50}}
    }
    recs = generate_recommendations(commit, risk, stats, {})
    assert any("⚠️" in r for r in recs), "Для высокорискового коммита должно быть предупреждение"
    assert any("📊" in r for r in recs), "Должна быть рекомендация по объёму изменений"

def test_empty_commit_recommendation():
    commit = copy.deepcopy(COMMITS_EXTENDED[1])
    commit['message_length'] = 50
    recs = generate_recommendations(commit, 0.05, {}, {})
    assert any("✅" in r for r in recs), "Для безопасного коммита должна быть зелёная рекомендация"

def test_model_handles_missing_fields_gracefully():
    partial_commit = {
        "lines_added": 20, "message_length": 25, "has_bug_keyword": 0
    }
    model = CommitRiskModel(RandomForestClassifier(n_estimators=10))
    model.fit(COMMITS_EXTENDED + [partial_commit])
    proba = model.predict_proba([partial_commit])
    assert 0.0 <= proba[0] <= 1.0

def test_model_predict_consistency():
    model = CommitRiskModel(RandomForestClassifier(n_estimators=50, random_state=123))
    model.fit(COMMITS_EXTENDED)
    preds1 = model.predict(COMMITS_EXTENDED)
    preds2 = model.predict(COMMITS_EXTENDED)
    assert np.array_equal(preds1, preds2), "Предсказания должны быть детерминированы"

def test_model_with_empty_input():
    model = CommitRiskModel(RandomForestClassifier(n_estimators=10))
    with pytest.raises(ValueError):
        model.fit([])
    model.fit(COMMITS_EXTENDED)
    with pytest.raises(ValueError):
        model.predict([])

def test_model_with_invalid_input_types():
    model = CommitRiskModel(RandomForestClassifier(n_estimators=10))
    model.fit(COMMITS_EXTENDED)
    invalid_commit = {"lines_added": -10, "message_length": "a lot", "has_bug_keyword": None}
    with pytest.raises(Exception):
        model.predict([invalid_commit])

def test_recommendations_on_typical_and_bugfix_commits():
    stats = {
        "total_changes": {"mean": 20, "std": 10},
        "files_changed": {"quantile_95": 5},
        "complexity_score": {"quantile_90": 7},
        "avg_file_history": {"mean": 2, "std": 0.5},
        "commit_interval": {"median": 30},
        "author_stats": {"Fixer": {"median_lines_added": 3}, "Normal": {"median_lines_added": 25}}
    }
    # Типичный коммит
    typical_commit = COMMITS_EXTENDED[2]
    recs_typical = generate_recommendations(typical_commit, 0.3, stats, {})
    assert recs_typical, "Должны быть рекомендации для типичного коммита"
    # Багфикс коммит
    bugfix_commit = COMMITS_EXTENDED[5]
    recs_bugfix = generate_recommendations(bugfix_commit, 0.7, stats, {})
    assert any("🐞" in r for r in recs_bugfix), "Должны быть предупреждения или советы для багфикс коммита"

def test_recommendations_for_risk_bounds():
    commit = COMMITS_EXTENDED[2]
    stats = {}
    recs_low = generate_recommendations(commit, 0.0, stats, {})
    recs_high = generate_recommendations(commit, 1.0, stats, {})
    assert recs_low, "Рекомендации для риска 0 должны возвращаться"
    assert recs_high, "Рекомендации для риска 1 должны возвращаться"

@patch('app.os.getenv')
@patch('app.GitHubRepoAnalyzer')
def test_load_and_analyze_repos(mock_analyzer_cls, mock_getenv):
    mock_getenv.side_effect = lambda key, default=None: {
        "GITHUB_TOKEN": "token",
        "GITHUB_REPOS": "owner1/repo1,owner2/repo2"
    }.get(key, default)

    mock_analyzer = MagicMock()
    mock_analyzer.analyze_commits.return_value = [
        {'sha': '1', 'message': 'fix bug', 'lines_added': 5, 'lines_deleted': 2, 'files_changed': 1, 'complexity_score': 3},
    ]
    mock_analyzer.analyze_and_pr.return_value = None
    mock_analyzer_cls.side_effect = [mock_analyzer, mock_analyzer]

    token, repos, analyses, all_commits = load_and_analyze_repos()

    assert token == "token"
    assert repos == ["owner1/repo1", "owner2/repo2"]
    assert isinstance(analyses, dict)
    assert all_commits

def test_train_and_update_model_structure():
    commits = [
        {'sha': '1', 'message': 'fix', 'lines_added': 1, 'lines_deleted': 0, 'files_changed': 1, 'complexity_score': 1},
        {'sha': '2', 'message': 'feat', 'lines_added': 10, 'lines_deleted': 2, 'files_changed': 3,
         'complexity_score': 3}
    ]
    repos = ['owner1/repo1']
    analyses = {
        'owner1/repo1': {'df': pd.DataFrame(commits)}
    }

    model, updated_analyses = train_and_update_model(commits, repos, analyses)
    assert 'model' in updated_analyses['owner1/repo1']
    assert 'feat_imps' in updated_analyses['owner1/repo1']
    assert 'metrics' in updated_analyses['owner1/repo1']
    df = updated_analyses['owner1/repo1']['df']
    assert 'Risk_Proba' in df.columns
    assert 'Risk' in df.columns
