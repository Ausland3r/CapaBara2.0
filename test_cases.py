from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

import ml_model as mlm
from ml_model import CommitRiskModel
from recommendations import RecommendationGenerator

RF = lambda: RandomForestClassifier(n_estimators=3, random_state=1)
generator = RecommendationGenerator()

COMMITS_OK = (
    mlm.COMMITS_OK
    if hasattr(mlm, "COMMITS_OK")
    else [
        {
            "lines_added": 10,
            "lines_deleted": 5,
            "files_changed": 2,
            "message_length": 20,
            "has_bug_keyword": 0,
            "complexity_score": 3,
            "pylint_warnings": 0,
            "pylint_errors": 0,
            "bandit_issues": 0,
            "eslint_warnings": 0,
            "eslint_errors": 0,
            "checkstyle_issues": 0,
            "avg_file_history": 1.5,
            "author_name": "Tester",
            "minutes_since_previous_commit": 30,
        },
        {
            "lines_added": 100,
            "lines_deleted": 50,
            "files_changed": 10,
            "message_length": 5,
            "has_bug_keyword": 1,
            "complexity_score": 15,
            "pylint_warnings": 5,
            "pylint_errors": 2,
            "bandit_issues": 1,
            "eslint_warnings": 3,
            "eslint_errors": 1,
            "checkstyle_issues": 10,
            "avg_file_history": 5.0,
            "author_name": "Tester",
            "minutes_since_previous_commit": 5,
        },
    ]
)

@pytest.fixture(scope="module")
def trained_model() -> CommitRiskModel:
    model = CommitRiskModel(RF(), sensitivity=1.0)
    model.fit(COMMITS_OK)
    return model

@pytest.mark.parametrize(
    ("bad_key", "bad_val", "exc"),
    [
        ("lines_added", -1, ValueError),
        ("files_changed", None, ValueError),
    ],
)
def test_invalid_input_raises(trained_model: CommitRiskModel, bad_key, bad_val, exc):
    bad = [dict(COMMITS_OK[0]), dict(COMMITS_OK[1])]
    bad[0][bad_key] = bad_val
    with pytest.raises(exc):
        trained_model.predict(bad)


def test_missing_field_raises(trained_model):
    bad = [dict(COMMITS_OK[0])]
    bad[0].pop("files_changed")
    with pytest.raises(KeyError):
        trained_model.predict(bad)


def test_string_in_numeric_field_raises(trained_model):
    bad = dict(COMMITS_OK[0], lines_added="ten")
    with pytest.raises(ValueError):
        trained_model.predict([bad])

def test_negative_complexity_rejected():
    bad = dict(COMMITS_OK[0], complexity_score=-0.1)
    with pytest.raises(ValueError):
        CommitRiskModel(RF())._extract_X([bad])

def test_proba_matrix_valid(trained_model):
    X = trained_model._extract_X(COMMITS_OK)
    proba = trained_model.classifier.predict_proba(X)
    assert proba.shape[0] == len(COMMITS_OK) and proba.ndim == 2
    for row in proba:
        assert np.all((0.0 <= row) & (row <= 1.0))
        assert pytest.approx(row.sum(), rel=1e-9) == 1.0

def test_risk_increases_with_total_changes(trained_model):
    tiny = dict(COMMITS_OK[0], lines_added=1, lines_deleted=0)
    huge = dict(COMMITS_OK[0], lines_added=200, lines_deleted=150)
    assert trained_model.predict([tiny])[0] <= trained_model.predict([huge])[0]


def test_risk_monotonic_complexity(trained_model):
    low = dict(COMMITS_OK[0], complexity_score=2)
    high = dict(COMMITS_OK[0], complexity_score=30)
    assert trained_model.predict([low])[0] <= trained_model.predict([high])[0]


@pytest.mark.parametrize("hist_low, hist_high", [(0.5, 10.0)])
def test_risk_monotonic_file_history(trained_model, hist_low, hist_high):
    low = dict(COMMITS_OK[0], avg_file_history=hist_low)
    high = dict(COMMITS_OK[0], avg_file_history=hist_high)
    assert trained_model.predict([low])[0] <= trained_model.predict([high])[0]


def test_more_deletes_less_risk(trained_model):
    plus = dict(COMMITS_OK[0], lines_added=10, lines_deleted=0)
    minus = dict(COMMITS_OK[0], lines_added=0, lines_deleted=10)
    assert trained_model.predict([plus])[0] >= trained_model.predict([minus])[0]


def test_sensitivity_affects_risky_count():
    lo = CommitRiskModel(RF(), sensitivity=0.0).fit(COMMITS_OK)
    hi = CommitRiskModel(RF(), sensitivity=3.0).fit(COMMITS_OK)
    assert hi._y.sum() <= lo._y.sum()

def test_none_message_riskier():
    base = COMMITS_OK[0]
    empty = dict(base, message=None)
    m = CommitRiskModel(RF()).fit([base, empty])
    proba_base = m.classifier.predict_proba(m._extract_X([base]))[0][-1]
    proba_empty = m.classifier.predict_proba(m._extract_X([empty]))[0][-1]
    assert proba_empty >= proba_base


def test_none_author_name_treated_as_risky(trained_model):
    ok = COMMITS_OK[0]
    none = dict(ok, author_name=None)
    p_ok = trained_model.classifier.predict_proba(trained_model._extract_X([ok]))[0][-1]
    p_none = trained_model.classifier.predict_proba(trained_model._extract_X([none]))[0][-1]
    assert p_none >= p_ok

def test_recommend_contains_bug_phrase():
    recs = generator.generate_recommendations(
        dict(COMMITS_OK[0], has_bug_keyword=1), 0.6, {}, {}
    )
    assert any("регрессионн" in r.lower() for r in recs)


def test_recommend_no_warning_on_low_risk():
    recs = generator.generate_recommendations(COMMITS_OK[0], 0.01, {}, {})
    assert all("📌" not in r for r in recs)


def test_recommend_split_large_commit():
    stats = {"total_changes": {"mean": 10, "std": 5}}
    recs = generator.generate_recommendations(COMMITS_OK[1], 0.4, stats, {})
    assert any("разбейте" in r.lower() for r in recs)

def test_message_length_positive_relation(trained_model):
    short = dict(COMMITS_OK[0], message_length=3)
    long  = dict(COMMITS_OK[0], message_length=100)
    assert trained_model.predict([short])[0] <= trained_model.predict([long])[0]

def test_bug_keyword_or_condition(trained_model):
    clean = dict(COMMITS_OK[0], has_bug_keyword=0, message=None)
    bug   = dict(clean,           has_bug_keyword=1)

    assert trained_model.predict([clean])[0] <= trained_model.predict([bug])[0]
