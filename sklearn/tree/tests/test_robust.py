"""Tests for the robust splitter ported from yangarbiter/scikit-learn@robustDT."""
import pickle

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sklearn.datasets import make_moons
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
)
from sklearn.tree._criterion import Gini
from sklearn.tree._splitter import RobustSplitter
from sklearn.utils._param_validation import InvalidParameterError


@pytest.fixture
def moons():
    return make_moons(n_samples=200, noise=0.2, random_state=0)


@pytest.mark.parametrize("eps", [0.0, 0.05, 0.5])
def test_robust_tree_fit_predict(moons, eps):
    X, y = moons
    clf = DecisionTreeClassifier(splitter="robust", eps=eps, max_depth=5)
    clf.fit(X, y)
    assert clf.tree_.node_count > 1
    assert clf.score(X, y) > 0.75
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_robust_tree_is_deterministic_and_eps_matters(moons):
    X, y = moons
    trees = [
        DecisionTreeClassifier(splitter="robust", eps=eps, max_depth=4).fit(X, y)
        for eps in (0.05, 0.05, 1.0)
    ]
    for attr in ("feature", "threshold"):
        np.testing.assert_array_equal(
            getattr(trees[0].tree_, attr), getattr(trees[1].tree_, attr)
        )
    assert not np.array_equal(trees[0].tree_.threshold, trees[2].tree_.threshold)


def test_robust_tree_multiclass_and_sample_weight():
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(60, 3))
    y = (X[:, 0] * 3).astype(int)
    clf = DecisionTreeClassifier(splitter="robust", eps=0.01, max_depth=3)
    clf.fit(X, y, sample_weight=rng.randint(1, 3, size=60).astype(float))
    assert set(clf.predict(X)) <= {0, 1, 2}


def test_robust_forest_fit_predict(moons):
    X, y = moons
    params = dict(
        splitter="robust", eps=0.05, n_estimators=5, min_samples_leaf=2, random_state=0
    )
    rf = RandomForestClassifier(**params).fit(X, y)
    assert rf.get_params()["splitter"] == "robust"
    assert all(t.splitter == "robust" and t.eps == 0.05 for t in rf.estimators_)
    assert rf.score(X, y) > 0.8
    # Threaded fitting gives the same forest as sequential fitting.
    rf2 = RandomForestClassifier(n_jobs=2, **params).fit(X, y)
    np.testing.assert_array_equal(rf.predict_proba(X), rf2.predict_proba(X))


def test_robust_tree_pickle(moons):
    X, y = moons
    clf = DecisionTreeClassifier(splitter="robust", eps=0.1, max_depth=3).fit(X, y)
    clf2 = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(clf.predict(X), clf2.predict(X))

    splitter = RobustSplitter(Gini(1, np.array([2], dtype=np.intp)), 2, 1, 0.0, None)
    splitter.eps = 0.25
    assert pickle.loads(pickle.dumps(splitter)).eps == 0.25


def test_robust_splitter_rejected_where_unsupported(moons):
    X, y = moons
    with pytest.raises(ValueError, match="does not support sparse"):
        DecisionTreeClassifier(splitter="robust").fit(csr_matrix(X), y)
    for est in (DecisionTreeRegressor, ExtraTreeClassifier):
        with pytest.raises(InvalidParameterError):
            est(splitter="robust").fit(X, y)
    with pytest.raises(InvalidParameterError):
        DecisionTreeClassifier(splitter="robust", eps=-1.0).fit(X, y)
    assert "eps" not in ExtraTreesClassifier().get_params()
