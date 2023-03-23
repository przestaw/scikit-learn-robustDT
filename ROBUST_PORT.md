# Robust decision trees on scikit-learn 1.2

This branch (`robustDT-sklearn-1.2`) is a port of the robust decision tree
splitter from the `robustDT` branch of
<https://github.com/yangarbiter/scikit-learn> to the scikit-learn 1.2
(Dec 2022) release line. It is based on the `1.2.2` tag.

## Origin and credits

- Original code: Yao-Yuan Yang (GitHub `yangarbiter`), commits "robust tree"
  and "update" from April 2019 on top of scikit-learn 0.21.dev0, last fork
  commit `e9d32da6cd92ced574ccfb74a532a3388cd33f3b`.
- The splitter implements the robust splitting defence of H. Chen, H. Zhang,
  D. Boning and C.-J. Hsieh, "Robust Decision Trees Against Adversarial
  Examples", ICML 2019 (arXiv:1902.10660).
- The fork is the "robust splitting" defence installed by
  <https://github.com/yangarbiter/adversarial-nonparametrics> (Y.-Y. Yang,
  C. Rashtchian, Y. Wang, K. Chaudhuri, "Robustness for Non-Parametric
  Classification: A Generic Attack and Defense", AISTATS 2020).
- scikit-learn itself: the scikit-learn developers, BSD-3-Clause (see
  `COPYING`). The port keeps the same license.

The port was made by Przemysław Stawczyk so that the robust models used in his
MSc thesis (Warsaw University of Technology, 2023) can be rebuilt with a modern
Python. **The thesis results were computed with the original fork
(scikit-learn 0.21.dev0, Python 3.7), not with this port.**

## What was ported

| Original fork | This port |
| --- | --- |
| `sklearn.tree._splitter.RobustSplitter` (dense input) | same name, rewritten for the 1.2 splitter API (memoryviews, new `impurity_improvement` signature) |
| `DecisionTreeClassifier(splitter="robust", eps=0.1)` | same names and defaults |
| `RandomForestClassifier(splitter="best", eps=0.1)`, both forwarded to the trees | same names and defaults |
| robust split score (`robust_entropy`) | kept as in the fork, see "Differences and limitations" |

Files changed against `1.2.2`: `sklearn/tree/_splitter.pyx`,
`sklearn/tree/_classes.py`, `sklearn/ensemble/_forest.py`, and the new tests
in `sklearn/tree/tests/test_robust.py`.

`eps` is the perturbation radius in the units of the raw features: a sample
whose feature value lies within `eps` of a candidate threshold is treated as
movable to either side of the split by an adversary.

## Usage

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

tree = DecisionTreeClassifier(splitter="robust", eps=0.05, max_depth=5)
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))

forest = RandomForestClassifier(
    splitter="robust", eps=0.05, n_estimators=10, min_samples_leaf=2,
    random_state=0,
)
forest.fit(X_train, y_train)
print(forest.score(X_test, y_test))
```

## Building (Linux)

The 1.2 line predates Cython 3 and NumPy 1.25, so pin the build tools.
Tested with Python 3.11; Python 3.10 should work the same way.

```bash
git clone -b robustDT-sklearn-1.2 <this repository> scikit-learn-robust
cd scikit-learn-robust

python3.11 -m venv .venv            # or: uv venv -p 3.11 .venv
. .venv/bin/activate
pip install "numpy<1.25" "scipy<1.12" "Cython<3" "setuptools<69" wheel \
    joblib threadpoolctl pytest

# needs a C/C++ compiler with OpenMP (e.g. gcc, g++)
SKLEARN_BUILD_PARALLEL=4 pip install --no-build-isolation --no-deps -e .

pytest -q sklearn/tree/tests/test_robust.py
```

## Differences and limitations

- Dense input only. `splitter="robust"` on sparse `X` raises `ValueError`.
- Classification only. `DecisionTreeRegressor`, `ExtraTreeClassifier` and
  the extra-trees ensembles reject `splitter="robust"` (the fork did not
  validate the splitter name, so it silently accepted it there).
- The score is binary: label 0 against all other labels. Multiclass `y` is
  accepted but scored as "class 0 vs rest".
- Every feature is evaluated at every node: `max_features` (and therefore the
  per-split feature subsampling of `RandomForestClassifier`) is ignored, and
  constant features are not tracked. This matches the fork.
- `RandomForestClassifier` honours `n_jobs` (thread-based, as in upstream).
  The fork had replaced the joblib call with a sequential loop; with a fixed
  `random_state` the trees do not depend on the number of jobs.
- The split score code is kept as in the fork so that results stay comparable.
  This includes behaviour that looks unintended, for example:
  - the node class totals in `robust_entropy` read `y[i]` by node position
    instead of `y[samples[i]]`;
  - the chosen split position is the index of the last sample at or below the
    threshold, not one past it, so during fitting one boundary sample is
    counted in the right child; `tree_.n_node_samples` and the node values
    can therefore differ from what routing the training data through the
    thresholds gives;
  - candidate thresholds between tied feature values are not skipped.
- Equivalence with the original fork was checked on small synthetic data
  only: `make_moons` and `make_classification` (2 and 3 classes, up to 5
  features, 150-200 samples), `eps` in {0, 0.05, 0.5}, with and without
  `sample_weight`, `max_depth`, `min_samples_leaf=2`, `criterion="entropy"`,
  `max_leaf_nodes`, and `RandomForestClassifier(n_estimators=5)`. In all 36
  configurations the port grew the same trees as the fork (features,
  thresholds, `n_node_samples`, node values). The fork was built for this
  check with Python 3.8, NumPy 1.17 and Cython 0.29, using an installed
  joblib instead of its vendored copy. This is not a proof of equality on
  every dataset.
- Other scikit-learn 1.2 features are unchanged; only the files listed above
  differ from the `1.2.2` tag.
