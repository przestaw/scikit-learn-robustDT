# Mirror of yangarbiter/scikit-learn, branch `robustDT`

This repository mirrors the `robustDT` branch of
<https://github.com/yangarbiter/scikit-learn>: robust decision trees
(`DecisionTreeClassifier(splitter="robust", eps=...)` and
`RandomForestClassifier(splitter="robust", eps=...)`) written by Yao-Yuan Yang
(`yangarbiter`) in 2019 on top of scikit-learn 0.21.dev0. The splitter follows
Chen et al., "Robust Decision Trees Against Adversarial Examples" (ICML 2019).

- Branch `robustDT` is kept byte-identical to the upstream fork (last commit
  `e9d32da6cd92ced574ccfb74a532a3388cd33f3b`). Nothing is committed on it.
- This branch (`mirror-notes`) is `robustDT` plus this file only.
- Branch `robustDT-sklearn-1.2` ports the robust splitter to the
  scikit-learn 1.2 release line (tag `1.2.2`) so it builds with Python
  3.10/3.11. See `ROBUST_PORT.md` on that branch for usage, build
  instructions and known differences.

The original branch targets Python 3.7 and the build tooling of early 2019
(NumPy, SciPy and Cython 0.29 of that time). It was used in the MSc thesis of
Przemysław Stawczyk (2023); the thesis results were computed with this
original code, not with the port.

License: BSD-3-Clause, as scikit-learn (see `COPYING`).
