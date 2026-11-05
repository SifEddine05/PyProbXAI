# PyProbXAI

[![PyPI version](https://img.shields.io/pypi/v/PyProbXAI.svg)](https://pypi.org/project/PyProbXAI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**PyProbXAI** is a Python library that explains predictions of machine learning classifiers using **probable explanations** grounded in rigorous theoretical foundations.

---

## What is a Probable Explanation?

Given a classifier $f$ and an instance $x$ whose prediction is $c = f(x)$, a **probable explanation** is a small subset of features / conditions (at most $k$) of $x$ such that:

> *When these conditions are held fixed and all other features are allowed to vary according to an underlying distribution, the model continues to predict $c$ with high probability.*

- **Error ($\text{error}$)**: The probability that the prediction changes when non-fixed features vary.
- **Precision ($\text{precision}$)**: Defined as $1 - \text{error}$.
- **Sparsity ($k$)**: The maximum number of literals/conditions in the explanation, guaranteeing cognitive interpretability.

Unlike heuristic feature attributions (e.g., LIME or standard SHAP), **PyProbXAI** provides formal probabilistic guarantees, exact model counting, or PAC guarantees depending on the model family.

---

## Supported Methods & Papers

The library brings together state-of-the-art algorithms across three distinct model settings:

| Model Setting | Supported Estimators | Algorithmic Approach | Theoretical Foundation |
| :--- | :--- | :--- | :--- |
| **Decision Trees** | `DecisionTreeClassifier` | Exact counting via orthogonal DNFs; Greedy Descent (GD) & Greedy Ascent (GA) | **UAI 2023** [1] |
| **Tree Ensembles** | `RandomForestClassifier`, `ExtraTreesClassifier`, `BaggingClassifier` | $k$-pseudo-probable explanations (GA with curvature-based stopping, GD) & Majority reasons | **ECML PKDD 2026** [2] |
| **Any Black-Box** | Neural networks, CNNs, SVMs, Boosting, any model with `.predict()` | Multi-Armed Bandit sampling (BanditExplain) with $(\eta, \delta)$-PAC guarantees | **BanditExplain** [3] |

---

## Quick Example

```python
from PyProbXAI import explain
from sklearn.tree import DecisionTreeClassifier

# Train your model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Generate a concise probable explanation (k <= 4 conditions)
exp = explain(clf, x, k=4, feature_names=feature_names)

print(exp)
# Output:
# class: 1
# conditions:
#   * marital-status = Married-civ-spouse
#   * education-num > 12.0
# error:     0.0412
# precision: 0.9588

# Interactive visualizations (Jupyter HTML card or standalone report):
exp.show_tabular(x, feature_names=feature_names)
```

---

## References

1. **UAI 2023**: L. Bounia, F. Koriche. *Approximating Probabilistic Explanations via Supermodular Minimization*. Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023), PMLR 216.
2. **ECML PKDD 2026**: L. Bounia. *k-Pseudo-Probable Path Explanations for Weighted Tree-Based Ensembles: Bridging Interpretability and Computational Efficiency*. Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2026), pp. 39-57. Springer Nature Switzerland.
3. **BanditExplain**: S. E. Sellami, L. Bounia, J. Agoun, C. Rouveirol. *BanditExplain: Scalable Probabilistic Explanations with PAC Guarantees via Multi-Armed Bandits*.

