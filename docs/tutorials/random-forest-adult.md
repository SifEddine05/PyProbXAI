# Tutorial: Random Forests & Majority Explanations

In this tutorial, we explain an ensemble of trees (`RandomForestClassifier`) using $k$-pseudo-probable explanations and majority explanations (ECML PKDD 2026).

---

## 1. Train the Random Forest

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PyProbXAI import TreeEnsembleExplainer

# Load and prepare the data
df = pd.read_csv("examples/data/adult.csv")
X, y = df.drop(columns="income"), df["income"]
for c in X.select_dtypes(include="object").columns:
    X[c] = pd.factorize(X[c], sort=True)[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Train a Random Forest with 20 estimators
rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0)
rf.fit(X_train, y_train)
```

---

## 2. Generate $k$-Pseudo Explanations

```python
explainer = TreeEnsembleExplainer(rf, feature_names=X.columns)
x = X_test.iloc[0].values

# 1. Greedy Ascent with curvature stopping rule (default)
exp_ga = explainer.explain(x, k=4, method="ga")

# 2. Greedy Descent
exp_gd = explainer.explain(x, k=4, method="gd")

print("=== Greedy Ascent with Curvature Stopping ===")
print(exp_ga)
```

---

## 3. Generate Majority Explanations

For voting ensembles, `method="majority"` finds a set of conditions that guarantees at least $\lfloor m/2 \rfloor + 1$ trees vote for the predicted class with 0 error:

```python
# No size limit k is required: the method halts once a strict majority is secured
exp_maj = explainer.explain(x, method="majority")
print("=== Majority Explanation ===")
print(exp_maj)
```

This guarantees that whenever those conditions hold, a strict majority of individual trees will predict the same class.

---

## 4. Interactive Visualization

```python
# Visualize certified constraints, voting statistics and invariant features
exp_ga.show_tabular(x, feature_names=X.columns, title="Adult - Random Forest Explanation")
```

> **Interactive Notebook:** Run [`examples/notebooks/tutorial_tabular_explanation.ipynb`](https://github.com/SifEddine05/PyProbXAI/blob/main/examples/notebooks/tutorial_tabular_explanation.ipynb) to interactively inspect feature invariance and compare k-pseudo vs majority explanations.

