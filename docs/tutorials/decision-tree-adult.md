# Tutorial: Decision Trees on Tabular Data

In this tutorial, we train a `DecisionTreeClassifier` on the Adult Census Income dataset and compute probable explanations using **Greedy Descent (GD)** and **Greedy Ascent (GA)** based on UAI 2023.

---

## 1. Load Data & Train the Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from PyProbXAI import DecisionTreeExplainer

# Load and preprocess Adult dataset
df = pd.read_csv("examples/data/adult.csv")
X, y = df.drop(columns="income"), df["income"]

# Encode categoricals as integer codes
for c in X.select_dtypes(include="object").columns:
    X[c] = pd.factorize(X[c], sort=True)[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Train a decision tree
dt = DecisionTreeClassifier(max_depth=8, random_state=0)
dt.fit(X_train, y_train)
```

---

## 2. Generate Probable Explanations

Initialize `DecisionTreeExplainer` and explain an individual instance $x$:

```python
explainer = DecisionTreeExplainer(dt, feature_names=X.columns)
x = X_test.iloc[0].values

# Compare Greedy Descent vs Greedy Ascent
exp_gd = explainer.explain(x, k=3, method="gd")
exp_ga = explainer.explain(x, k=3, method="ga")

print("=== Greedy Descent (GD) ===")
print(exp_gd)

print("\n=== Greedy Ascent (GA) ===")
print(exp_ga)
```

### Example Output

```text
=== Greedy Descent (GD) ===
pred: 0
conditions:
  * marital-status != 2 (Married-civ-spouse)
  * capital-gain <= 7073.5
  * education-num <= 12.5
error: 0.0156
precision: 0.9844
```

Notice how `DecisionTreeExplainer` computes the exact error on the orthogonal DNF representing tree branches voting for the alternative class.

---

## 3. Interactive Visualization

In a Jupyter notebook, render the interactive tabular card with feature invariance inspection:

```python
# Displays interactive card highlighting certified constraints vs invariant features
exp_gd.show_tabular(x, feature_names=X.columns)
```

> **Interactive Notebook:** See [`examples/notebooks/tutorial_tabular_explanation.ipynb`](https://github.com/SifEddine05/PyProbXAI/blob/main/examples/notebooks/tutorial_tabular_explanation.ipynb) for the full interactive walkthrough across Decision Trees, Random Forests, and MLPs.

