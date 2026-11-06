# Quickstart

The core philosophy of `PyProbXAI` is a unified, intuitive entry point: the `explain()` function automatically inspects your model and dispatches it to the mathematically appropriate explainer.

---

## The Unified `explain` Function

```python
from PyProbXAI import explain

# Explain instance x with at most k conditions
exp = explain(model, x, k=4)
print(exp)
```

The explainer is selected automatically based on the model type:

| Model | Underlying Explainer | Default Method | Additional Requirements |
| :--- | :--- | :--- | :--- |
| `DecisionTreeClassifier` | `DecisionTreeExplainer` | `"gd"` (Greedy Descent) | None |
| `RandomForestClassifier`, `ExtraTreesClassifier`, `BaggingClassifier` | `TreeEnsembleExplainer` | `"ga"` (Greedy Ascent) | None |
| Neural Network, CNN, or any black-box model | `BlackBoxExplainer` | Bandit PAC sampling | `X_background` (e.g. training set) |

---

## 1. Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier
from PyProbXAI import explain

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

# Explain instance x
# method can be "gd" (greedy descent) or "ga" (greedy ascent)
exp = explain(clf, x, k=3, method="gd", feature_names=feature_names)
print(exp)
```

---

## 2. Tree Ensembles (Random Forest, Extra Trees)

```python
from sklearn.ensemble import RandomForestClassifier
from PyProbXAI import explain

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# method can be:
# - "ga": greedy ascent with curvature-based stopping (default)
# - "gd": greedy descent
# - "majority": majority explanation
exp = explain(rf, x, k=4, method="ga", feature_names=feature_names)
print(exp)
```

---

## 3. Black-Box Models (BanditExplain)

For models where trees are not available (e.g., PyTorch neural networks, LightGBM, SVMs), `BlackBoxExplainer` requires background data to estimate the error distribution:

```python
from PyProbXAI import explain

exp = explain(
    neural_net,
    x,
    k=4,
    X_background=X_train,
    feature_names=feature_names,
    categorical_features=["workclass", "occupation"],
    eta=0.1,       # Accuracy tolerance (PAC)
    delta=0.05,    # Failure probability (PAC)
    random_state=42
)
print(exp)
```

---

## The `Explanation` Result Object

Every explainer returns an instance of `Explanation`:

```python
# Access attributes programmatically:
print(exp.prediction)    # Predicted class
print(exp.conditions)    # Human-readable condition strings
print(exp.features)      # Indices of selected features
print(exp.error)         # Probability of prediction flip (epsilon)
print(exp.size)          # Number of conditions
```

---

## Interactive Visualizations & Notebooks

In Jupyter environments, evaluating an `Explanation` automatically renders an HTML summary card with certified error bound ($\varepsilon \le \epsilon$) and condition badges.

`PyProbXAI` also provides dedicated interactive views for different data modalities:

```python
# Tabular: inspect active constraints and invariant features
view = exp.show_tabular(x, feature_names=feature_names)

# Text: highlight active words and absent guard terms in a document
view = exp.show_text(document)

# Vision: 3-panel heatmap (original image, super-pixel heatmap, overlay)
view = exp.show_image(image, block_size=4)
# Or return a matplotlib figure:
fig = exp.plot_image(image, block_size=4)

# Export any interactive view as a standalone HTML file:
view.save("explanation_report.html")
```

Interactive, step-by-step Jupyter notebooks are available in `examples/notebooks/`:

- `tutorial_tabular_explanation.ipynb` (Adult dataset: Decision Trees, Random Forests, MLPs)
- `tutorial_text_explanation.ipynb` (20 Newsgroups: document classification with word highlights)
- `tutorial_vision_explanation.ipynb` (MNIST CNN: super-pixel heatmap visualization)

