# Tutorial: Neural Networks with BanditExplain

For non-tree models, such as Multi-Layer Perceptrons (MLPs), `PyProbXAI` uses **BanditExplain (BE)**. It only accesses the `.predict()` method and samples perturbed instances from background data.

---

## 1. Train a Neural Network Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from PyProbXAI import BlackBoxExplainer

df = pd.read_csv("examples/data/adult.csv")
X, y = df.drop(columns="income"), df["income"]
categorical = list(X.select_dtypes(include="object").columns)

# Keep decoders so the explanation displays original category strings
decoders = {}
for c in categorical:
    X[c], names = pd.factorize(X[c], sort=True)
    decoders[X.columns.get_loc(c)] = list(names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

nn = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0))
nn.fit(X_train, y_train)
```

---

## 2. Explain Predictions with PAC Guarantees

```python
explainer = BlackBoxExplainer(
    nn,
    X_background=X_train,
    categorical_features=categorical,
    cat_decoders=decoders,
    n_bins=3,         # Quantile binning for numeric features
    eta=0.1,          # Error tolerance: estimated error is within 0.1 of true error
    delta=0.05,       # Confidence: with probability >= 95%
    random_state=0
)

x = X_test.iloc[0].values
exp = explainer.explain(x, k=4)
print(exp)
```

### Output

```text
pred: 0
conditions:
  * marital-status = Never-married
  * education-num <= 9.0
  * age <= 28.0
error: 0.0480
```

