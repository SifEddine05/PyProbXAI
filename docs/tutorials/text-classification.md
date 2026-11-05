# Tutorial: Text Classification with BanditExplain

This tutorial demonstrates explaining NLP models on documents from the 20 Newsgroups dataset (e.g. `sci.med` vs. `sci.space`).

---

## 1. Feature Representation

Each document is converted into binary indicator features representing the presence (1) or absence (0) of the most discriminant vocabulary terms:

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier
from PyProbXAI import BlackBoxExplainer

# Load 2 categories from 20 Newsgroups
train = fetch_20newsgroups(subset="train", categories=["sci.med", "sci.space"])
test = fetch_20newsgroups(subset="test", categories=["sci.med", "sci.space"])

# Extract top 50 discriminating words
vectorizer = CountVectorizer(binary=True, stop_words="english", min_df=5)
counts = vectorizer.fit_transform(train.data)
selector = SelectKBest(chi2, k=50).fit(counts, train.target)
words = list(vectorizer.get_feature_names_out()[selector.get_support()])

def to_features(docs):
    return selector.transform(vectorizer.transform(docs)).toarray().astype(float)

X_train = to_features(train.data)
X_test = to_features(test.data)

# Train a classifier
clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
clf.fit(X_train, train.target)
```

---

## 2. Explain Text Predictions

Because each feature represents binary presence/absence, we treat all columns as categorical:

```python
explainer = BlackBoxExplainer(
    clf,
    X_background=X_train,
    feature_names=words,
    categorical_features=list(range(len(words))),
    eta=0.15,
    delta=0.1,
    random_state=42
)

# Explain document
x = X_test[0]
exp = explainer.explain(x, k=5)
print(exp)
```

The output highlights key terms (e.g., `orbit = present`, `medical = absent`) that suffice to retain the predicted category with PAC probabilistic guarantees.

---

## 3. Interactive Document Highlight View

```python
# Render the text document with inline highlighted words and absence guards:
exp.show_text(raw_document_string)
```

> **Interactive Notebook:** Run [`examples/notebooks/tutorial_text_explanation.ipynb`](https://github.com/SifEddine05/PyProbXAI/blob/main/examples/notebooks/tutorial_text_explanation.ipynb) to interactively inspect text explanations and document highlighting.

