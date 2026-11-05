"""PyProbXAI: probable explanations of classifiers.

  * decision trees           -> Submodular probable explanations       (DecisionTreeExplainer)
  * tree ensembles           -> k-pseudo-probable explanations       (TreeEnsembleExplainer)
  * any other (black box)    -> BE, BanditExplain PAC explanations   (BlackBoxExplainer)

>>> from PyProbXAI import explain
>>> e = explain(model, x, k=4, X_background=X_train)
>>> print(e)
"""
from ._models import adapted_explainer
from .blackbox import BlackBoxExplainer
from .ensemble import TreeEnsembleExplainer
from .explanation import Explanation
from .display import (
    BaseExplanationView,
    ImageExplanationView,
    TabularExplanationView,
    TextExplanationView,
    plot_image_explanation,
    render_explanation_html,
    render_image_explanation_html,
    render_tabular_explanation_html,
    render_text_explanation_html,
)
from .tree import DecisionTreeExplainer

__all__ = [
    "explain",
    "make_explainer",
    "Explanation",
    "DecisionTreeExplainer",
    "TreeEnsembleExplainer",
    "BlackBoxExplainer",
    "TextExplanationView",
    "ImageExplanationView",
    "TabularExplanationView",
    "render_explanation_html",
    "render_text_explanation_html",
    "render_image_explanation_html",
    "render_tabular_explanation_html",
    "plot_image_explanation",
]
__version__ = "0.1.0"


def make_explainer(model, X_background=None, **kwargs):
    """Explainer adapted to the model: Submodular for a DecisionTreeClassifier, k-pseudo for a forest
    of decision trees (RandomForest, ExtraTrees, Bagging), BE for anything else.

    kwargs are passed to the explainer constructor (feature_names, categorical_features, n_bins,
    eta, delta, cat_decoders, random_state)."""
    right = adapted_explainer(model)
    if right == "DecisionTreeExplainer":
        return DecisionTreeExplainer(model, feature_names=kwargs.get("feature_names"))
    if right == "TreeEnsembleExplainer":
        return TreeEnsembleExplainer(model, feature_names=kwargs.get("feature_names"))
    if X_background is None:
        raise ValueError(f"{type(model).__name__} is explained with BE (black box): "
                         "X_background (e.g. the training data) is required")
    return BlackBoxExplainer(model, X_background, **kwargs)


def explain(model, x, k, method=None, X_background=None, **kwargs):
    """Explain the prediction of ``model`` on instance ``x`` with an explanation of size <= k.

    method: decision tree "gd" (default) or "ga";
            tree ensemble "ga" (default), "gd" or "majority";
            black box: ignored (BE).
    Other keyword arguments go to the explainer constructor (see make_explainer)."""
    explainer = make_explainer(model, X_background, **kwargs)
    if isinstance(explainer, BlackBoxExplainer) or method is None:
        return explainer.explain(x, k)
    return explainer.explain(x, k, method=method)
