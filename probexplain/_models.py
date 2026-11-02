"""Which explainer is adapted to which model.

  * DecisionTreeClassifier                                   -> DecisionTreeExplainer
  * RandomForest / ExtraTrees / Bagging of decision trees    -> TreeEnsembleExplainer
  * any other classifier                                     -> BlackBoxExplainer
"""
import warnings

from sklearn.tree import DecisionTreeClassifier


def is_tree_ensemble(model):
    estimators = getattr(model, "estimators_", None)
    return (isinstance(estimators, list) and len(estimators) > 0 and hasattr(model, "classes_")
            and all(isinstance(e, DecisionTreeClassifier) for e in estimators))


def adapted_explainer(model):
    """Name of the explainer adapted to ``model``."""
    if isinstance(model, DecisionTreeClassifier):
        return "DecisionTreeExplainer"
    if is_tree_ensemble(model):
        return "TreeEnsembleExplainer"
    return "BlackBoxExplainer"


def _message(model, explainer, right):
    return (f"{explainer} is not adapted to a {type(model).__name__}: use {right} instead "
            f"(or probexplain.explain, which chooses the explainer from the model)")


def check_model(model, explainer):
    """Raise a TypeError when ``explainer`` (a class name) is not the one adapted to ``model``.

    BlackBoxExplainer works with any model, so for a tree model it only warns."""
    right = adapted_explainer(model)
    if right == explainer:
        return
    if explainer == "BlackBoxExplainer":
        warnings.warn(_message(model, explainer, right) + "; BE only gives estimated errors here")
    else:
        raise TypeError(_message(model, explainer, right))
