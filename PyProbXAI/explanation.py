"""Result object returned by every explainer."""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


def predict_one(model, x):
    """model.predict on a single instance (as a DataFrame if the model was fit on one)."""
    X = np.asarray(x, dtype=float).reshape(1, -1)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        import pandas as pd
        X = pd.DataFrame(X, columns=names)
    return model.predict(X)[0]


@dataclass
class Explanation:
    """A (probable / pseudo-probable / PAC) explanation of a prediction.

    Attributes
    ----------
    method : str        algorithm used, e.g. "submodular-ga", "k-pseudo-gd", "be"
    prediction : Any    the predicted class being explained
    literals : list     the explanation, as Boolean literals of the model's binarization
    features : list     indices of the original features involved
    conditions : list   human readable conditions (x[f] > t, intervals, categorical values...)
    error : float       error of the explanation: probability (Submodular), weighted pseudo-probability
                        (k-pseudo), average of the tree errors (majority) or Monte-Carlo estimate (BE)
                        that the prediction changes when the features outside the explanation vary
    n_sufficient_trees : int or None   majority only: number of trees with error 0
    n_trees : int or None              majority only: number of trees of the ensemble
    """
    method: str
    prediction: Any
    literals: List[int]
    features: List[int]
    conditions: List[str]
    error: float
    n_sufficient_trees: Optional[int] = None
    n_trees: Optional[int] = None

    @property
    def size(self):
        return len(self.literals)

    @property
    def precision(self):
        """Deprecated / convenience property: 1.0 - error."""
        return 1.0 - self.error

    def __str__(self):
        lines = [f"Explanation [{self.method}] of prediction {self.prediction!r}",
                 f"  size = {self.size}, error = {self.error:.6g}"]
        if self.n_sufficient_trees is not None:
            lines.append(f"  trees with error 0: {self.n_sufficient_trees} / {self.n_trees}")
        lines += [f"  * {c}" for c in self.conditions] or ["  (empty explanation)"]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display in Jupyter / IPython environments."""
        from .display import render_explanation_html
        return render_explanation_html(self)

    def show_tabular(
        self,
        x: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
    ):
        """Return an interactive TabularExplanationView highlighting active and invariant features."""
        from .display import TabularExplanationView
        return TabularExplanationView(
            self,
            x=x,
            feature_names=feature_names,
            title=title,
            predicted_class=predicted_class,
            true_class=true_class,
        )

    def show_text(
        self,
        text: str,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
    ):
        """Return a visual TextExplanationView highlighting conditions in a text document."""
        from .display import TextExplanationView
        return TextExplanationView(
            self, text, title=title, predicted_class=predicted_class, true_class=true_class
        )

    def show_image(
        self,
        image: np.ndarray,
        block_size: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
        **kwargs,
    ):
        """Return an interactive ImageExplanationView with embedded 3-panel heatmaps in a unified card."""
        from .display import ImageExplanationView
        return ImageExplanationView(
            self, image, block_size=block_size, grid_shape=grid_shape,
            title=title, predicted_class=predicted_class, true_class=true_class, **kwargs
        )

    def plot_image(
        self,
        image: np.ndarray,
        block_size: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        **kwargs,
    ):
        """Generate and return a 3-panel matplotlib Figure for this image explanation."""
        from .display import plot_image_explanation
        return plot_image_explanation(
            self, image, block_size=block_size, grid_shape=grid_shape, title=title, **kwargs
        )
