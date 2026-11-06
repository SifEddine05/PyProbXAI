"""Computer vision explanation rendering and spatial super-pixel heatmap plotting.
"""
import base64
import io
import os
import tempfile
from typing import Any, List, Optional, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "PyProbXAI_mpl")

import numpy as np

from .base import BaseExplanationView, _escape, _complete_method_name, render_card_shell


def _fig_to_base64_png(fig) -> str:
    """Serialize a matplotlib Figure to base64 Data URI string."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _infer_image_and_grid(
    image: np.ndarray,
    features: List[int],
    block_size: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, int, int, int, int]:
    """Infer image shape and block grid parameters (image_2d, rows, cols, block_h, block_w)."""
    img = np.asarray(image)
    if img.ndim == 1:
        dim = int(np.round(np.sqrt(len(img))))
        if dim * dim == len(img):
            img = img.reshape(dim, dim)
        else:
            raise ValueError(f"1-D image of length {len(img)} cannot be automatically reshaped to a square.")

    H, W = img.shape[:2]

    if grid_shape is not None:
        rows, cols = grid_shape
    elif block_size is not None:
        rows, cols = H // block_size, W // block_size
    else:
        max_f = max(features, default=0) + 1
        dim = int(np.ceil(np.sqrt(max_f)))
        dim = max(dim, 1)
        rows, cols = dim, dim

    bh = max(1, H // rows)
    bw = max(1, W // cols)
    return img, rows, cols, bh, bw


def plot_image_explanation(
    explanation: Any,
    image: np.ndarray,
    block_size: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (9.5, 3.2),
    alpha: float = 0.65,
    cmap_image: str = "gray",
    cmap_overlay: str = "Reds",
    show_grid: bool = True,
):
    """Generate and return a publication-ready 3-panel matplotlib Figure.

    Panels
    ------
    1. Original Input Image
    2. Selected Super-Pixel Blocks (Discrete Combinatorial Mask)
    3. Active Evidence Overlay (Overlay on Original Image with boundaries)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, rows, cols, bh, bw = _infer_image_and_grid(image, explanation.features, block_size, grid_shape)
    H, W = img.shape[:2]
    n_blocks = rows * cols

    block_mask = np.zeros(n_blocks)
    for f in explanation.features:
        if f < n_blocks:
            block_mask[f] = 1.0

    grid_mask = block_mask.reshape(rows, cols)
    heatmap = np.kron(grid_mask, np.ones((bh, bw)))
    heatmap_full = np.zeros((H, W))
    h_sub, w_sub = min(H, heatmap.shape[0]), min(W, heatmap.shape[1])
    heatmap_full[:h_sub, :w_sub] = heatmap[:h_sub, :w_sub]

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=120)

    # Panel 1: Original Image
    axes[0].imshow(img, cmap=cmap_image)
    axes[0].set_title("Original Input", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: Combinatorial Block Mask
    axes[1].imshow(grid_mask, cmap="Blues", interpolation="nearest")
    axes[1].set_title(f"Selected Blocks ({len(explanation.features)} / {n_blocks})",
                      fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")
    if show_grid and rows > 1 and cols > 1:
        for r in range(1, rows):
            axes[1].axhline(r - 0.5, color="#94a3b8", linestyle=":", linewidth=0.7, alpha=0.7)
        for c in range(1, cols):
            axes[1].axvline(c - 0.5, color="#94a3b8", linestyle=":", linewidth=0.7, alpha=0.7)

    # Panel 3: Active Overlay
    axes[2].imshow(img, cmap=cmap_image)
    axes[2].imshow(heatmap_full, cmap=cmap_overlay, alpha=alpha * heatmap_full, interpolation="nearest")
    axes[2].set_title("Active Evidence Overlay", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")
    if show_grid and rows > 1 and cols > 1:
        for r in range(1, rows):
            axes[2].axhline(r * bh - 0.5, color="#ffffff", linestyle=":", linewidth=0.7, alpha=0.5)
        for c in range(1, cols):
            axes[2].axvline(c * bw - 0.5, color="#ffffff", linestyle=":", linewidth=0.7, alpha=0.5)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


def render_image_explanation_html(
    explanation: Any,
    image: np.ndarray,
    block_size: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    predicted_class: Optional[Any] = None,
    true_class: Optional[Any] = None,
    **plot_kwargs,
) -> str:
    """Render a unified, single-block HTML view combining header metrics and the 3-panel figure."""
    fig = plot_image_explanation(
        explanation, image, block_size=block_size, grid_shape=grid_shape, title=None, **plot_kwargs
    )
    b64_img = _fig_to_base64_png(fig)

    pred_val = predicted_class if predicted_class is not None else explanation.prediction

    section = f"""
    <div style="padding:14px 18px;background:#ffffff;">
        <div style="font-size:12px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">
            Visual Evidence (Super-Pixel Blocks)
        </div>
        <div style="display:flex;justify-content:center;">
            <img src="{b64_img}" alt="PyProbXAI Image Heatmap"
                 style="max-width:100%;height:auto;border-radius:6px;" />
        </div>
    </div>
"""
    return render_card_shell(
        card_title=title or f"Class {_escape(pred_val)}",
        method_str=_complete_method_name(explanation.method),
        precision=explanation.precision,
        error=explanation.error,
        metrics_extra=f"Size <em>k</em> = <strong>{explanation.size}</strong>",
        predicted_class=pred_val,
        true_class=true_class,
        sections_html=[section],
    )


class ImageExplanationView(BaseExplanationView):
    """Interactive visual representation for Computer Vision explanations."""

    def __init__(
        self,
        explanation: Any,
        image: np.ndarray,
        block_size: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
        **plot_kwargs,
    ):
        super().__init__(
            explanation=explanation,
            title=title,
            predicted_class=predicted_class,
            true_class=true_class,
        )
        self.image = image
        self.block_size = block_size
        self.grid_shape = grid_shape
        self.plot_kwargs = plot_kwargs

    def plot(self):
        """Return the 3-panel matplotlib Figure."""
        return plot_image_explanation(
            self.explanation, self.image, block_size=self.block_size,
            grid_shape=self.grid_shape, title=self.title, **self.plot_kwargs
        )

    def to_html(self) -> str:
        """Return the self-contained unified HTML card."""
        return render_image_explanation_html(
            self.explanation, self.image, block_size=self.block_size,
            grid_shape=self.grid_shape, title=self.title,
            predicted_class=self.predicted_class, true_class=self.true_class,
            **self.plot_kwargs
        )

    def save(self, filepath: str) -> str:
        """Save as high-resolution figure (PDF, PNG, SVG) or standalone HTML report."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".pdf", ".png", ".svg", ".jpg", ".jpeg"):
            fig = self.plot()
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            return filepath
        return super().save(filepath)

    def __str__(self) -> str:
        return f"{self.explanation}\n[Image shape: {np.asarray(self.image).shape}]"

