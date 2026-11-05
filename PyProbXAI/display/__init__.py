"""Visual displays and report generation for PyProbXAI explanations across modalities.
"""
from .base import BaseExplanationView, render_card_shell, render_explanation_html
from .tabular import TabularExplanationView, render_tabular_explanation_html
from .text import TextExplanationView, render_text_explanation_html
from .vision import ImageExplanationView, plot_image_explanation, render_image_explanation_html

__all__ = [
    "BaseExplanationView",
    "render_card_shell",
    "render_explanation_html",
    "TabularExplanationView",
    "render_tabular_explanation_html",
    "TextExplanationView",
    "render_text_explanation_html",
    "ImageExplanationView",
    "render_image_explanation_html",
    "plot_image_explanation",
]

