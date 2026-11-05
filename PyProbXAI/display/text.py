"""Text explanation rendering and interactive document viewer for NLP classifiers.
"""
import re
from typing import Any, Optional

from .base import BaseExplanationView, _escape, _complete_method_name, _parse_condition, render_card_shell


def render_text_explanation_html(
    explanation: Any,
    text: str,
    title: Optional[str] = None,
    predicted_class: Optional[Any] = None,
    true_class: Optional[Any] = None,
    max_height: str = "320px",
) -> str:
    """Render a unified, single-block HTML card combining header metrics, guards, and document reader."""
    present_words = set()
    absent_words = set()
    other_conds = []

    for c in explanation.conditions:
        feat, status, raw = _parse_condition(c)
        if status == "present":
            present_words.add(feat.lower())
        elif status == "absent":
            absent_words.add(feat.lower())
        else:
            other_conds.append(raw)

    raw_text = text.strip()
    is_truncated = False
    if len(raw_text) > 3000:
        raw_text = raw_text[:3000]
        is_truncated = True

    clean_text = _escape(raw_text)

    if present_words:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in sorted(present_words, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )

        def replacer(match):
            word = match.group(0)
            return (
                f'<mark style="background:#dcfce7;color:#166534;padding:2px 5px;border-radius:3px;'
                f'font-weight:600;border-bottom:2px solid #86efac;"'
                f' title="Certified Evidence: &quot;{word}&quot;">{word}</mark>'
            )

        highlighted_body = pattern.sub(replacer, clean_text)
    else:
        highlighted_body = clean_text

    formatted_body = highlighted_body.replace("\n\n", "<br><br>").replace("\n", "<br>")
    if is_truncated:
        formatted_body += '<br><br><span style="color:#64748b;font-style:italic;">... [Document truncated after 3,000 characters for display]</span>'

    sections = []

    # 1. Condition & Guard Badges
    badges_html = []
    if present_words:
        p_pills = "".join(
            f'<span style="display:inline-flex;align-items:center;background:#dcfce7;color:#166534;'
            f'border:1px solid #bbf7d0;padding:2px 8px;border-radius:12px;font-size:12px;'
            f'font-weight:600;margin:2px 4px 2px 0;">{_escape(w)}</span>'
            for w in sorted(present_words)
        )
        badges_html.append(
            f'<div style="margin-top:4px;">'
            f'<span style="font-size:11px;font-weight:700;color:#15803d;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-right:8px;">Active Evidence (Present):</span>'
            f'{p_pills}'
            f'</div>'
        )

    if absent_words:
        a_pills = "".join(
            f'<span style="display:inline-flex;align-items:center;background:#fee2e2;color:#991b1b;'
            f'border:1px solid #fecaca;padding:2px 8px;border-radius:12px;font-size:12px;'
            f'font-weight:600;margin:2px 4px 2px 0;">{_escape(w)}</span>'
            for w in sorted(absent_words)
        )
        badges_html.append(
            f'<div style="margin-top:4px;">'
            f'<span style="font-size:11px;font-weight:700;color:#b91c1c;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-right:8px;">Absence Guards (Forbidden):</span>'
            f'{a_pills}'
            f'</div>'
        )

    if other_conds:
        o_pills = "".join(
            f'<span style="display:inline-flex;align-items:center;background:#f1f5f9;color:#334155;'
            f'border:1px solid #cbd5e1;padding:2px 8px;border-radius:12px;font-size:12px;'
            f'font-weight:600;margin:2px 4px 2px 0;">{_escape(c)}</span>'
            for c in other_conds
        )
        badges_html.append(
            f'<div style="margin-top:4px;">'
            f'<span style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-right:8px;">Conditions:</span>'
            f'{o_pills}'
            f'</div>'
        )

    if badges_html:
        sections.append(
            f'<div style="padding:10px 18px;background:#ffffff;border-top:1px solid #f1f5f9;">'
            f'{"".join(badges_html)}'
            f'</div>'
        )

    # 2. Document Reader Panel
    max_h_style = f"max-height:{max_height};overflow-y:auto;" if max_height and max_height != "none" else ""
    sections.append(f"""
    <div style="border-top:1px solid #e2e8f0;background:#ffffff;">
        <div style="background:#f8fafc;padding:8px 18px;border-bottom:1px solid #e2e8f0;display:flex;
                    justify-content:space-between;align-items:center;">
            <span style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">
                Document Content &amp; Evidence Spans
            </span>
            <span style="font-size:11px;color:#64748b;">
                Green spans = certified active evidence
            </span>
        </div>
        <div style="padding:16px 18px;{max_h_style}font-size:13.5px;
                    line-height:1.65;color:#1e293b;background:#ffffff;white-space:normal;word-break:break-word;">
            {formatted_body}
        </div>
    </div>
""")

    pred_val = predicted_class if predicted_class is not None else explanation.prediction
    return render_card_shell(
        card_title=title or f"Class {_escape(pred_val)}",
        method_str=_complete_method_name(explanation.method),
        precision=explanation.precision,
        error=explanation.error,
        metrics_extra=f"Size <em>k</em> = <strong>{explanation.size}</strong>",
        predicted_class=pred_val,
        true_class=true_class,
        sections_html=sections,
    )


class TextExplanationView(BaseExplanationView):
    """A visual wrapper for an Explanation applied to a specific text document."""

    def __init__(
        self,
        explanation: Any,
        text: str,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
        max_height: str = "320px",
    ):
        super().__init__(
            explanation=explanation,
            title=title,
            predicted_class=predicted_class,
            true_class=true_class,
        )
        self.text = text
        self.max_height = max_height

    def to_html(self, max_height: Optional[str] = None) -> str:
        """Render self-contained HTML representation."""
        return render_text_explanation_html(
            self.explanation,
            self.text,
            title=self.title,
            predicted_class=self.predicted_class,
            true_class=self.true_class,
            max_height=max_height or self.max_height,
        )

    def __str__(self) -> str:
        return f"{self.explanation}\n\n[Document Preview]\n{self.text[:300]}..."

