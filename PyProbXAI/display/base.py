"""Core HTML shell rendering and base view classes for PyProbXAI visual representations.
"""
import html
import os
import re
from typing import Any, List, Optional, Tuple


def _escape(text: Any) -> str:
    """Escape text for safe HTML embedding."""
    return html.escape(str(text))


def _complete_method_name(method: str) -> str:
    """Convert raw method identifier to human-readable publication label."""
    m = str(method).lower()
    if m == "be":
        return "BanditExplain"
    if "uai2023-gd" in m or m == "gd":
        return "UAI 2023 (GD)"
    if "uai2023-ga" in m or m == "ga":
        return "UAI 2023 (GA)"
    if "kpseudo-ga" in m:
        return "k-pseudo (GA)"
    if "kpseudo-gd" in m:
        return "k-pseudo (GD)"
    if "majority" in m:
        return "UAI Majority"
    return str(method)


def _parse_condition(condition: str) -> Tuple[str, str, str]:
    """Parse a condition string into (feature_name, status, raw_condition).

    Examples
    --------
    "medical = present" -> ("medical", "present", "medical = present")
    "space = absent"    -> ("space", "absent", "space = absent")
    "age <= 30"         -> ("age", "<= 30", "age <= 30")
    """
    m = re.match(r"^(.*?)\s*=\s*(present|absent)$", condition.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).lower(), condition.strip()
    return condition.strip(), "rule", condition.strip()


def render_card_shell(
    card_title: str,
    method_str: str,
    precision: float,
    error: float,
    metrics_extra: str,
    predicted_class: Any,
    true_class: Optional[Any] = None,
    sections_html: Optional[List[str]] = None,
) -> str:
    """Single source of truth for the unified visual card layout and typography."""
    pred_str = _escape(predicted_class)
    error_val = float(error)
    prec_val = float(precision)
    precision_pct = prec_val * 100.0

    if prec_val >= 0.90:
        prec_color = "#15803d"  # green
    elif prec_val >= 0.75:
        prec_color = "#1d4ed8"  # blue
    else:
        prec_color = "#b45309"  # amber

    class_info = [f'<span><strong>Predicted class:</strong> <span style="font-weight:700;color:#0369a1;">{pred_str}</span></span>']
    if true_class is not None:
        class_info.append(f'<span><strong>True class:</strong> <span style="font-weight:700;color:#334155;">{_escape(true_class)}</span></span>')

    class_header_html = " &nbsp;|&nbsp; ".join(class_info)
    title_str = _escape(card_title) if card_title else f"Class {pred_str}"
    body_content = "".join(sections_html or [])

    out_html = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
            max-width:800px;margin:14px 0;background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.06);overflow:hidden;color:#1e293b;line-height:1.5;">
    <!-- Unified Header -->
    <div style="background:#f8fafc;padding:12px 18px;border-bottom:1px solid #e2e8f0;display:flex;
                justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:14px;font-weight:700;color:#0f172a;">{title_str} ({_escape(method_str)})</span>
        </div>
        <div style="font-size:13px;color:#334155;">
            {class_header_html}
        </div>
    </div>

    <!-- Metrics Bar (Clean typography, no progress bar) -->
    <div style="padding:10px 18px;border-bottom:1px solid #f1f5f9;display:flex;justify-content:space-between;
                align-items:baseline;font-size:13px;flex-wrap:wrap;gap:8px;background:#ffffff;">
        <div>
            Certified Precision: <strong style="font-size:15px;color:{prec_color};">{precision_pct:.1f}%</strong>
        </div>
        <div style="color:#64748b;">
            Bound Error &epsilon; &le; <strong>{error_val:.4f}</strong> &nbsp;|&nbsp; {metrics_extra}
        </div>
    </div>
    {body_content}
</div>
"""
    return out_html.strip()


def render_explanation_html(
    explanation: Any,
    title: Optional[str] = None,
    predicted_class: Optional[Any] = None,
    true_class: Optional[Any] = None,
    show_conditions: bool = True,
) -> str:
    """Render a clean, modern single-block HTML card for any generic Explanation object."""
    pred_val = predicted_class if predicted_class is not None else explanation.prediction
    method_str = _complete_method_name(explanation.method)
    metrics_extra = f"Size <em>k</em> = <strong>{explanation.size}</strong>"
    if explanation.n_sufficient_trees is not None:
        metrics_extra += f" &nbsp;|&nbsp; ({explanation.n_sufficient_trees}/{explanation.n_trees} trees at error 0)"

    sections: List[str] = []
    if show_conditions and explanation.conditions:
        present_conds = []
        absent_conds = []
        other_conds = []

        for c in explanation.conditions:
            feat, status, raw = _parse_condition(c)
            if status == "present":
                present_conds.append(feat)
            elif status == "absent":
                absent_conds.append(feat)
            else:
                other_conds.append(raw)

        badges_html = []
        if present_conds:
            p_pills = "".join(
                f'<span style="display:inline-flex;align-items:center;background:#dcfce7;color:#166534;'
                f'border:1px solid #bbf7d0;padding:2px 8px;border-radius:12px;font-size:12px;'
                f'font-weight:600;margin:2px 4px 2px 0;">{_escape(w)}</span>'
                for w in present_conds
            )
            badges_html.append(
                f'<div style="margin-top:6px;">'
                f'<span style="font-size:11px;font-weight:700;color:#15803d;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-right:8px;">Active Evidence (Present):</span>'
                f'{p_pills}'
                f'</div>'
            )

        if absent_conds:
            a_pills = "".join(
                f'<span style="display:inline-flex;align-items:center;background:#fee2e2;color:#991b1b;'
                f'border:1px solid #fecaca;padding:2px 8px;border-radius:12px;font-size:12px;'
                f'font-weight:600;margin:2px 4px 2px 0;">{_escape(w)}</span>'
                for w in absent_conds
            )
            badges_html.append(
                f'<div style="margin-top:6px;">'
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
                f'<div style="margin-top:6px;">'
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

    return render_card_shell(
        card_title=title or "Probable Explanation",
        method_str=method_str,
        precision=explanation.precision,
        error=explanation.error,
        metrics_extra=metrics_extra,
        predicted_class=pred_val,
        true_class=true_class,
        sections_html=sections,
    )


class BaseExplanationView:
    """Base class for interactive visual explanation views in Jupyter notebooks and standalone reports."""

    def __init__(
        self,
        explanation: Any,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
    ):
        self.explanation = explanation
        self.title = title
        self.predicted_class = predicted_class
        self.true_class = true_class

    def to_html(self) -> str:
        """Render self-contained HTML representation."""
        raise NotImplementedError("Subclasses must implement to_html()")

    def _repr_html_(self) -> str:
        """Native Jupyter Notebook HTML representation."""
        return self.to_html()

    def save(self, filepath: str) -> str:
        """Save representation as standalone HTML report."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".html", ".htm"):
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{_escape(self.title or 'PyProbXAI View')}</title>
</head>
<body style="background:#f1f5f9;padding:24px;display:flex;justify-content:center;">
    {self.to_html()}
</body>
</html>"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            return filepath
        raise ValueError(f"Unsupported file extension: {ext} (expected .html)")

    def __str__(self) -> str:
        return str(self.explanation)

