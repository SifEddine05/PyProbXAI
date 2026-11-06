"""Tabular explanation rendering and feature invariance inspection.
"""
import re
from typing import Any, List, Optional, Tuple

import numpy as np

from .base import BaseExplanationView, _escape, _complete_method_name, render_card_shell


def _parse_tabular_condition(condition: str) -> Tuple[str, str]:
    """Parse a tabular condition string into (feature_name, constraint)."""
    c = condition.strip()
    # Pattern: low <=/< feat <=/< high
    m_range = re.match(r"^(.*?)\s*(<=|<)\s*([a-zA-Z0-9_\-\.]+)\s*(<=|<)\s*(.*?)$", c)
    if m_range:
        return m_range.group(3).strip(), c
    # Pattern: feat (op) value
    m_op = re.match(r"^([a-zA-Z0-9_\-\.]+)\s*(=|<=|<|>=|>|in)\s*(.*?)$", c)
    if m_op:
        op = m_op.group(2)
        val = m_op.group(3).strip()
        return m_op.group(1).strip(), f"{op} {val}"
    return c, c


def _extract_tabular_data(
    x: Optional[Any],
    feature_names: Optional[List[str]],
) -> Tuple[Optional[List[str]], Optional[List[Any]]]:
    """Normalize input instance x and feature names into parallel lists."""
    if x is None:
        return (list(feature_names) if feature_names is not None else None, None)

    # Pandas Series
    if hasattr(x, "index") and hasattr(x, "values") and not hasattr(x, "columns"):
        names = list(feature_names) if feature_names is not None else [str(k) for k in x.index]
        vals = list(x.values)
        return names, vals

    # Pandas DataFrame (1-row)
    if hasattr(x, "iloc") and hasattr(x, "columns"):
        names = list(feature_names) if feature_names is not None else [str(k) for k in x.columns]
        vals = list(x.iloc[0].values)
        return names, vals

    # Dictionary
    if isinstance(x, dict):
        names = list(feature_names) if feature_names is not None else list(x.keys())
        vals = [x.get(k) for k in names]
        return names, vals

    # NumPy array, list, or tuple
    vals = list(x)
    if feature_names is not None:
        names = list(feature_names)
    else:
        names = [f"Feature {i}" for i in range(len(vals))]
    return names, vals


def _format_tabular_val(val: Any) -> str:
    """Format an instance feature value nicely for display."""
    if val is None:
        return "&mdash;"
    if isinstance(val, (bool, np.bool_)):
        return str(val)
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if float(val).is_integer():
            return str(int(val))
        return f"{val:.4g}"
    return _escape(str(val))


def render_tabular_explanation_html(
    explanation: Any,
    x: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    predicted_class: Optional[Any] = None,
    true_class: Optional[Any] = None,
) -> str:
    """Render a unified, publication-grade HTML card for tabular explanations."""
    names, vals = _extract_tabular_data(x, feature_names)

    active_rows: List[Tuple[str, Optional[Any], str]] = []
    active_indices = set()

    if names is not None and vals is not None:
        # Match each condition by feature name first (robust to sort order differences)
        for i, cond in enumerate(explanation.conditions):
            parsed_name, constraint = _parse_tabular_condition(cond)
            if parsed_name in names:
                idx = names.index(parsed_name)
                active_rows.append((parsed_name, vals[idx], constraint))
                active_indices.add(idx)
            elif explanation.features and i < len(explanation.features):
                f_idx = explanation.features[i]
                if f_idx < len(names):
                    active_rows.append((names[f_idx], vals[f_idx], constraint))
                    active_indices.add(f_idx)
                else:
                    active_rows.append((parsed_name, None, constraint))
            else:
                active_rows.append((parsed_name, None, constraint))
    else:
        # No instance x provided: render condition rules
        for cond in explanation.conditions:
            parsed_name, constraint = _parse_tabular_condition(cond)
            active_rows.append((parsed_name, None, constraint))

    has_vals = any(inst_val is not None for _, inst_val, _ in active_rows)

    # 1. Table rows
    rows_html_list = []
    for f_name, inst_val, constraint in active_rows:
        val_td = (
            f'<td style="padding:10px 18px;color:#1e293b;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px;">'
            f'{_format_tabular_val(inst_val)}</td>'
            if has_vals else ""
        )
        rows_html_list.append(
            f'<tr style="border-bottom:1px solid #f1f5f9;background:#ffffff;">'
            f'<td style="padding:10px 18px;font-weight:600;color:#0f172a;">{_escape(f_name)}</td>'
            f'{val_td}'
            f'<td style="padding:10px 18px;color:#0369a1;font-weight:600;">{_escape(constraint)}</td>'
            f'<td style="padding:10px 18px;text-align:right;">'
            f'<span style="display:inline-block;background:#dcfce7;color:#166534;border:1px solid #bbf7d0;'
            f'padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">'
            f'Certified Bound</span>'
            f'</td>'
            f'</tr>'
        )

    val_th = '<th style="padding:10px 18px;font-weight:700;">Instance Value</th>' if has_vals else ""

    sections = [
        f"""
    <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;">
            <thead>
                <tr style="background:#f8fafc;border-bottom:1px solid #e2e8f0;color:#475569;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">
                    <th style="padding:10px 18px;font-weight:700;">Feature</th>
                    {val_th}
                    <th style="padding:10px 18px;font-weight:700;">Active Constraint</th>
                    <th style="padding:10px 18px;font-weight:700;text-align:right;">Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html_list)}
            </tbody>
        </table>
    </div>
"""
    ]

    total_features = len(names) if names is not None else None
    if names is not None and vals is not None and total_features and total_features > len(active_indices):
        n_free = total_features - len(active_indices)
        inv_pct = (n_free / total_features) * 100.0
        free_pills = []
        for i in range(total_features):
            if i not in active_indices:
                f_name = names[i]
                v = vals[i]
                free_pills.append(
                    f'<span style="display:inline-flex;align-items:center;background:#ffffff;color:#334155;'
                    f'border:1px solid #cbd5e1;padding:2px 8px;border-radius:6px;font-size:11.5px;">'
                    f'<strong>{_escape(f_name)}</strong>&nbsp;=&nbsp;'
                    f'<span style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace;">{_format_tabular_val(v)}</span>'
                    f'</span>'
                )

        metrics_extra = f"Size <em>k</em> = <strong>{explanation.size}</strong> / <strong>{total_features}</strong> features ({inv_pct:.1f}% invariant)"
        if explanation.n_sufficient_trees is not None:
            metrics_extra += f" &nbsp;|&nbsp; Trees: <strong>{explanation.n_sufficient_trees}/{explanation.n_trees}</strong>"

        sections.append(
            f"""
    <div style="background:#f8fafc;padding:12px 18px;border-top:1px solid #e2e8f0;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <span style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:0.5px;">
                Invariant / Free Features ({n_free} / {total_features} &bull; {inv_pct:.1f}% Invariant)
            </span>
            <span style="font-size:11px;color:#64748b;">
                Unconstrained features
            </span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;max-height:160px;overflow-y:auto;">
            {''.join(free_pills)}
        </div>
        <div style="font-size:11.5px;color:#64748b;margin-top:8px;">
            The model&#39;s prediction is certified invariant to arbitrary perturbations across these {n_free} unconstrained features.
        </div>
    </div>
"""
        )
    else:
        metrics_extra = f"Size <em>k</em> = <strong>{explanation.size}</strong>"
        if explanation.n_sufficient_trees is not None:
            metrics_extra += f" &nbsp;|&nbsp; Trees: <strong>{explanation.n_sufficient_trees}/{explanation.n_trees}</strong>"

    pred_val = predicted_class if predicted_class is not None else explanation.prediction

    return render_card_shell(
        card_title=title or f"Class {_escape(pred_val)}",
        method_str=_complete_method_name(explanation.method),
        precision=explanation.precision,
        error=explanation.error,
        metrics_extra=metrics_extra,
        predicted_class=pred_val,
        true_class=true_class,
        sections_html=sections,
    )


class TabularExplanationView(BaseExplanationView):
    """A visual wrapper for an Explanation applied to a tabular data instance."""

    def __init__(
        self,
        explanation: Any,
        x: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        title: Optional[str] = None,
        predicted_class: Optional[Any] = None,
        true_class: Optional[Any] = None,
    ):
        super().__init__(
            explanation=explanation,
            title=title,
            predicted_class=predicted_class,
            true_class=true_class,
        )
        self.x = x
        self.feature_names = feature_names

    def to_html(self) -> str:
        """Return the self-contained unified HTML card."""
        return render_tabular_explanation_html(
            self.explanation,
            x=self.x,
            feature_names=self.feature_names,
            title=self.title,
            predicted_class=self.predicted_class,
            true_class=self.true_class,
        )

