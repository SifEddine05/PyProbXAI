# Visual Displays & Interactive Views

The `PyProbXAI.display` module provides interactive HTML cards and visualizers for tabular data, text documents, and computer vision heatmaps.

These views are accessible directly from any `Explanation` instance via `exp.show_tabular()`, `exp.show_text()`, `exp.show_image()`, or `exp.plot_image()`.

---

::: PyProbXAI.display.TabularExplanationView
    options:
      show_root_heading: true
      members:
        - to_html
        - save

---

::: PyProbXAI.display.TextExplanationView
    options:
      show_root_heading: true
      members:
        - to_html
        - save

---

::: PyProbXAI.display.ImageExplanationView
    options:
      show_root_heading: true
      members:
        - to_html
        - save

---

::: PyProbXAI.display.plot_image_explanation
    options:
      show_root_heading: true

