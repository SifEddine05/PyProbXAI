# Installation

`PyProbXAI` requires **Python >= 3.9**. The core library has minimal dependencies: only `numpy` and `scikit-learn>=1.0`.

---

## 1. Install via pip

To install the latest stable version from PyPI:

```bash
pip install PyProbXAI
```

---

## 2. Optional Dependency Groups

`PyProbXAI` organizes dependencies into optional groups depending on your use case:

=== "Full Setup (Recommended for Reproducibility)"
    ```bash
    pip install "PyProbXAI[all]"
    ```
    Includes documentation generators, test runners, and dependencies needed to run all scientific examples (pandas, matplotlib, PyTorch).

=== "Scientific Examples"
    ```bash
    pip install "PyProbXAI[examples]"
    ```
    Installs `pandas`, `matplotlib`, and `torch` to reproduce the demo scripts (tabular, neural network, CNN image heatmaps).

=== "Documentation Building"
    ```bash
    pip install "PyProbXAI[docs]"
    ```
    Installs `mkdocs-material`, `mkdocstrings`, and `pymdown-extensions`.

---

## 3. Install from Source (Development)

If you are developing algorithms or contributing to `PyProbXAI`:

```bash
# Clone the repository
git clone https://github.com/SifEddine05/PyProbXAI.git
cd PyProbXAI

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install in editable mode
pip install -e .
```

