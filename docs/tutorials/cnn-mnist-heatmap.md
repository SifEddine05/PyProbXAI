# Tutorial: Image Heatmaps on MNIST with CNNs

This tutorial demonstrates how `BlackBoxExplainer` explains image classifications by treating super-pixels (blocks of pixels) as numeric features, and rendering the selected conditions as visual heatmaps.

---

## 1. Super-Pixel Pooling

Images from MNIST ($28 \times 28$) can be averaged in blocks of $4 \times 4$ pixels to form a $7 \times 7$ grid of $49$ super-pixel features:

```python
import numpy as np
import torch
from torch import nn
from PyProbXAI import BlackBoxExplainer

def pool_images(images_flat, block_size=4, img_shape=(28, 28)):
    H, W = img_shape
    rows, cols = H // block_size, W // block_size
    images = images_flat.reshape(-1, H, W)[:, :rows * block_size, :cols * block_size]
    pooled = images.reshape(-1, rows, block_size, cols, block_size).mean(axis=(2, 4))
    return pooled.reshape(-1, rows * cols), rows, cols
```

---

## 2. Train a PyTorch CNN

Wrap the CNN's forward pass in a standard `.predict()` interface:

```python
class CNNWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            logits = self.model(tensor_x)
            return logits.argmax(dim=1).cpu().numpy()
```

---

## 3. Generate Probable Explanations & Plot Heatmaps

```python
# Initialize BE on pooled training background data
explainer = BlackBoxExplainer(
    wrapped_cnn,
    X_background=X_train_pooled,
    n_bins=3,
    eta=0.15,
    delta=0.1,
    random_state=42
)

# Explain image x
exp = explainer.explain(x_pooled, k=7)
print(exp)
```

The resulting conditions identify the critical blocks (e.g., the top loop distinguishing a 3 from an 8) required to preserve the CNN's classification with $>90\%$ probability.

