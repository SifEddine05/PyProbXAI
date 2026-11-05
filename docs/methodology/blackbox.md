# Black-Box Models (BanditExplain)

When a model is non-tree-based (such as deep neural networks, CNNs, SVMs, or external API endpoints), internal representations and decision paths cannot be examined.

`PyProbXAI` includes **BanditExplain (BE)**, a model-agnostic explainer that requires only the `.predict()` function and background data, equipped with formal $(\eta, \delta)$-PAC guarantees.

---

## 1. Feature Discretization

To form interpretable symbolic conditions from continuous inputs:

- **Numeric features**: Partitioned into quantile intervals using `n_bins` (default $3$: Low, Medium, High).
- **Categorical features**: Matched on exact category equality.

Each instance $x$ is converted into $d$ candidate literals $\{l_1, \dots, l_d\}$.

---

## 2. Monte-Carlo Background Sampling

To evaluate the error of candidate condition set $S$:

1. Features specified in $S$ are held fixed to the values of $x$.
2. All non-fixed features $\bar{S}$ are drawn randomly from rows of the background dataset $X_{\text{background}}$.
3. The model predicts on the perturbed samples.
4. The empirical error is the fraction of samples where $f(x') \neq f(x)$.

---

## 3. $(\eta, \delta)$-PAC Guarantee via Multi-Armed Bandits

BanditExplain formulates the search for optimal conditions as a multi-armed bandit problem. Rather than sampling uniformly across all combinations, it adaptively samples candidates to eliminate suboptimal literals quickly.

### Sample Complexity Bound

To ensure that the estimated error $\hat{\epsilon}(S)$ is within $\eta$ of the true distribution error $\epsilon(S)$ with confidence at least $1 - \delta$:

$$N = \left\lceil \frac{2}{\eta^2} \ln \left( \frac{2m}{\delta_{\text{step}}} \right) \right\rceil, \quad \delta_{\text{step}} = \frac{\delta}{d + 1}$$

where:
- $\eta$: Maximum permissible error tolerance (default: $0.1$).
- $\delta$: Maximum probability of failure (default: $0.05$).
- $d$: Total number of features.
- $m$: Number of competing candidates evaluated at each bandit round.

This PAC guarantee ensures that explanations produced by `BlackBoxExplainer` are statistically robust and reproducible.

---

## References

- S. E. Sellami, L. Bounia, J. Agoun, C. Rouveirol. *BanditExplain: Scalable Probabilistic Explanations with PAC Guarantees via Multi-Armed Bandits*.

