# Tree Ensembles (ECML PKDD 2026)

Computing exact probable explanations for tree ensembles (such as `RandomForestClassifier` or `ExtraTreesClassifier`) is computationally intractable (#P-hard) due to combinatorial interactions between overlapping trees.

`PyProbXAI` introduces **$k$-pseudo-probable explanations** and **majoritary explanations** (Bounia, ECML PKDD 2026), providing an efficient polynomial-time approximation.

---

## 1. Weighted Tree Confidence

Let $F$ be an ensemble of trees $T_1, \dots, T_m$, predicting class $c = F(x)$. For each tree $T_i$, let $P_i$ denote its leaf confidence:

$$P_i = P(T_i(x) = c \mid x)$$

estimated by the empirical proportion of class $c$ samples in the leaf reached by $x$ during training.

### Weighted Disagreement Function $\mu(S)$

For an orthogonal DNF $\phi_i$ representing the disagreement of tree $T_i$ with class $c$, the ensemble error of condition set $S$ is defined as:

$$\mu(S) = \sum_{i=1}^m (1 - P_i) \cdot M(\phi_i \land S)$$

$$\epsilon(S) = \frac{\mu(S)}{|F_c| \cdot 2^{n - |S|}}$$

Trees that are more confident about the prediction contribute less penalty, while trees with low margin or high disagreement penalize the explanation more heavily.

---

## 2. Curvature-Based Greedy Ascent (`method="ga"`)

The default method for tree ensembles starts from the empty set $S = \emptyset$ and adds conditions greedily.

Unlike simple fixed-$k$ stopping, it employs a **curvature-based stopping rule**:

$$\gamma = \max\left(\frac{1}{e}, C_F\right)$$

The algorithm automatically detects the "elbow" where adding further conditions yields diminishing returns relative to the curvature of $\mu(S)$, ensuring sparse yet highly confident explanations.

---

## 3. Majority Explanations (`method="majority"`)

For voting-based ensembles, a user often wants a guarantee that *a strict majority of individual trees will agree with the prediction*:

- The algorithm considers the sub-forest of trees voting for the majority class $c$.
- It performs greedy ascent to minimize unweighted tree errors.
- It halts as soon as the exact error $W_f(\phi_i, S) = 0$ for at least $\lfloor m/2 \rfloor + 1$ trees.

This produces a **majoritary reason**: any instance satisfying these conditions is guaranteed to have a majority of trees voting for class $c$.

!!! note "Approximation Note"
    As discussed in the paper, this method provides an approximation of a majority explanation over the constituent trees. It is not an exact guarantee on the prediction of a scikit-learn forest (which combines leaf class probabilities rather than hard majority votes).

---

## References

- L. Bounia. *k-Pseudo-Probable Path Explanations for Weighted Tree-Based Ensembles: Bridging Interpretability and Computational Efficiency*. Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2026), pp. 39-57. Springer Nature Switzerland.

