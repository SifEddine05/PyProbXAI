# Decision Trees (UAI 2023)

Probable explanations for decision trees are based on **exact model counting** on Boolean variables and **supermodular minimization**, introduced by Bounia and Koriche (UAI 2023).

---

## 1. Boolean Formulation & Orthogonal DNFs

Given a decision tree $T$ and instance $x$:

1. **Binarization**: All decision thresholds in $T$ partition the continuous feature space into $n$ Boolean variables $v_1, \dots, v_n$.
2. **Direct Reason ($I_s$)**: The conjunction of all Boolean literals satisfied by $x$ along the branch that $x$ traverses to reach its leaf.
3. **Orthogonal DNF ($\phi$)**: All paths in $T$ that lead to a prediction *different* from $c = T(x)$ are collected. Because distinct branches in a decision tree are mutually exclusive, their union forms an **orthogonal Disjunctive Normal Form (ODNF)**:

$$\phi = \bigvee_{j=1}^m C_j \quad \text{where } C_j \land C_{j'} \equiv \bot \text{ for } j \neq j'$$

---

## 2. Exact Model Counting and Error Function

Because $\phi$ is orthogonal, model counting $\mu_{h,x}(S) = M(\phi \land S)$ (determining the number of satisfying assignments) can be computed in linear time:

$$\mu_{h,x}(S) = M(\phi \land S) = \sum_{j=1}^m M(C_j \land S)$$

For a term $S$ of size $|S|$, under the uniform distribution on the $n$ Boolean variables, the exact explanation error $\varepsilon_{h,x}(S)$ is:

$$\varepsilon_{h,x}(S) = \frac{\mu_{h,x}(S)}{2^{n - |S|}} = \frac{M(\phi \land S)}{2^{n - |S|}}$$

This provides the exact probability that the tree predicts an alternate class when conditions $S$ are enforced and all other Boolean conditions are varied uniformly.

---

## 3. Algorithms: Greedy Descent vs. Greedy Ascent

Finding the optimal subset $S \subseteq I_s$ with $|S| \le k$ that minimizes $\varepsilon_{h,x}(S)$ is NP-hard. However, the counting function $\mu_{h,x}$ exhibits supermodularity properties that guarantee strong theoretical approximation bounds for greedy algorithms:

### Greedy Descent (GD - `method="gd"`)
- **Initialization**: Starts from the full direct reason $S = I_s$ (which has zero error).
- **Step**: Greedily drops the literal $l \in S$ that minimizes the error increase:
  $$l^* = \arg\min_{l \in S} M(\phi \land (S \setminus \{l\}))$$
- **Termination**: Continues until $|S| \le k$.
- **Best Use Case**: Default for decision trees; works well because it prunes down from a known consistent premise.

### Greedy Ascent (GA - `method="ga"`)
- **Initialization**: Starts from the empty set $S = \emptyset$.
- **Step**: Greedily adds the literal $l \in I_s \setminus S$ that minimizes the error:
  $$l^* = \arg\min_{l \in I_s \setminus S} W_f(\phi, S \cup \{l\})$$
- **Termination**: Stops when $|S| = k$.
- **Best Use Case**: Fast heuristic when $k \ll |I_s|$.

---

## References

- L. Bounia, F. Koriche. *Approximating Probabilistic Explanations via Supermodular Minimization*. Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023), PMLR 216.

