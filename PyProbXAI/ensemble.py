"""k-pseudo-probable explanations for weighted tree ensembles.

For a forest F_c of trees T_i and the predicted class c = F(x), each tree is weighted by its
confidence P_i = P(T_i(x) = F(x) | x), estimated by the proportion of the majority class in the
leaf reached by x. The error of a term S is

    mu(S)      = sum_i (1 - P_i) * M(phi_i ∧ S)   phi_i: orthogonal DNF of "T_i does not predict c"
    epsilon(S) = mu(S) / (|F_c| * 2^(n - |S|))

Methods
  * "ga"        greedy ascent with the curvature based stopping rule
  * "gd"        greedy descent
  * "majority"  greedy ascent of Submodular on the trees voting for c, stopped as soon as the error
                (unweighted, W_f of Submodular) of at least half of the trees is 0: approximation of a
                majoritary reason
"""
import math
import warnings

import numpy as np

from ._boolean import Binarization, BooleanTree
from ._counting import M_s, W_s, log
from ._models import check_model
from .explanation import Explanation, predict_one


def mu_forest(S, Phi, weights, n):
    t = list(S)
    return sum(w * M_s(phi, t, n) for phi, w in zip(Phi, weights))


def epsilon_forest(S, Phi, weights, n):
    return mu_forest(S, Phi, weights, n) / (len(Phi) * (1 << (n - len(S))))


C_F = 1    # constant of the curvature based stopping rule


def greedy_ascent(k, I, Phi, weights, n):
    S = []
    history = [S.copy()]

    mu_empty = mu_forest([], Phi, weights, n)
    gamma = max(1 / math.e, C_F)

    while len(S) < k:
        best_i, best_val = None, float('inf')
        for i in I:
            if i not in S:
                val = mu_forest(S + [i], Phi, weights, n)
                if val < best_val:
                    best_val, best_i = val, i
        if best_i is None:   # every candidate already taken
            break

        S = S + [best_i]
        history.append(S.copy())

        mu_S = mu_forest(S, Phi, weights, n)
        if mu_S == 0:
            break
        limit = k * math.ceil(log(mu_empty) - log(mu_S) - math.log(gamma))
        if len(S) >= limit:
            break

    return min(history, key=lambda s: epsilon_forest(s, Phi, weights, n))


def greedy_descent(k, I, Phi, weights, n):
    S = list(I)
    history = [S.copy()]

    while len(S) > 0:
        best_i, best_val = None, float('inf')
        for i in S:
            val = mu_forest([x for x in S if x != i], Phi, weights, n)
            if val < best_val:
                best_val, best_i = val, i

        S = [x for x in S if x != best_i]
        history.append(S.copy())

    # best of the solutions of size <= k
    candidates = [s for s in history if len(s) <= k]
    return min(candidates, key=lambda s: epsilon_forest(s, Phi, weights, n))


def mean_error(S, Phi, n):
    """Average over the trees of their Submodular error W_f (no weights)."""
    return sum(W_s(phi, S, n) for phi in Phi) / len(Phi)


def majority_ascent(I, Phi, voters, n, q, k=None):
    """Greedy ascent of Submodular stopped when at least q trees have error 0.

    A tree has error 0 when S implies its prediction (M(phi_i ∧ S) = 0). Only the trees in
    ``voters`` (those predicting c on x) can reach 0, since S is made of literals of x. At each
    step, the literal minimizing the sum of M(phi_i ∧ S) over the voters not solved yet is added.
    Returns (S, the set of trees with error 0)."""
    S, X = [], list(I)
    solved = {i for i in voters if M_s(Phi[i], S, n) == 0}
    while len(solved) < q and X and (k is None or len(S) < k):
        open_ = [i for i in voters if i not in solved]
        vals = [sum(M_s(Phi[i], S + [l], n) for i in open_) for l in X]
        e = X[min(range(len(X)), key=vals.__getitem__)]
        S = S + [e]
        X.remove(e)
        # adding literals only removes models: a solved tree stays solved
        solved |= {i for i in open_ if M_s(Phi[i], S, n) == 0}
    return S, solved


def curvature(I, Phi, weights, n):
    t_I = list(I)
    mu_I = mu_forest(t_I, Phi, weights, n)
    ratios = []
    for i in I:
        Li_full = mu_forest([x for x in t_I if x != i], Phi, weights, n) - mu_I
        Li_single = mu_forest([], Phi, weights, n) - mu_forest([i], Phi, weights, n)
        if Li_single > 0:
            ratios.append(Li_full / Li_single)
    return 1.0 - float(min(ratios)) if ratios else 1.0


def _trees_of(model):
    """(estimator, class_map, feature_map) for the supported scikit-learn ensembles."""
    feats = getattr(model, "estimators_features_", None)    # BaggingClassifier
    out = []
    for i, est in enumerate(model.estimators_):
        if not hasattr(est, "tree_") or not hasattr(est, "classes_"):
            raise TypeError(f"{type(model).__name__}: estimator {i} is not a decision tree classifier")
        out.append((est, est.classes_, None if feats is None else feats[i]))
    return out


class TreeEnsembleExplainer:
    """k-pseudo-probable explanations for a fitted tree ensemble classifier
    (RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier of decision trees).

    Parameters
    ----------
    model : fitted ensemble with ``estimators_`` and ``classes_``
    feature_names : optional names of the features, used to print the explanations

    The weight of each tree, 1 - P_i, depends on the instance: it is computed by ``explain``.
    """

    METHODS = ("ga", "gd", "majority")

    def __init__(self, model, feature_names=None):
        check_model(model, "TreeEnsembleExplainer")
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.bina = Binarization()
        self.trees = [BooleanTree(est, self.bina, class_map=cm, feature_map=fm)
                      for est, cm, fm in _trees_of(model)]
        self.n = len(self.bina)

    def weights(self, x):
        """Weight 1 - P_i of each tree, P_i: proportion of the majority class in its leaf."""
        return [1 - t.confidence(x) for t in self.trees]

    def direct_reason(self, x, target):
        """Union of the decision paths of the trees voting for ``target``."""
        lits = {}
        for t in self.trees:
            if t.predict_index(x) == target:
                lits.update(dict.fromkeys(t.decision_path(x)))
        return list(lits)

    def explain(self, x, k=None, method="ga"):
        """Explanation of size <= k of the prediction of x, chosen among the literals of the
        direct reason (union of the paths of the trees voting for the prediction).

        k is optional (no limit when None). For method="majority", the ascent stops as soon as
        half of the trees (rounded up) have error 0; the reported error is then the average of the
        unweighted Submodular errors of the trees."""
        x = np.asarray(x, dtype=float).ravel()
        prediction = predict_one(self.model, x)
        target = int(np.flatnonzero(self.model.classes_ == prediction)[0])
        Phi = [t.dnf_not(target) for t in self.trees]
        n = self.n

        I = self.direct_reason(x, target)
        if k is None:
            k = len(I)

        if method == "majority":
            m = len(self.trees)
            q = math.ceil(m / 2)
            voters = [i for i, t in enumerate(self.trees) if t.predict_index(x) == target]
            if len(voters) < q:
                warnings.warn(f"only {len(voters)} trees out of {m} predict {prediction!r} on x: "
                              f"{q} trees with error 0 cannot be reached")
            S, solved = majority_ascent(I, Phi, voters, n, q, k)
            if len(voters) >= q and len(solved) < q:
                warnings.warn(f"size limit k={k} reached with {len(solved)} trees out of {m} "
                              f"with error 0 (required: {q})")
            return Explanation(method="k-pseudo-majority", prediction=prediction, literals=list(S),
                               features=self.bina.features(S),
                               conditions=self.bina.describe(S, self.feature_names),
                               error=float(mean_error(S, Phi, n)),
                               n_sufficient_trees=len(solved), n_trees=m)

        w = self.weights(x)
        if not any(w):
            warnings.warn("every tree has a pure leaf for x (P_i = 1): all the weights 1 - P_i "
                          "are 0, so every explanation has error 0")

        if method == "ga":
            S = greedy_ascent(k, I, Phi, w, n)
        elif method == "gd":
            S = greedy_descent(k, I, Phi, w, n)
        else:
            raise ValueError(f"unknown method {method!r}, expected one of {self.METHODS}")

        return Explanation(method=f"k-pseudo-{method}", prediction=prediction, literals=list(S),
                           features=self.bina.features(S),
                           conditions=self.bina.describe(S, self.feature_names),
                           error=float(epsilon_forest(S, Phi, w, n)))
