"""Member 3 — decision trees and a random forest written from scratch.

Guide: docs/roles/member3-random-forest.md. Check your work with
    python -m unittest tests.test_member3_random_forest -v

X is float32 [N, 1020], already standardized with TRAIN statistics by run.py.
A tree is stored as parallel arrays (one entry per node):
    feature[node]   split feature, or -1 for a leaf
    threshold[node] go left when x[feature] <= threshold
    left[node], right[node]  child node indices (-1 for a leaf)
    value[node]     class distribution of the training rows that reached the node
This version grows trees depth-first and limits them with max_depth and
min_samples_leaf; it ignores the reference's max_leaf_nodes (see the guide).
Your code must not import or call the reference (sklearn RandomForestClassifier).
"""
import math

import numpy as np

IMPLEMENTATION_STATUS = "student_modified"
# BEGIN SOLUTION member3
IMPLEMENTATION_NOTE = ("Teaching answer key (Xinjie): NumPy CART trees (Gini, midpoint thresholds, random feature "
                       "subsets) grown depth-first, bootstrap forest, averaged leaf distributions.")
# BEGIN STUDENT
# IMPLEMENTATION_NOTE = "TODO(member3): describe what you implemented yourself."
# END SOLUTION


def gini_impurity(class_counts):
    """TODO 1. Gini impurity 1 - sum_k p_k^2 of each row of class counts.

    class_counts: [..., num_classes] nonnegative counts. Rows that sum to 0 have
    impurity 0. Example: counts [3, 1] -> 1 - (0.75^2 + 0.25^2) = 0.375.
    """
    # BEGIN SOLUTION member3
    counts = np.asarray(class_counts, dtype=np.float64)
    totals = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return np.where(totals[..., 0] > 0, 1.0 - np.sum(p * p, axis=-1), 0.0)
    # END SOLUTION


def best_split(X, y, num_classes, feature_ids, min_samples_leaf=1):
    """TODO 2. Best (feature, threshold, impurity_decrease) among feature_ids.

    For each candidate feature: sort the rows by that feature, try every cut
    between two DIFFERENT consecutive values (threshold = their midpoint), keep
    only cuts with at least min_samples_leaf rows on each side, and score
    impurity_decrease = gini(parent) - (n_left * gini(left) + n_right * gini(right)) / n.
    Return the cut with the largest decrease; on exact ties keep the earlier
    feature in feature_ids and then the smaller threshold. Return (None, None, 0.0)
    when no valid cut exists.
    Hint: with one-hot labels, np.cumsum over the sorted rows gives the class
    counts of every possible left side at once.
    """
    # BEGIN SOLUTION member3
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    onehot = np.zeros((n, num_classes))
    onehot[np.arange(n), y] = 1.0
    parent = onehot.sum(axis=0)
    parent_impurity = float(gini_impurity(parent))
    best = (None, None, 0.0)
    for feature in feature_ids:
        values = X[:, feature]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        left_counts = np.cumsum(onehot[order], axis=0)[:-1]   # left side = first i+1 rows
        n_left = np.arange(1, n)
        allowed = ((sorted_values[1:] > sorted_values[:-1]) & (n_left >= min_samples_leaf)
                   & (n - n_left >= min_samples_leaf))
        if not allowed.any():
            continue
        right_counts = parent - left_counts
        weighted = (n_left * gini_impurity(left_counts) + (n - n_left) * gini_impurity(right_counts)) / n
        decrease = np.where(allowed, parent_impurity - weighted, -np.inf)
        i = int(np.argmax(decrease))
        if decrease[i] > best[2]:
            threshold = (float(sorted_values[i]) + float(sorted_values[i + 1])) / 2.0
            best = (int(feature), threshold, float(decrease[i]))
    return best
    # END SOLUTION


def build_tree(X, y, num_classes, max_depth, min_samples_leaf, max_features, rng):
    """TODO 3. Grow one tree depth-first and return it as a dict of NumPy arrays.

    At every node: store value = class counts / rows reached; stop (leaf) when
    depth == max_depth, the node is pure, it has fewer than 2 * min_samples_leaf
    rows, or best_split finds no cut with a positive decrease. Otherwise pick
    max_features random features with rng.choice(D, size=max_features,
    replace=False), split rows with x[feature] <= threshold, and recurse.
    Use max_depth=None for "no depth limit".
    """
    # BEGIN SOLUTION member3
    y = np.asarray(y, dtype=np.int64)
    nodes = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}

    def grow(rows, depth):
        node = len(nodes["feature"])
        counts = np.bincount(y[rows], minlength=num_classes).astype(np.float64)
        for key, value in (("feature", -1), ("threshold", 0.0), ("left", -1), ("right", -1),
                           ("value", counts / counts.sum())):
            nodes[key].append(value)
        if ((max_depth is not None and depth >= max_depth) or counts.max() == len(rows)
                or len(rows) < 2 * min_samples_leaf):
            return node
        features = rng.choice(X.shape[1], size=max_features, replace=False)
        feature, threshold, decrease = best_split(X[rows], y[rows], num_classes, features, min_samples_leaf)
        if feature is None or decrease <= 0:
            return node
        go_left = X[rows, feature] <= threshold
        nodes["feature"][node], nodes["threshold"][node] = feature, threshold
        nodes["left"][node] = grow(rows[go_left], depth + 1)
        nodes["right"][node] = grow(rows[~go_left], depth + 1)
        return node

    grow(np.arange(len(y)), 0)
    return {"feature": np.array(nodes["feature"], dtype=np.int64),
            "threshold": np.array(nodes["threshold"], dtype=np.float64),
            "left": np.array(nodes["left"], dtype=np.int64), "right": np.array(nodes["right"], dtype=np.int64),
            "value": np.stack(nodes["value"])}
    # END SOLUTION


def tree_predict_proba(tree, X):
    """TODO 4. Class distribution of the leaf each row of X reaches: [N, num_classes].

    Hint: keep an array `node` (all zeros = the root) and move every row that is
    not yet at a leaf one level down per loop iteration.
    """
    # BEGIN SOLUTION member3
    node = np.zeros(len(X), dtype=np.int64)
    while True:
        active = tree["feature"][node] >= 0
        if not active.any():
            return tree["value"][node]
        rows = np.flatnonzero(active)
        current = node[rows]
        go_left = X[rows, tree["feature"][current]] <= tree["threshold"][current]
        node[rows] = np.where(go_left, tree["left"][current], tree["right"][current])
    # END SOLUTION


def resolve_max_features(setting, num_features):
    if setting == "sqrt":
        return max(1, int(math.sqrt(num_features)))
    if isinstance(setting, int) and 1 <= setting <= num_features:
        return setting
    raise ValueError("max_features must be 'sqrt' or an integer between 1 and the number of features")


def fit(X_train, y_train, num_classes, config):
    """TODO 5. Random forest: n_estimators trees, each on a bootstrap sample.

    Create rng = np.random.default_rng(config["random_state"]). For every tree draw
    a bootstrap sample of N row indices WITH replacement (rng.integers(0, N, N)),
    then call build_tree with the same rng. Return a dict with the list of trees
    and num_classes (it must be saveable with joblib).
    """
    trees_wanted = int(config.get("n_estimators", 64))
    max_depth = config.get("max_depth", 12)
    min_samples_leaf = int(config.get("min_samples_leaf", 2))
    max_features = resolve_max_features(config.get("max_features", "sqrt"), np.asarray(X_train).shape[1])
    # BEGIN SOLUTION member3
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    rng = np.random.default_rng(config.get("random_state", 20261002))
    trees = []
    for _ in range(trees_wanted):
        sample = rng.integers(0, len(y), len(y))
        trees.append(build_tree(X[sample], y[sample], num_classes, max_depth, min_samples_leaf, max_features, rng))
    return {"trees": trees, "num_classes": int(num_classes)}
    # END SOLUTION


def forest_predict_proba(model, X):
    """Average of the trees' leaf distributions."""
    return np.mean([tree_predict_proba(tree, X) for tree in model["trees"]], axis=0)


def predict_topk(model, X, top_k=5):
    """Rank classes by forest probability; only classes with probability > 0 are returned."""
    probabilities = forest_predict_proba(model, np.asarray(X, dtype=np.float32))
    count = min(top_k, probabilities.shape[1])
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :count]
    picked = np.take_along_axis(probabilities, order, axis=1)
    valid = picked > 0
    ids = np.full((len(probabilities), top_k), -1, dtype=np.int64)
    scores = np.zeros((len(probabilities), top_k), dtype=np.float64)
    valid_mask = np.zeros((len(probabilities), top_k), dtype=bool)
    ids[:, :count] = np.where(valid, order, -1)
    scores[:, :count] = np.where(valid, picked, 0.0)
    valid_mask[:, :count] = valid
    return {"topk_indices": ids, "topk_scores": scores, "valid_mask": valid_mask, "score_kind": "forest_probability"}
