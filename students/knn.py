"""Member 1 — k-nearest neighbours written from scratch with NumPy.

Guide: docs/roles/member1-knn.md. Check your work with
    python -m unittest tests.test_member1_knn -v

X is float32 [N, 1020], already standardized with TRAIN statistics by run.py.
fit never receives validation features or labels. predict_topk returns a dict
with integer topk_indices [N, K], finite topk_scores and boolean valid_mask of
the same shape, K <= top_k. Use ID -1 with valid_mask=False for unfilled ranks.

The reference (scripts/classical_models.py) is cosine kNN with uniform votes.
Write your own version first; compare with the reference only after your tests
pass. Your code must not import or call the reference.
"""
import numpy as np

# After your implementation passes the tests, keep "student_modified" and
# describe what you actually wrote (and any AI help) in IMPLEMENTATION_NOTE.
IMPLEMENTATION_STATUS = "student_modified"
# BEGIN SOLUTION member1
IMPLEMENTATION_NOTE = ("Teaching answer key (Xinjie): NumPy cosine kNN, stable train-row neighbour ties, "
                       "votes ranked by count, nearest support rank, class ID; optional similarity weighting.")
# BEGIN STUDENT
# IMPLEMENTATION_NOTE = "TODO(member1): describe what you implemented yourself."
# END SOLUTION


def l2_normalize_rows(X):
    """TODO 1. Return a float32 copy of X where every nonzero row has L2 norm 1.

    Rows that are entirely zero must stay zero (do not divide by zero).
    Hint: compute the norms in float64 with np.linalg.norm(..., axis=1), then use
    np.divide(..., out=..., where=...) so zero rows are skipped.
    """
    # BEGIN SOLUTION member1
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X.astype(np.float64), axis=1)
    result = np.zeros_like(X, dtype=np.float32)
    np.divide(X, norms[:, None], out=result, where=norms[:, None] != 0)
    return result
    # END SOLUTION


def cosine_similarity(queries, references):
    """TODO 2. Cosine similarity matrix S with S[i, j] = cos(queries[i], references[j]).

    Both inputs are already L2-normalized float32 matrices, so this is a single
    matrix product. Output shape: [num_queries, num_references], dtype float32.
    """
    # BEGIN SOLUTION member1
    return np.asarray(queries, dtype=np.float32) @ np.asarray(references, dtype=np.float32).T
    # END SOLUTION


def nearest_neighbors(similarities, k):
    """TODO 3. Indices and similarities of the k most similar references per row.

    Order each row from most to least similar. When two references have exactly
    the same similarity, the one with the smaller training-row index comes first.
    Hint: np.argsort(-similarities, axis=1, kind="stable") keeps that order.
    Returns (indices int64 [N, k], neighbour_similarities float32 [N, k]).
    """
    # BEGIN SOLUTION member1
    order = np.argsort(-similarities, axis=1, kind="stable")[:, :k].astype(np.int64)
    return order, np.take_along_axis(similarities, order, axis=1)
    # END SOLUTION


def vote(neighbor_labels, top_k=5):
    """TODO 4. Uniform vote for ONE query; neighbor_labels is ordered nearest first.

    Rank the classes that received at least one vote by:
      1. more votes first;
      2. on equal votes, the class whose nearest supporting neighbour is closer;
      3. then the smaller class ID.
    Score of a class = its votes / number of neighbours. Return at most top_k
    (class_ids, scores) as two Python lists. Never add classes with zero votes.
    """
    # BEGIN SOLUTION member1
    labels = [int(label) for label in neighbor_labels]
    first, counts = {}, {}
    for rank, label in enumerate(labels):
        first.setdefault(label, rank)
        counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts, key=lambda c: (-counts[c], first[c], c))[:top_k]
    return ranked, [counts[c] / len(labels) for c in ranked]
    # END SOLUTION


def similarity_vote(neighbor_labels, neighbor_similarities, top_k=5):
    """TODO 5 (extension). Similarity-weighted vote for ONE query.

    Each neighbour adds max(similarity, 0) to its class. Rank classes with a
    positive total by (larger total, nearer first support, smaller class ID);
    score = class total / sum of all clipped weights. If every weight is 0, fall
    back to vote(neighbor_labels, top_k).
    """
    # BEGIN SOLUTION member1
    labels = [int(label) for label in neighbor_labels]
    weights = [float(s) if s > 0 else 0.0 for s in neighbor_similarities]
    total = sum(weights)
    if total <= 0:
        return vote(labels, top_k)
    first, score = {}, {}
    for rank, (label, weight) in enumerate(zip(labels, weights)):
        first.setdefault(label, rank)
        score[label] = score.get(label, 0.0) + weight
    ranked = sorted((c for c in score if score[c] > 0), key=lambda c: (-score[c], first[c], c))[:top_k]
    return ranked, [score[c] / total for c in ranked]
    # END SOLUTION


def fit(X_train, y_train, num_classes, config):
    k = int(config.get("n_neighbors", 5))
    weighting = config.get("vote_weighting", "uniform")
    if not 1 <= k <= len(X_train):
        raise ValueError("n_neighbors must be between 1 and the number of training rows")
    if weighting not in ("uniform", "similarity"):
        raise ValueError("vote_weighting must be 'uniform' or 'similarity'")
    # kNN has no training step: store the normalized training set (the "memory").
    return {"references": l2_normalize_rows(X_train), "labels": np.asarray(y_train, dtype=np.int64).copy(),
            "num_classes": int(num_classes), "n_neighbors": k, "vote_weighting": weighting}


def predict_topk(model, X, top_k=5):
    queries = l2_normalize_rows(X)
    neighbors, similarities = nearest_neighbors(cosine_similarity(queries, model["references"]), model["n_neighbors"])
    ids = np.full((len(queries), top_k), -1, dtype=np.int64)
    scores = np.zeros((len(queries), top_k), dtype=np.float64)
    valid = np.zeros((len(queries), top_k), dtype=bool)
    for row in range(len(queries)):
        labels = model["labels"][neighbors[row]]
        if model["vote_weighting"] == "similarity":
            classes, values = similarity_vote(labels, similarities[row], top_k)
        else:
            classes, values = vote(labels, top_k)
        ids[row, :len(classes)] = classes
        scores[row, :len(classes)] = values
        valid[row, :len(classes)] = True
    kind = "similarity_weighted_neighbor_vote_fraction" if model["vote_weighting"] == "similarity" else "uniform_neighbor_vote_fraction"
    return {"topk_indices": ids, "topk_scores": scores, "valid_mask": valid, "score_kind": kind}
