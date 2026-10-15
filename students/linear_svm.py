"""Member 2 — one-vs-rest linear SVM written from scratch.

Guide: docs/roles/member2-linear-svm.md. Check your work with
    python -m unittest tests.test_member2_linear_svm -v

X is float32 [N, 1020], already standardized with TRAIN statistics by run.py.
You train one linear classifier per class ("this class" = +1, every other class
= -1) by minimizing, for each class c,

    0.5 * ||w_c||^2 + C * sum_i max(0, 1 - Y[i, c] * (x_i . w_c))^2

where x_i has an extra constant 1 appended so the last entry of w_c is the bias.
This is the same "squared hinge, L2, regularized bias" objective that the
reference sklearn LinearSVC minimizes, so a correct implementation should give
almost the same weights. Your code must not import or call the reference.
"""
import numpy as np
from scipy.optimize import minimize

IMPLEMENTATION_STATUS = "student_modified"
# BEGIN SOLUTION member2
IMPLEMENTATION_NOTE = ("Teaching answer key (Xinjie): one-vs-rest squared-hinge L2 linear SVM with a regularized "
                       "bias column, analytic gradient, optimized jointly with SciPy L-BFGS-B.")
# BEGIN STUDENT
# IMPLEMENTATION_NOTE = "TODO(member2): describe what you implemented yourself."
# END SOLUTION


def one_vs_rest_targets(y, num_classes):
    """TODO 1. Return Y float64 [N, num_classes] with Y[i, c] = +1 if y[i] == c else -1."""
    # BEGIN SOLUTION member2
    Y = -np.ones((len(y), num_classes), dtype=np.float64)
    Y[np.arange(len(y)), np.asarray(y, dtype=np.int64)] = 1.0
    return Y
    # END SOLUTION


def add_bias_column(X):
    """Append a column of ones (float64), so the bias is the last weight row."""
    X = np.asarray(X, dtype=np.float64)
    return np.hstack([X, np.ones((len(X), 1))])


def squared_hinge_objective(W, Xb, Y, C):
    """TODO 2. Objective value and gradient for ALL classes at once.

    W: [D+1, num_classes] weights (last row = bias); Xb: [N, D+1]; Y: [N, num_classes].
    margins = Y * (Xb @ W); slack = max(0, 1 - margins)
    loss    = 0.5 * sum(W ** 2) + C * sum(slack ** 2)
    Derive the gradient yourself first (see the guide), then check it with
    tests.test_member2_linear_svm.Todo2Objective. Returns (loss float, grad [D+1, num_classes]).
    """
    # BEGIN SOLUTION member2
    margins = Y * (Xb @ W)
    slack = np.maximum(0.0, 1.0 - margins)
    loss = 0.5 * float(np.sum(W * W)) + C * float(np.sum(slack * slack))
    grad = W - 2.0 * C * (Xb.T @ (Y * slack))
    return loss, grad
    # END SOLUTION


def train_weights(Xb, Y, C, max_iter=2000, tol=1e-3):
    """TODO 3. Minimize squared_hinge_objective over W, starting from zeros.

    scipy.optimize.minimize works on 1-D vectors: flatten W with ravel() and
    reshape it back inside the function you pass in. Use method="L-BFGS-B" and
    jac=True (your function returns both the value and the gradient).
    Use options={"maxiter": max_iter, "gtol": tol * 1e-3, "ftol": 1e-12}.
    Returns (W [D+1, num_classes], info dict with at least "converged" and "iterations").
    """
    # BEGIN SOLUTION member2
    shape = (Xb.shape[1], Y.shape[1])

    def objective(flat):
        loss, grad = squared_hinge_objective(flat.reshape(shape), Xb, Y, C)
        return loss, grad.ravel()

    result = minimize(objective, np.zeros(shape[0] * shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter, "gtol": tol * 1e-3, "ftol": 1e-12})
    return result.x.reshape(shape), {"converged": bool(result.success), "iterations": int(result.nit),
                                     "final_objective": float(result.fun), "message": str(result.message)}
    # END SOLUTION


def top_k_from_scores(scores, top_k=5):
    """TODO 4. Rank classes by score, highest first; ties go to the smaller class ID.

    scores: [N, num_classes]. Return (ids int64 [N, K], values float64 [N, K]) with
    K = min(top_k, num_classes). Hint: np.argsort(-scores, axis=1, kind="stable").
    """
    # BEGIN SOLUTION member2
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, axis=1, kind="stable")[:, :min(top_k, scores.shape[1])].astype(np.int64)
    return order, np.take_along_axis(scores, order, axis=1)
    # END SOLUTION


def fit(X_train, y_train, num_classes, config):
    C = float(config.get("C", 0.01))
    if C <= 0:
        raise ValueError("C must be positive")
    W, info = train_weights(add_bias_column(X_train), one_vs_rest_targets(y_train, num_classes), C,
                            max_iter=int(config.get("max_iter", 2000)), tol=float(config.get("tol", 1e-3)))
    return {"W": W, "num_classes": int(num_classes), "C": C, "info": info}


def decision_function(model, X):
    """Margin of every class: [N, num_classes]. Larger means "more like this class"."""
    return add_bias_column(X) @ model["W"]


def predict_topk(model, X, top_k=5):
    ids, values = top_k_from_scores(decision_function(model, X), top_k)
    return {"topk_indices": ids, "topk_scores": values, "valid_mask": np.ones(ids.shape, dtype=bool),
            "score_kind": "ovr_decision_margin"}
