"""Your kNN entry point. The starting version explicitly reuses the reference.

X is float32 [N,1020], already standardized with TRAIN statistics by run.py.
fit never receives validation features or labels. Save any learned state in the
returned model. predict_topk returns a dict with integer topk_indices [N,K],
finite topk_scores and boolean valid_mask of the same shape, K <= top_k. Use
ID -1 with valid_mask=False for unfilled ranks. Scores need not be probabilities.
"""
from team import reference

IMPLEMENTATION_STATUS = "reference_delegate"
IMPLEMENTATION_NOTE = "Unmodified starting template; standard cosine kNN reference."


def fit(X_train, y_train, num_classes, config):
    # TODO: understand this implementation, or replace with your own fit logic.
    # If you change the algorithm, set status to "student_modified" and explain
    # your actual changes in IMPLEMENTATION_NOTE. Reusing a library is allowed.
    return reference.fit("knn", X_train, y_train, num_classes, config)


def predict_topk(model, X, top_k=5):
    # TODO: implement neighbor search/voting here, retaining deterministic ties.
    return reference.predict_topk(model, X, top_k)
