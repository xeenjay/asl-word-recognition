"""Your RF entry point; currently delegates to the shared scikit-learn reference.

Inputs are already standardized with train-only statistics. fit returns a
serializable model; predict_topk returns topk_indices/topk_scores/valid_mask
arrays with shape [N,K], K <= top_k. Use -1/False for unsupported ranks. Keep
tree/leaf limits explicit: a large vocabulary can make forests memory intensive.
"""
from team import reference

IMPLEMENTATION_STATUS = "reference_delegate"
IMPLEMENTATION_NOTE = "Unmodified starting template; bounded scikit-learn forest reference."


def fit(X_train, y_train, num_classes, config):
    # TODO: compare bounded settings or replace the fitting logic.
    # Mark actual code changes as "student_modified" and describe them above.
    return reference.fit("random_forest", X_train, y_train, num_classes, config)


def predict_topk(model, X, top_k=5):
    # TODO: keep class-column alignment and return only supported predictions.
    return reference.predict_topk(model, X, top_k)
