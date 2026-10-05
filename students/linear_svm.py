"""Your Linear SVM entry point; currently delegates to the shared reference.

Inputs are already standardized with train-only statistics. fit returns a
serializable model; predict_topk returns topk_indices/topk_scores/valid_mask
arrays with shape [N,K], K <= top_k. Margins are not probabilities. Do not read
validation labels or fit a second scaler in this module.
"""
from team import reference

IMPLEMENTATION_STATUS = "reference_delegate"
IMPLEMENTATION_NOTE = "Unmodified starting template; scikit-learn LinearSVC reference."


def fit(X_train, y_train, num_classes, config):
    # TODO: change fit logic or compare a documented, bounded configuration.
    # Mark real changes as "student_modified" and describe them above.
    return reference.fit("linear_svm", X_train, y_train, num_classes, config)


def predict_topk(model, X, top_k=5):
    # TODO: use your model's scores. Return class IDs in descending-score order.
    return reference.predict_topk(model, X, top_k)
