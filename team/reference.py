"""Adapters for the unchanged reference classifiers, with delegation tracking."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from classical_models import fit_classifier, predict_topk as _predict_topk

_calls = ContextVar("reference_calls", default=None)


@contextmanager
def trace_calls():
    calls = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


def _record(operation, method):
    calls = _calls.get()
    if calls is not None:
        calls.append({"operation": operation, "method": method})


def fit(method, X_train, y_train, num_classes, config):
    """X_train is already standardized by the runner using train statistics."""
    _record("fit", method)
    return fit_classifier(method, X_train, y_train, num_classes, **config)


def predict_topk(model, X, top_k=5):
    """Return IDs, scores and valid_mask; never pad with unvoted classes."""
    _record("predict_topk", model.method)
    return _predict_topk(model, X, top_k=top_k, device="cpu", batch_size=128,
                         reference_chunk_size=8192)
