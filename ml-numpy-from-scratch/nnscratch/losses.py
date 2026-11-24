import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """Mean cross-entropy of integer labels y given logits (N, C). Returns (loss, dlogits).

    Uses the log-sum-exp trick, so it is finite for logits of any magnitude (e.g. 1000).
    """
    logits = np.asarray(logits)
    y = np.asarray(y)
    if logits.ndim != 2:
        raise ValueError("logits must be a 2-D array of shape (N, C)")
    if y.ndim != 1 or y.shape[0] != logits.shape[0]:
        raise ValueError("y must be a 1-D array with one label per row")
    if logits.shape[1] < 1:
        raise ValueError("logits must contain at least one class")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integers")
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError("label index out of range")
    z = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logsumexp
    n = logits.shape[0]
    loss = -logp[np.arange(n), y].mean()
    d = np.exp(logp)
    d[np.arange(n), y] -= 1.0
    return float(loss), d / n
