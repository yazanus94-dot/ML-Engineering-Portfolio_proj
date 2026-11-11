import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray):
    """Mean cross-entropy of integer labels y given logits (N, C). Returns (loss, dlogits).

    Uses the log-sum-exp trick, so it is finite for logits of any magnitude (e.g. 1000).
    """
    z = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logsumexp
    n = logits.shape[0]
    loss = -logp[np.arange(n), y].mean()
    d = np.exp(logp)
    d[np.arange(n), y] -= 1.0
    return float(loss), d / n
