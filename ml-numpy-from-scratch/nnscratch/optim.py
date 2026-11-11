import numpy as np


class SGD:
    def __init__(self, params, lr=0.1, momentum=0.9):
        self.params, self.lr, self.m = params, lr, momentum
        self.v = [np.zeros_like(p) for _, p, _ in params]

    def step(self):
        for v, (_, p, g) in zip(self.v, self.params):
            v *= self.m
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for _, p, _ in params]
        self.v = [np.zeros_like(p) for _, p, _ in params]

    def step(self):
        self.t += 1
        for m, v, (_, p, g) in zip(self.m, self.v, self.params):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
