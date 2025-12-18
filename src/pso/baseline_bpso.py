#!/usr/bin/env python3
"""
Binary PSO baseline for feature selection.
Encoding: positions in [0,1], update velocity like PSO, then use sigmoid transform to sample binary position.
Fitness: same as NetGPSO (alpha * accuracy - (1-alpha) * (#features / d)).
"""
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

EPS = 1e-12

class BPSO:
    def __init__(self, X, y, feature_names, pop_size=10, max_iter=50, w=0.7, c1=1.4, c2=1.4, alpha=0.9, seed=0):
        random.seed(seed); np.random.seed(seed)
        self.X = X; self.y = y
        self.feature_names = list(feature_names)
        self.d = X.shape[1]
        self.pop_size = pop_size; self.max_iter = max_iter
        self.w = w; self.c1 = c1; self.c2 = c2
        self.alpha = alpha
        # initialize
        self.pos = np.random.rand(pop_size, self.d)  # real-valued
        self.vel = np.random.uniform(-1, 1, size=(pop_size, self.d))
        self.pbest = self.pos.copy()
        self.pbest_score = np.array([self._fitness_from_pos(self.pos[i]) for i in range(pop_size)])
        self.gbest_idx = int(np.argmax(self.pbest_score))
        self.gbest = self.pbest[self.gbest_idx].copy()
        self.gbest_score = self.pbest_score[self.gbest_idx]

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def decode(self, pos_vec):
        # pos_vec is real; apply sigmoid and threshold 0.5
        probs = self.sigmoid(pos_vec)
        return (probs >= 0.5).astype(int)

    def _fitness_from_pos(self, pos_vec):
        mask = self.decode(pos_vec).astype(bool)
        if mask.sum() == 0:
            return -1.0
        Xs = self.X[:, mask]
        clf = RandomForestClassifier(n_estimators=50, random_state=0)
        try:
            acc = cross_val_score(clf, Xs, self.y, cv=3, scoring='accuracy').mean()
        except Exception:
            return -1.0
        penalty = mask.sum() / float(self.d)
        return self.alpha * acc - (1.0 - self.alpha) * penalty

    def run(self, verbose=False):
        for it in range(self.max_iter):
            for i in range(self.pop_size):
                r1 = np.random.rand(self.d)
                r2 = np.random.rand(self.d)
                cognitive = self.c1 * r1 * (self.pbest[i] - self.pos[i])
                social = self.c2 * r2 * (self.gbest - self.pos[i])
                self.vel[i] = self.w * self.vel[i] + cognitive + social
                self.pos[i] = self.pos[i] + self.vel[i]
                # no bounds for pos (sigmoid handles), but clip to avoid overflow
                self.pos[i] = np.clip(self.pos[i], -10, 10)
                score = self._fitness_from_pos(self.pos[i])
                if score > self.pbest_score[i]:
                    self.pbest[i] = self.pos[i].copy()
                    self.pbest_score[i] = score
                    if score > self.gbest_score:
                        self.gbest_score = score
                        self.gbest = self.pos[i].copy()
                        if verbose:
                            print(f"Iter {it} new gbest_score={self.gbest_score:.4f} sel={self.decode(self.gbest).sum()}")
        return self.gbest, self.gbest_score
