#!/usr/bin/env python3
"""
DE-based feature selection baseline (binary mapping).
Simplified DE: continuous DE followed by transfer to binary using sigmoid-like transfer.
"""
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

EPS = 1e-12

class DEFS:
    def __init__(self, X, y, feature_names, pop_size=10, max_iter=50, F=0.5, CR=0.7, alpha=0.9, seed=0):
        random.seed(seed); np.random.seed(seed)
        self.X = X; self.y = y
        self.feature_names = list(feature_names)
        self.d = X.shape[1]
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F; self.CR = CR
        self.alpha = alpha
        # initialize population in [0,1]
        self.pop = np.random.rand(pop_size, self.d)
        self.scores = np.array([self._fitness_from_vec(self.pop[i]) for i in range(pop_size)])

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-10*(x-0.5)))  # steeper to convert near 0.5

    def decode(self, vec):
        probs = self.sigmoid(vec)
        return (probs >= np.random.rand(*probs.shape)).astype(int)

    def _fitness_from_vec(self, vec):
        mask = self.decode(vec).astype(bool)
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
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                # crossover
                cross = np.random.rand(self.d) < self.CR
                trial = np.where(cross, mutant, self.pop[i])
                # ensure in [0,1]
                trial = np.clip(trial, 0.0, 1.0)
                trial_score = self._fitness_from_vec(trial)
                if trial_score > self.scores[i]:
                    self.pop[i] = trial
                    self.scores[i] = trial_score
                    if verbose:
                        print(f"Iter {it} idx {i} improved score {trial_score:.4f}")
        best_idx = int(np.argmax(self.scores))
        return self.pop[best_idx], float(self.scores[best_idx])
