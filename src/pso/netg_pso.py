#!/usr/bin/env python3
"""
NetG-PSO implementation (improved initialization + pj computation + velocity regulation)
Usage:
  import NetGPSO from this file and run via run_experiment.py
Notes:
 - Expects net_meta to be a dict with keys: 'graph' (networkx Graph), 'degrees' (dict feature->weighted_degree),
   'partition' (dict feature->group_id)
 - feature_names (list) must match the column order used to create X (excluding label column).
"""
import numpy as np
import networkx as nx
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

EPS = 1e-12

class NetGPSO:
    def __init__(self, X, y, feature_names, net_meta,
                 pop_size=10, max_iter=50, w=0.7, c1=1.4, c2=1.4,
                 alpha=0.9, perc_scale=0.25, seed=0):
        """
        X: numpy array (n_samples, d)
        y: labels
        feature_names: list length d where each name matches graph node names
        net_meta: dict with keys 'graph', 'degrees', 'partition'
        alpha: weight for accuracy in combined fitness (0..1)
        perc_scale: 'Percen' used in velocity regulation (paper uses scaling factor)
        """
        random.seed(seed)
        np.random.seed(seed)
        self.X = X
        self.y = y
        self.feature_names = list(feature_names)
        self.d = X.shape[1]
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w; self.c1 = c1; self.c2 = c2
        self.alpha = alpha
        self.perc_scale = perc_scale
        # network meta
        self.graph = net_meta.get('graph', None)
        self.deg = net_meta.get('degrees', {})  # map name->weighted degree
        self.partition = net_meta.get('partition', {})  # map name->group id
        # derived
        self.name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        self.idx_to_name = {i: n for n, i in self.name_to_idx.items()}
        # ensure deg and partition have default for missing names
        for n in self.feature_names:
            self.deg.setdefault(n, 0.0)
            self.partition.setdefault(n, -1)
        # compute helper maps
        self.dmaxw = max(self.deg.values()) if len(self.deg) > 0 else 1.0
        self.groups = self._group_map()
        # initialize population using Algorithm 1 style
        self._init_population()
        # store best
        self.gbest_score = max(self.pbest_score)
        self.gbest = self.pbest[self.pbest_score.argmax()].copy()

    def _group_map(self):
        groups = {}
        for name, g in self.partition.items():
            groups.setdefault(g, []).append(name)
        return groups

    def _features_by_category(self):
        # categorize features: Xindp (degree==0), Xnet (degree>0 but group size==1), Xnetg (group size>1)
        Xindp = []
        Xnet = []
        Xnetg = []
        for n in self.feature_names:
            deg = self.deg.get(n, 0.0)
            grp = self.partition.get(n, -1)
            grp_size = len(self.groups.get(grp, []))
            if deg == 0 or grp == -1 or grp_size == 1:
                if deg == 0:
                    Xindp.append(n)
                else:
                    Xnet.append(n)
            else:
                Xnetg.append(n)
        return Xindp, Xnet, Xnetg

    def _init_population(self):
        # Implement Algorithm 1 logic (categorical selection + S and S' + pj)
        self.pos = np.random.rand(self.pop_size, self.d) * 0.5  # positions initially (0,0.5)
        self.vel = np.random.uniform(-1.0, 1.0, size=(self.pop_size, self.d))
        self.pbest = self.pos.copy()
        # prepare categories
        Xindp, Xnet, Xnetg = self._features_by_category()
        all_groups = list(self.groups.keys())
        # for each particle, create S (selected set) following algorithm steps
        for i in range(self.pop_size):
            S = set()
            # 4) randomly select 10% of Xindp
            n_indp = max(1, int(round(0.10 * max(1, len(Xindp))))) if len(Xindp) > 0 else 0
            if n_indp > 0:
                S.update(random.sample(Xindp, min(n_indp, len(Xindp))))
            # 5) randomly select 5% of Xnet
            n_net = max(1, int(round(0.05 * max(1, len(Xnet))))) if len(Xnet) > 0 else 0
            if n_net > 0:
                S.update(random.sample(Xnet, min(n_net, len(Xnet))))
            # 6) for Xnetg: pick groups
            if len(all_groups) < 10:
                # pick one random feature per group
                for g in all_groups:
                    members = self.groups.get(g, [])
                    if len(members) > 0:
                        S.add(random.choice(members))
            else:
                # randomly select 50% groups and one feature per selected group
                sel_count = max(1, int(round(0.50 * len(all_groups))))
                chosen = random.sample(all_groups, sel_count)
                for g in chosen:
                    members = self.groups.get(g, [])
                    if len(members) > 0:
                        S.add(random.choice(members))
            # 7) select another 50% groups to build S'
            sel_count2 = max(1, int(round(0.50 * len(all_groups)))) if len(all_groups) > 0 else 0
            S_prime = set()
            if sel_count2 > 0:
                chosen2 = random.sample(all_groups, sel_count2)
                for g in chosen2:
                    members = self.groups.get(g, [])
                    if len(members) > 0:
                        S_prime.add(random.choice(members))
            # 9) for each f_j in S' compute p_j and add with probability p_j
            for fname in S_prime:
                pj = self._compute_pj_for_init(fname, selected_S=S)
                if random.random() < pj:
                    S.add(fname)
            # 10) set position elements corresponding to S to rand(0.5,1.0)
            for fname in S:
                idx = self.name_to_idx.get(fname, None)
                if idx is not None:
                    self.pos[i, idx] = np.random.uniform(0.5001, 1.0)
        # initialize pbest scores
        self.pbest = self.pos.copy()
        self.pbest_score = np.array([self._fitness_from_pos(p) for p in self.pbest])
        # ensure gbest exists
        return

    def _compute_pj_for_init(self, fname, selected_S):
        """
        Compute p_j according to paper Eq. (5):
        p_j = alpha * exp(-3 / dw_j) + beta * exp(- wjN / dmaxw) + gamma * exp(- n_jC)
        where:
         - dw_j: weighted degree of j
         - wjN: influence from selected neighbors: sum_{k in SN(j)} w_jk * dw_k
         - n_jC: number of selected features from j's group (same-group count)
        We'll normalize wjN by dmaxw to keep scale similar.
        We set weights alpha,beta,gamma = 0.5, 0.3, 0.2 by default (sum=1).
        """
        alpha = 0.5; beta = 0.3; gamma = 0.2
        dw_j = float(self.deg.get(fname, 0.0)) + EPS
        # selected neighbors SN(j)
        wjN = 0.0
        if self.graph is not None and self.graph.has_node(fname):
            for nbr, data in self.graph[fname].items():
                if nbr in selected_S:
                    wjk = float(data.get('weight', 1.0))
                    dw_k = float(self.deg.get(nbr, 0.0))
                    wjN += wjk * dw_k
        # normalize wjN by dmaxw
        wjN_norm = wjN / (self.dmaxw + EPS)
        # n_jC: same-group selected features count
        grp = self.partition.get(fname, -1)
        same_group_selected = 0
        if grp != -1:
            members = self.groups.get(grp, [])
            for m in members:
                if m in selected_S:
                    same_group_selected += 1
        n_jC = same_group_selected
        term1 = alpha * np.exp(-3.0 / (dw_j + EPS))
        term2 = beta * np.exp(-wjN_norm)
        term3 = gamma * np.exp(- float(n_jC))
        pj = float(term1 + term2 + term3)
        # clip
        pj = float(np.clip(pj, 0.01, 0.99))
        return pj

    def decode(self, pos_vec):
        return (pos_vec >= 0.5).astype(int)

    def _fitness_from_pos(self, pos_vec):
        mask = self.decode(pos_vec).astype(bool)
        if mask.sum() == 0:
            return -1.0
        Xs = self.X[:, mask]
        clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=1)
        try:
            acc = cross_val_score(clf, Xs, self.y, cv=3, scoring='accuracy', n_jobs=1).mean()
        except Exception:
            return -1.0
        penalty = mask.sum() / float(self.d)
        return self.alpha * acc - (1.0 - self.alpha) * penalty

    def _compute_global_pj(self, current_selected_sets=None):
        """
        Compute pj vector (one pj per feature) for use during velocity regulation.
        We approximate neighbor influence using currently selected features aggregated across population (if provided),
        otherwise use degree-only estimate.
        """
        pj_arr = np.zeros(self.d, dtype=float)
        # estimate currently selected neighbors counts; if current_selected_sets provided (list of sets) aggregate them
        agg_selected = set()
        if current_selected_sets:
            for s in current_selected_sets:
                agg_selected.update(s)
        for i, fname in enumerate(self.feature_names):
            # use deg, neighbor influence from agg_selected
            dw_j = float(self.deg.get(fname, 0.0)) + EPS
            wjN = 0.0
            if self.graph is not None and self.graph.has_node(fname):
                for nbr, data in self.graph[fname].items():
                    if nbr in agg_selected:
                        wjk = float(data.get('weight', 1.0))
                        dw_k = float(self.deg.get(nbr, 0.0))
                        wjN += wjk * dw_k
            wjN_norm = wjN / (self.dmaxw + EPS)
            grp = self.partition.get(fname, -1)
            same_group_selected = 0
            if grp != -1:
                members = self.groups.get(grp, [])
                for m in members:
                    if m in agg_selected:
                        same_group_selected += 1
            n_jC = same_group_selected
            # same weights as init but you can tune
            alpha = 0.5; beta = 0.3; gamma = 0.2
            term1 = alpha * np.exp(-3.0 / (dw_j + EPS))
            term2 = beta * np.exp(-wjN_norm)
            term3 = gamma * np.exp(- float(n_jC))
            pj = float(term1 + term2 + term3)
            pj_arr[i] = np.clip(pj, 0.01, 0.99)
        return pj_arr

    def velocity_regulation(self, v, p, pj):
        """
        Implement Eq. (8) & (9) behavior:
         - f1(pj) = pj * Percen
         - f2(pj) = (pj - 1) * Percen
        Conditions:
         - if v>0 and p<0.5 and rand < pj -> v *= exp(f1)
         - if v<0 and p<0.5 and rand < pj -> v *= exp(-f1)
         - if v>0 and p>=0.5 and rand >= pj -> v *= exp(f2)
         - if v<0 and p>=0.5 and rand >= pj -> v *= exp(-f2)
        """
        f1 = pj * self.perc_scale
        f2 = (pj - 1.0) * self.perc_scale
        v_new = v.copy()
        rand_vec = np.random.rand(self.d)
        # conditions vectorized
        mask_a = (v > 0) & (p < 0.5) & (rand_vec < pj)
        mask_b = (v < 0) & (p < 0.5) & (rand_vec < pj)
        mask_c = (v > 0) & (p >= 0.5) & (rand_vec >= pj)
        mask_d = (v < 0) & (p >= 0.5) & (rand_vec >= pj)
        v_new[mask_a] = v_new[mask_a] * np.exp(f1[mask_a])
        v_new[mask_b] = v_new[mask_b] * np.exp(-f1[mask_b])
        v_new[mask_c] = v_new[mask_c] * np.exp(f2[mask_c])
        v_new[mask_d] = v_new[mask_d] * np.exp(-f2[mask_d])
        return v_new

    def run(self, verbose=True):
        # main PSO loop
        # to compute pj during run, we aggregate pbest masks as approximation
        for it in range(self.max_iter):
            # aggregate selected sets from pbest
            current_selected_sets = [set([self.idx_to_name[idx] for idx,v in enumerate(self.decode(p)) if v==1]) for p in self.pbest]
            pj_vec = self._compute_global_pj(current_selected_sets=current_selected_sets)
            for i in range(self.pop_size):
                r1 = np.random.rand(self.d)
                r2 = np.random.rand(self.d)
                cognitive = self.c1 * r1 * (self.pbest[i] - self.pos[i])
                social = self.c2 * r2 * (self.gbest - self.pos[i])
                self.vel[i] = self.w * self.vel[i] + cognitive + social
                # regulate velocity using pj_vec
                self.vel[i] = self.velocity_regulation(self.vel[i], self.pos[i], pj_vec)
                # update position
                self.pos[i] = self.pos[i] + self.vel[i]
                self.pos[i] = np.clip(self.pos[i], 0.0, 1.0)
                # evaluate
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
