#!/usr/bin/env python3
"""
Evaluate no-feature-selection baseline and perform Wilcoxon comparison.
Two modes:
 1) --mode eval_no_fs : prints no-FS accuracies for seeds [1,2,3,4,5]
 2) --mode wilcoxon  --netcsv experiments/experiments_results.csv
    : computes Wilcoxon between best-alpha NetG-PSO per-seed and no-FS per-seed (requires same seeds)
Usage:
 python3 -m src.experiments.eval_no_fs_and_wilcoxon --mode eval_no_fs --data data/colon_processed.csv
 python3 -m src.experiments.eval_no_fs_and_wilcoxon --mode wilcoxon --netcsv experiments/experiments_results.csv
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import wilcoxon

def eval_no_fs(data_path):
    df = pd.read_csv(data_path)
    # convert to numpy arrays to satisfy type-checkers and sklearn
    X = df[[c for c in df.columns if c != '__label__']].to_numpy()
    y = df['__label__'].to_numpy()
    seeds = [1,2,3,4,5]
    accs = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=s, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        accs.append(acc)
        print(f"seed={s} no-FS acc={acc:.4f}")
    print("no-FS acc mean/std:", np.mean(accs), np.std(accs))
    # save
    pd.DataFrame({'seed':seeds,'no_fs_acc':accs}).to_csv('experiments/no_fs_accs.csv', index=False)

def wilcoxon_test(netcsv):
    df = pd.read_csv(netcsv)
    # find best alpha (highest mean test_acc)
    best_alpha = df.groupby('alpha')['test_acc'].mean().idxmax()
    netg = df[df['alpha'] == best_alpha].sort_values('seed')
    # load nofs
    nofs = pd.read_csv('experiments/no_fs_accs.csv').sort_values('seed')
    # align by seed
    common = pd.merge(netg, nofs, on='seed')
    net_accs = common['test_acc'].values
    nofs_accs = common['no_fs_acc'].values
    print("best_alpha:", best_alpha)
    print("NetG-PSO accs:", net_accs)
    print("No-FS accs   :", nofs_accs)
    # perform Wilcoxon signed-rank test (paired)
    stat, p = wilcoxon(net_accs, nofs_accs)
    print("Wilcoxon stat, p-value:", stat, p)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=['eval_no_fs','wilcoxon'], required=True)
    p.add_argument("--data", default='data/colon_processed.csv')
    p.add_argument("--netcsv", default='experiments/experiments_results.csv')
    args = p.parse_args()
    if args.mode == 'eval_no_fs':
        eval_no_fs(args.data)
    else:
        wilcoxon_test(args.netcsv)
