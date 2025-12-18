#!/usr/bin/env python3
"""
Experiment driver: run multiple NetG-PSO runs with different seeds/alphas and evaluate on holdout.
Usage example:
  python3 src/experiments/experiment_driver.py \
    --data data/colon_processed.csv \
    --net results/colon_network.pickle \
    --out experiments/experiments_results.csv \
    --runs 5 \
    --alphas 0.92 0.8 0.7 \
    --pop_size 12 \
    --max_iter 20
"""
import argparse
import time
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.pso.netg_pso import NetGPSO

def evaluate_on_holdout(X_train, y_train, X_test, y_test, selected_mask):
    if selected_mask.sum() == 0:
        return {'test_acc': 0.0, 'test_precision': 0.0, 'test_recall': 0.0, 'test_f1': 0.0}
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
    clf.fit(X_train[:, selected_mask], y_train)
    preds = clf.predict(X_test[:, selected_mask])
    return {
        'test_acc': float(accuracy_score(y_test, preds)),
        'test_precision': float(precision_score(y_test, preds, average='binary', zero_division=0)),
        'test_recall': float(recall_score(y_test, preds, average='binary', zero_division=0)),
        'test_f1': float(f1_score(y_test, preds, average='binary', zero_division=0))
    }

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_csv(args.data)
    if '__label__' not in df.columns:
        raise RuntimeError("Processed CSV must have '__label__' column")
    feature_names = [c for c in df.columns if c != '__label__']
    X_all = df[feature_names].to_numpy()
    y_all = df['__label__'].to_numpy(dtype=int)

    with open(args.net, 'rb') as f:
        net_meta = pickle.load(f)
    results = []
    header_written = False
    for alpha in args.alphas:
        for run_i in range(args.runs):
            seed = args.seed_base + run_i
            t0 = time.time()
            # train/test split (stratified)
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=args.test_size,
                random_state=seed,
                stratify=y_all.astype(int)
            )
            pso = NetGPSO(X_train, y_train, feature_names, net_meta,
                          pop_size=args.pop_size, max_iter=args.max_iter, alpha=alpha, seed=seed)
            best_pos, best_score = pso.run(verbose=False)
            mask = (best_pos >= 0.5).astype(bool)
            selected = [feature_names[i] for i, m in enumerate(mask) if m]
            # evaluate holdout
            metrics = evaluate_on_holdout(X_train, y_train, X_test, y_test, mask)
            t1 = time.time()
            runtime = t1 - t0
            # save selected features to file
            sel_fname = f"experiments/selected_seed{seed}_alpha{alpha:.2f}.txt"
            os.makedirs(os.path.dirname(sel_fname), exist_ok=True)
            with open(sel_fname, "w") as fh:
                fh.write("\n".join(selected))
            row = {
                'method': 'NetG-PSO',
                'seed': seed,
                'alpha': alpha,
                'pop_size': args.pop_size,
                'max_iter': args.max_iter,
                'best_score_cv': float(best_score),
                'selected_count': int(mask.sum()),
                'runtime_sec': float(runtime),
                'test_acc': metrics['test_acc'],
                'test_precision': metrics['test_precision'],
                'test_recall': metrics['test_recall'],
                'test_f1': metrics['test_f1'],
                'selected_file': sel_fname
            }
            results.append(row)
            # append to CSV incrementally
            df_row = pd.DataFrame([row])
            if not os.path.exists(args.out):
                df_row.to_csv(args.out, index=False)
            else:
                df_row.to_csv(args.out, mode='a', header=False, index=False)
            print(f"Done seed={seed} alpha={alpha:.2f} best_cv={best_score:.4f} sel={mask.sum()} test_acc={metrics['test_acc']:.4f} time={runtime:.1f}s")
    print("All runs complete. Results written to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--alphas", type=float, nargs='+', default=[0.92])
    p.add_argument("--pop_size", type=int, default=12)
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--seed_base", type=int, default=1)
    args = p.parse_args()
    main(args)
