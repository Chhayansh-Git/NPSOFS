#!/usr/bin/env python3
"""
Run BPSO baseline for multiple seeds and append results to experiments/experiments_results_full.csv
"""
import argparse, time, os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.pso.baseline_bpso import BPSO


def main(args):
    os.makedirs('experiments', exist_ok=True)
    df = pd.read_csv(args.data)
    feature_names = [c for c in df.columns if c!='__label__']
    X_all = df[feature_names].to_numpy()
    y_all = df['__label__'].to_numpy()
    results=[]
    for run_i in range(args.runs):
        seed = args.seed_base + run_i
        t0 = time.time()
        Xtr,Xte,ytr,yte = train_test_split(X_all, y_all, test_size=args.test_size, random_state=seed, stratify=y_all)
        bpso = BPSO(Xtr, ytr, feature_names, pop_size=args.pop_size, max_iter=args.max_iter, alpha=args.alpha, seed=seed)
        best_vec, best_score = bpso.run(verbose=False)
        mask = bpso.decode(best_vec).astype(int)
        # evaluate on holdout
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if mask.sum()==0:
            test_acc=test_prec=test_rec=test_f1=0.0
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            clf.fit(Xtr[:, mask.astype(bool)], ytr)
            preds = clf.predict(Xte[:, mask.astype(bool)])
            test_acc = float(accuracy_score(yte, preds))
            test_prec = float(precision_score(yte, preds, zero_division=0))
            test_rec = float(recall_score(yte, preds, zero_division=0))
            test_f1 = float(f1_score(yte, preds, zero_division=0))
        runtime = time.time()-t0
        row = {
            'method':'BPSO',
            'seed':seed,
            'alpha':args.alpha,
            'pop_size':args.pop_size,
            'max_iter':args.max_iter,
            'best_score_cv':best_score,
            'selected_count':int(mask.sum()),
            'runtime_sec':runtime,
            'test_acc':test_acc,
            'test_precision':test_prec,
            'test_recall':test_rec,
            'test_f1':test_f1,
            'selected_file':f"experiments/selected_bpso_seed{seed}_a{args.alpha:.2f}.txt"
        }
        with open(row['selected_file'],'w') as fh:
            for i,m in enumerate(mask):
                if m==1:
                    fh.write(feature_names[i]+"\n")
        results.append(row)
        # append to CSV
        out = 'experiments/experiments_results_full.csv'
        df_row = pd.DataFrame([row])
        if not os.path.exists(out):
            df_row.to_csv(out,index=False)
        else:
            df_row.to_csv(out, mode='a', header=False, index=False)
        print(f"Done BPSO seed={seed} test_acc={test_acc:.4f} sel={mask.sum()} time={runtime:.1f}s")
    print("BPSO runs complete.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--seed_base", type=int, default=1)
    p.add_argument("--test_size", type=float, default=0.3)
    p.add_argument("--pop_size", type=int, default=12)
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.8)
    args=p.parse_args()
    main(args)
