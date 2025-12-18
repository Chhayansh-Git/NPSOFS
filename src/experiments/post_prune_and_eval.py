#!/usr/bin/env python3
"""
Post-selection pruning using Random Forest feature importance
Saves printed output to stdout (use redirection to save to file).
Usage:
 python3 -m src.experiments.post_prune_and_eval \
   --data data/colon_processed.csv \
   --selected experiments/selected_seed3_alpha0.80.txt \
   --topk 10 20 30 50
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(args):
    df = pd.read_csv(args.data)

    feature_names = [c for c in df.columns if c != '__label__']
    X = df[feature_names].to_numpy()
    y = df['__label__'].to_numpy()   # ← FIX: convert to NumPy array

    with open(args.selected) as f:
        selected = [line.strip() for line in f if line.strip() in feature_names]

    if len(selected) == 0:
        print("No selected features found in", args.selected)
        return

    sel_idx = [feature_names.index(f) for f in selected]
    X_sel = X[:, sel_idx]

    # Stratified train/test split (research-standard)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=0,
        n_jobs=1
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]

    print("\nPost-pruning results for:", args.selected)
    print("-" * 72)
    print("Initial selected pool size:", X_sel.shape[1])
    print()

    for k in args.topk:
        k = int(k)
        k = min(k, len(order))

        top_idx = order[:k]
        Xtr_k = X_train[:, top_idx]
        Xte_k = X_test[:, top_idx]

        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=0,
            n_jobs=1
        )
        clf.fit(Xtr_k, y_train)
        preds = clf.predict(Xte_k)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(
            f"Top-{k:>2} features → "
            f"Accuracy: {acc:.4f} | "
            f"Precision: {prec:.4f} | "
            f"Recall: {rec:.4f} | "
            f"F1: {f1:.4f}"
        )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--selected", required=True)
    p.add_argument("--topk", nargs="+", default=["10", "20", "30", "50"])
    args = p.parse_args()
    main(args)
