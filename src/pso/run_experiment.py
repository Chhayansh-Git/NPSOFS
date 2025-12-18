import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#!/usr/bin/env python3
"""
Run a short experiment:
Usage:
  python3 src/pso/run_experiment.py data/colon_processed.csv results/colon_network.pickle
This updated runner provides feature_names -> NetGPSO mapping.
"""
import sys
import pandas as pd
import numpy as np
import pickle
from pso.netg_pso import NetGPSO

def main(data_csv, net_pickle):
    df = pd.read_csv(data_csv)
    # assume last column is __label__
    if '__label__' not in df.columns:
        raise RuntimeError("Processed CSV must have '__label__' as last/label column")
    feature_names = [c for c in df.columns if c != '__label__']
    X = df[feature_names].values
    y = df['__label__'].values
    with open(net_pickle, 'rb') as f:
        meta = pickle.load(f)
    # meta expected to contain 'graph','degrees','partition'
    pso = NetGPSO(X, y, feature_names, meta, pop_size=12, max_iter=20, alpha=0.92, seed=1)
    best_pos, best_score = pso.run(verbose=True)
    mask = (best_pos >= 0.5).astype(int)
    selected = [feature_names[i] for i, m in enumerate(mask) if m==1]
    print("BEST score:", best_score)
    print("selected features count:", mask.sum())
    print("selected features (first 50):", selected[:50])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 src/pso/run_experiment.py data/colon_processed.csv results/colon_network.pickle")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
