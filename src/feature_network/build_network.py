#!/usr/bin/env python3
"""
Build a feature network using DISTANCE CORRELATION between features and save network + metadata.

Notes:
 - Uses the `dcor` package (pip install dcor).
 - Distance correlation is in [0,1] and captures non-linear dependence.
 - For small sample microarray data this is appropriate; computing all pairwise dCor values
   is O(d^2 * n^2) in worst-case (d = #features, n = #samples) but usually fine for gene-expression datasets.
Usage:
  python3 src/feature_network/build_network.py data/colon_processed.csv results/colon_network_dcor.pickle --delta 0.2
(Choose delta sensible for dCor -- typical starting values: 0.2, 0.3, 0.4)
Outputs:
 - Pickle file containing dict with: graph (networkx), degrees dict, communities dict
 - Also writes results/colon_edgelist_dcor.csv and results/colon_degrees_dcor.csv
"""
import sys
import argparse
import pandas as pd
import networkx as nx
import pickle
from community import community_louvain
import dcor
import numpy as np
import math

def build_network_dcor(df, delta=0.2, verbose=True):
    # df: DataFrame where last column '__label__' is label
    X = df.drop(columns=['__label__'])
    cols = X.columns.tolist()
    n_features = len(cols)
    if verbose:
        print(f"Computing distance correlation matrix for {n_features} features (may take a moment)...")
    data = X.values
    # Pre-allocate symmetric matrix
    corr = np.zeros((n_features, n_features), dtype=float)
    # compute upper triangle
    for i in range(n_features):
        xi = data[:, i]
        for j in range(i + 1, n_features):
            xj = data[:, j]
            try:
                # dcor.distance_correlation returns float in [0,1]
                v = float(dcor.distance_correlation(xi, xj))
            except Exception:
                v = 0.0
            corr[i, j] = v
            corr[j, i] = v
        corr[i, i] = 1.0
    # Build graph using threshold delta
    G = nx.Graph()
    G.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for j in range(i + 1, n_features):
            b = cols[j]
            v = corr[i, j]
            if math.isnan(v):
                continue
            if abs(v) >= delta:
                w = float(v)
                G.add_edge(a, b, weight=w)
    # weighted degree
    deg = {n: sum(d.get('weight', 1.0) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    # communities via Louvain
    if len(G) > 0 and G.number_of_edges() > 0:
        part = community_louvain.best_partition(G, weight='weight')
    else:
        part = {n: 0 for n in G.nodes()}
    return G, deg, part, corr, cols

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("out_pickle")
    parser.add_argument("--delta", type=float, default=0.2,
                        help="edge threshold on distance correlation (0..1). Try 0.2/0.3/0.4")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    print(f"Loaded processed data {args.input_csv} shape={df.shape}")
    G, deg, part, corr, cols = build_network_dcor(df, delta=args.delta, verbose=args.verbose)
    out = {
        "graph": G,
        "degrees": deg,
        "partition": part,
        "dcor_matrix": corr,
        "feature_names": cols
    }
    with open(args.out_pickle, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved network pickle to {args.out_pickle}")
    # also save edge list and degrees (add suffix _dcor)
    import csv, os
    os.makedirs("results", exist_ok=True)
    edgelist_path = os.path.join("results", "colon_edgelist_dcor.csv")
    degrees_path = os.path.join("results", "colon_degrees_dcor.csv")
    with open(edgelist_path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(['u', 'v', 'weight'])
        for u, v, data in G.edges(data=True):
            w.writerow([u, v, data.get('weight', 1.0)])
    # degrees to CSV
    pd.Series(deg).to_csv(degrees_path, header=['weighted_degree'])
    print(f"Wrote edge list to {edgelist_path} and degrees to {degrees_path}")

if __name__ == "__main__":
    main()
