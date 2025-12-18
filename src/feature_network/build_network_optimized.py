#!/usr/bin/env python3
"""
Optimized network builder using distance correlation.

- Optional Pearson prefilter to reduce candidate pairs.
- Parallel dCor computation with joblib.
- Produces same pickle structure used by NetG-PSO code.

Usage:
  python3 src/feature_network/build_network_optimized.py \
    data/colon_processed.csv \
    results/colon_network_dcor_opt.pickle \
    --delta 0.3 --pearson_thresh 0.12 --topk 80 --n_jobs 8 --verbose
"""

import argparse
import os
import math
import pickle
from typing import Optional, Set, Tuple, List, Dict, Sequence, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import dcor
import networkx as nx
from community import community_louvain
from tqdm.auto import tqdm


# --------------------------------------------------
# Safe distance-correlation computation for one pair
# --------------------------------------------------
def compute_pair_dcor(i: int, j: int, data: np.ndarray) -> Tuple[int, int, float]:
    xi = data[:, i]
    xj = data[:, j]
    try:
        v = float(dcor.distance_correlation(xi, xj))
    except Exception:
        v = 0.0
    return i, j, v


# --------------------------------------------------
# Optimized network builder
# --------------------------------------------------
def build_network_optimized(
    df: pd.DataFrame,
    delta: float = 0.3,
    pearson_thresh: Optional[float] = None,
    topk: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[
    nx.Graph,
    Dict[str, float],
    Dict[str, int],
    Optional[np.ndarray],
    List[str]
]:
    if "__label__" not in df.columns:
        raise ValueError("Input dataframe must contain '__label__' column.")

    X = df.drop(columns=["__label__"]).values
    cols: List[str] = df.drop(columns=["__label__"]).columns.tolist()
    n_features = len(cols)

    if verbose:
        print(
            f"Features: {n_features} | delta={delta} | "
            f"pearson_thresh={pearson_thresh} | topk={topk} | n_jobs={n_jobs}"
        )

    candidate_pairs: Set[Tuple[int, int]] = set()
    pearson_mat: Optional[np.ndarray] = None

    # ---------- Pearson prefilter ----------
    if pearson_thresh is not None or topk is not None:
        if verbose:
            print("Computing Pearson correlation matrix (prefilter)...")
        pearson_mat = np.corrcoef(X, rowvar=False)
        abspear = np.abs(pearson_mat)

        if pearson_thresh is not None:
            if not (0.0 <= pearson_thresh <= 1.0):
                raise ValueError("pearson_thresh must be between 0 and 1")
            idxs = np.where(np.triu(abspear, k=1) >= pearson_thresh)
            for i, j in zip(idxs[0], idxs[1]):
                candidate_pairs.add((int(i), int(j)))

        if topk is not None:
            if topk <= 0:
                raise ValueError("topk must be > 0")
            k = min(topk, n_features - 1)
            for i in range(n_features):
                row = abspear[i].copy()
                row[i] = -1.0
                neighbors = (
                    np.argpartition(-row, k)[:k]
                    if k < n_features - 1
                    else np.argsort(-row)[:k]
                )
                for j in neighbors:
                    a, b = (i, j) if i < j else (j, i)
                    candidate_pairs.add((int(a), int(b)))

    # ---------- Full pairwise fallback ----------
    if not candidate_pairs:
        if verbose:
            print("No prefilter selected: computing all O(d²) pairs")
        for i in range(n_features):
            for j in range(i + 1, n_features):
                candidate_pairs.add((i, j))

    pairs: List[Tuple[int, int]] = list(candidate_pairs)
    if verbose:
        print(f"Total candidate pairs for dCor: {len(pairs)}")

    # ---------- Parallel dCor computation ----------
    raw_results = Parallel(
        n_jobs=n_jobs,
        prefer="threads"
    )(
        delayed(compute_pair_dcor)(i, j, X)
        for (i, j) in tqdm(pairs, disable=not verbose)
    )

    # KEY FIX: Explicitly cast joblib output to the expected List type
    # This resolves the "list[Unknown | None] not assignable" Pylance error.
    results: Sequence[Tuple[int, int, float]] = cast(
        List[Tuple[int, int, float]],
        raw_results
    )

    # ---------- Graph construction ----------
    G = nx.Graph()
    G.add_nodes_from(cols)
    weight_sum: Dict[str, float] = {c: 0.0 for c in cols}

    for i, j, v in results:
        if math.isnan(v):
            continue
        if abs(v) >= delta:
            u, vname = cols[i], cols[j]
            G.add_edge(u, vname, weight=float(v))
            weight_sum[u] += float(v)
            weight_sum[vname] += float(v)

    deg: Dict[str, float] = {n: weight_sum.get(n, 0.0) for n in G.nodes()}

    if G.number_of_edges() > 0:
        part: Dict[str, int] = community_louvain.best_partition(G, weight="weight")
    else:
        part = {n: 0 for n in G.nodes()}

    return G, deg, part, pearson_mat, cols


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("out_pickle")
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--pearson_thresh", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    result = build_network_optimized(
        df,
        delta=args.delta,
        pearson_thresh=args.pearson_thresh,
        topk=args.topk,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )
    assert result is not None
    G, deg, part, pearson_mat, cols = result

    out = {
        "graph": G,
        "degrees": deg,
        "partition": part,
        "dcor_matrix": None,
        "feature_names": cols,
    }

    os.makedirs(os.path.dirname(args.out_pickle) or ".", exist_ok=True)
    with open(args.out_pickle, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved network pickle to {args.out_pickle}")

    edgelist_path = args.out_pickle.replace(".pickle", "_edgelist.csv")
    degrees_path = args.out_pickle.replace(".pickle", "_degrees.csv")

    with open(edgelist_path, "w") as f:
        f.write("u,v,weight\n")
        for u, v, data in G.edges(data=True):
            f.write(f"{u},{v},{data.get('weight', 1.0)}\n")

    pd.Series(deg).to_csv(degrees_path, header=["weighted_degree"])
    print(f"Wrote edge list to {edgelist_path} and degrees to {degrees_path}")


if __name__ == "__main__":
    main()