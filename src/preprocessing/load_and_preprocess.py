#!/usr/bin/env python3
"""
Load CSV, basic cleaning, z-score normalization, save processed CSV.
Usage:
  python3 src/preprocessing/load_and_preprocess.py data/colon.csv data/colon_processed.csv
"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main(in_path, out_path):
    df = pd.read_csv(in_path, index_col=None)
    print(f"Loaded {in_path} shape={df.shape}")
    # If label column named 'label' or 'target', try to detect; else assume last column is label
    if 'label' in df.columns:
        y = df['label'].values
        X = df.drop(columns=['label'])
    elif 'target' in df.columns:
        y = df['target'].values
        X = df.drop(columns=['target'])
    else:
        # assume last column is label
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
    # Fill NA if any
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    out_df = pd.DataFrame(Xs, columns=X.columns)
    out_df['__label__'] = y
    out_df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path} shape={out_df.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 src/preprocessing/load_and_preprocess.py input.csv output_processed.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

