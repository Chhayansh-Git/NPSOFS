import pandas as pd

# Paths
train_path = "data/alizadeh/alizadeh_trainData.txt"
test_path  = "data/alizadeh/alizadeh_testData.txt"
out_path   = "data/alizadeh/alizadeh_processed.csv"

# Load (comma-separated, no header)
train_df = pd.read_csv(train_path, header=None)
test_df  = pd.read_csv(test_path, header=None)

# Drop first column (sample ID)
train_df = train_df.iloc[:, 1:]
test_df  = test_df.iloc[:, 1:]

# Rename last column as label
n_cols = train_df.shape[1]
feature_cols = [f"g{i}" for i in range(n_cols - 1)]
cols = feature_cols + ["__label__"]

train_df.columns = cols
test_df.columns  = cols

# Merge
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Save
full_df.to_csv(out_path, index=False)

print("Saved:", out_path)
print("Shape:", full_df.shape)
print("Label distribution:")
print(full_df["__label__"].value_counts())
