#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
df_net = pd.read_csv("experiments/experiments_results_dcor_delta03.csv")
df_full = pd.read_csv("experiments/experiments_results_full.csv")
df = pd.concat([df_net, df_full], ignore_index=True)
plt.figure(figsize=(7,4))
df.boxplot(column="test_acc", by="method", grid=False)
plt.title("Test Accuracy Comparison by Method (dCor, \\delta=0.3)")
plt.suptitle("")
plt.ylabel("Test Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("paper/figures/fig_accuracy_by_method_dcor_delta03.png", dpi=150)
print("Saved paper/figures/fig_accuracy_by_method_dcor_delta03.png")
