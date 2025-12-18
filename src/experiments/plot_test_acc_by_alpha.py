#!/usr/bin/env python3
"""
Plot boxplot of test_acc grouped by alpha. Saves png to experiments/test_acc_by_alpha.png
Usage:
 python3 -m src.experiments.plot_test_acc_by_alpha
"""
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('experiments/experiments_results.csv')
plt.figure(figsize=(6,4))
df.boxplot(column='test_acc', by='alpha')
plt.title('Test accuracy by alpha')
plt.suptitle('')
plt.xlabel('alpha')
plt.ylabel('test_acc')
plt.tight_layout()
plt.savefig('experiments/test_acc_by_alpha.png', dpi=150)
print('Saved experiments/test_acc_by_alpha.png')
