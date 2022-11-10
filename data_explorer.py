from turtle import color
import pandas as pd
import os
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt
import numpy as np

# Declare expression data and metadata path 
expression_path = os.path.join("data","Shokhirev_2020","raw_filtered.txt")
meta_path = os.path.join("data","Shokhirev_2020","meta_filtered.txt")

# Read expression file and make minnor modifications
expression = pd.read_csv(expression_path, sep="\t", header=0)
expression = expression.rename(columns={'Unnamed: 0': 'SRR.ID'})
expression = expression.set_index('SRR.ID')
expression = expression.T

# Read metadata file and make minnor modifications
meta = pd.read_csv(meta_path, sep="\t", header=0)
del meta['Unnamed: 0']
meta = meta.set_index('SRR.ID')

plt.figure()
ax = meta['Age'].plot.hist(bins=100, color='k', alpha=1)
plt.title('Age distribution\nShokhirev 2020 dataset', fontsize = 16)
plt.xlabel('Age (yrs)', fontsize = 16)
plt.ylabel('Frecuency', fontsize = 16)
plt.xlim([0, 107])
plt.grid(alpha = 0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('age_histogram', dpi=200)

print(max(meta['Age']))
print(min(meta['Age']))
print(meta['Age'].mean())
print(meta['Age'].std())
