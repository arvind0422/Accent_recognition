import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Getting dimension from the command line
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--path', type=str, help="Path to non-dimensionality-reduced fstat files")
parser.add_argument('--n', type=int, help="Number of output dimensions", default=200)
args = parser.parse_args()
print(args.n)


# Read fstat matrix
newlist = [f for f in os.listdir(args.path) if f.endswith('.fstat')]
fstat_matrix=[]
for f in newlist:
        fstat_matrix.append(np.fromfile(open(args.path+f,"r"), dtype=np.double))
fstat_matrix = np.asarray(fstat_matrix)
print(fstat_matrix.shape)

# Scaling
scaler = StandardScaler()
scaler.fit(fstat_matrix)
fstat_matrix = scaler.transform(fstat_matrix)

# PCA
pca = PCA(args.n)
pca.fit(fstat_matrix)
fstat_matrix = pca.transform(fstat_matrix)
print(fstat