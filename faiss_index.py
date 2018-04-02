import faiss
import h5py
import numpy as np
from itertools import chain
from PCA import apply_pca as PCA
from feature_extractor import get_feature_file
"""
    creates a faiss index object
    updated to create index without explicitely having dictionary in mem
    --should increase speed--
"""


def build_index(feature_file):
    xb = get_feature_file(feature_file)
    m,dim = xb.shape

    # reduced_dim = 6272
    # xb = PCA(feature_file,reduced_dim)
    
    idx = faiss.IndexFlatL2(dim)
    idx.add(xb)
    # return idx

    # for performance tester
    return xb,idx

# For testing PCA unquote following code
'''
from datetime import datetime
feature_file = '/home/dinesh/Documents/index_debug.hdf5'
# feature_file = '/input/index_debug.hdf5'
start = datetime.now()
idx = build_index(feature_file)
print("PCA working fine...")
print("Time taken: %f secs" % (datetime.now() - start).total_seconds())

# f = h5py.File('/output/pca_index.hdf5','w')
with h5py.File('/home/dinesh/Documents/cbir/pca_index.hdf5','w') as f:
    f.create_dataset("PCA on Ukbench",data = idx)
'''