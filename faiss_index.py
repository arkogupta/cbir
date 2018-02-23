import faiss
import h5py
import numpy as np
import argparse
"""
    creates a faiss index object
"""


def build_index(feature_file):
    h = h5py.File(feature_file, 'r')
    itms = h.items()
    m = itms.__len__()
    # currently hardcoded
    dim = 25088
    xb = np.zeros((m,dim))
    i = 0
    for imageName,features in itms:
        xb[i] = np.array(features)
        i += 1
    xb = np.float32(xb)
    idx = faiss.IndexFlatL2(dim)
    idx.add(xb)
    return idx,xb
