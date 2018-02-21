import faiss
import h5py
import numpy as np
import argparse
"""
    creates a faiss index object which will be accessed by 'idx'
"""

def build_index(feature_file):
    file = h5py.File(feature_file,'r')
    itms = file.items()
    m = itms.__len__()
    #currently hardcoded     
    n = 25088
    xb = np.zeros((m,n))
    i = 0
    for imageName,features in itms:
        xb[i] = np.array(features)
        i += 1
    xb = np.float32(xb)
    idx = faiss.IndexFlatL2(n)
    idx.add(xb)
    return idx