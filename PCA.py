import numpy as np
import h5py
from scipy import linalg as la
from itertools import chain


def apply_pca(feature_file,reduced_dims = 6272):
    h = h5py.File(feature_file,'r')
    data = np.fromiter(chain.from_iterable(np.array(feature) for imageName,feature in h.iteritems()),'float32')
    # hardcode
    dim = 25088
    data.shape = len(h),dim
    data -= data.mean(axis=0)
    Sigma = np.cov(data,rowvar=False)
    evals,evecs = la.eigh(Sigma)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evecs = evecs[:,:reduced_dims]
    return np.dot(data,evecs)