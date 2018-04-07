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

# For testing PCA unquote following code

from datetime import datetime
feature_file = '/home/da_230896/index_debug.hdf5'
start = datetime.now()
reduced_dim = 10114
idx = apply_pca(feature_file,reduced_dim)
print("PCA working fine...")
print("Time taken: %f secs" % (datetime.now() - start).total_seconds())

# f = h5py.File('/output/pca_index.hdf5','w')
# with h5py.File('/home/da_230896/Documents/cbir/pca_index_debug.hdf5','w') as f:
#     f.create_dataset("PCA on Ukbench",data = idx)

output = '/home/da_230896/pca_index_debug.hdf5'
for imagePath in xrange(10200):
    # extract our unique image ID (i.e. the filename)
    # k = imagePath[imagePath.rfind('h') + 1:]
    k = str(imagePath) + '.jpg'
    features = idx[imagePath]
    with h5py.File(output, 'a') as h:
        h.create_dataset(k, data=features)