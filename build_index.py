import argparse
import glob
from feature_extractor import get_features,layer
import h5py
import os
from sklearn.decomposition import PCA
import numpy as np

# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required="True",
                help="Path to the directory that contains images to be indexed")

ap.add_argument("-i", "--index", required="True",
                help="Path to where the computed index will be stored")

ap.add_argument("-l", "--layer", required="True",
                help="Name of the layer from which features are to be extracted")

args = vars(ap.parse_args())


print('Creating Index...')
index_file = args['index'] # 'index_pca.hdf5'
dataset = args['dataset']


def apply_pca():

    dataset_size = 112
    dim = 100352
    i = 0
    matrix = np.zeros((dataset_size, dim))
    for imagePath in glob.glob(dataset + os.path.sep + "*.*"):
        # extract our unique image ID (i.e. the filename)
        features = get_features(imagePath)
        matrix[i] = features
        i += 1

    print(matrix.shape)
    reduced_dim = 100
    pca = PCA(n_components=reduced_dim)
    principal_comp = pca.fit_transform(matrix)
    print(principal_comp.shape)
    # print()
    i = 0
    for imagePath in glob.glob(dataset + os.path.sep + "*.*"):
        with h5py.File(index_file, 'a') as h:
            k = imagePath[imagePath.rfind('h') + 1:]
            h.create_dataset(k, data=principal_comp[i])
            i += 1


# apply_pca()

for imagePath in glob.glob(dataset + os.path.sep + "*.*"):
    # extract our unique image ID (i.e. the filename)
    k = imagePath[imagePath.rfind('h') + 1:]
    features = get_features(imagePath)
    with h5py.File(index_file, 'a') as h:
        h.create_dataset(k, data=features)