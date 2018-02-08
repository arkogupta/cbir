import numpy as np
import argparse
import cv2
from searcher import Searcher
# import _pickle as cPickle
from feature_extractor import get_features
import h5py
import datetime
import glob
import os

# Parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument('-q', '--query', required=True, help='Path to the query image')
ap.add_argument('-i', '--index', required=True, help='Path to the index file')
ap.add_argument("-l", "--layer", required=True, help="Name of the layer from which features are to be extracted")
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains images to be indexed")
args = vars(ap.parse_args())


# get info from the arguments
_img_path = args['query']
_index_file = args['index']
_layer_name = args['layer']
dataset = args['dataset']

# load the query image and show it


def get_results(img_path,index_file,layer_name):

    query_image = cv2.imread(img_path)
    query_image = cv2.resize(query_image, (400, 166))
    cv2.imshow("Query", query_image)
    print("query: %s" % img_path)

    # init dict of results and retrieve query features
    results = {}
    queryFeatures = get_features(img_path)
    searcher = Searcher()

    for imagePath in glob.glob(dataset + os.path.sep + "*.*"):
        key = imagePath[imagePath.rfind('h') + 1:]
        with h5py.File(index_file, 'r') as h:
            features = h[key]
            dist = searcher.euclidian_distance(queryFeatures, features)
            results[key] = dist
            print(dist)

    # sort the results by distance from query image
    results = sorted([(v, k) for (k, v) in results.items()])
    return results


results = get_results(_img_path, _index_file, _layer_name)
# initialize the two groups to display our results --
groupA = np.zeros((166 * 4, 400, 3), dtype="uint8")
groupB = np.zeros((166 * 4, 400, 3), dtype="uint8")

start = datetime.datetime.now()
# loop over the top 8 results
for j in range(0, 8):
    # grab the result (we are using row-major order) and
    # load the result image
    (score, imageName) = results[j]
    path = dataset + "%s" % imageName
    result = cv2.imread(path)
    result = cv2.resize(result, (400, 166))

    # print(path, result)

    print("\t%d. %s : %.3f" % (j + 1, imageName, score))

    # check to see if the first group should be used
    if j < 4:
        groupA[j * 166:(j + 1) * 166, :] = result

    # otherwise, the second group should be used
    else:
        groupB[(j - 4) * 166:((j - 4) + 1) * 166, :] = result

# show the results

total_time = (datetime.datetime.now() - start).total_seconds()
print("Time taken : %f seconds" %(total_time))

cv2.imshow("Results 1-4", groupA)
cv2.imshow("Results 5-8", groupB)
cv2.waitKey(0)
