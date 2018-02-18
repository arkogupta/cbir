import numpy as np
import argparse
import cv2
from searcher import Searcher
from feature_extractor import get_features
import h5py
import datetime
# import glob
import os


# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-q', '--query', required=True, help='Path to the query image')
ap.add_argument('-i', '--index', required=True, help='Path to the index file')
# ap.add_argument("-l", "--layer", required=True, help="Name of the layer from which features are to be extracted")
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains images to be indexed")
args = vars(ap.parse_args())


# get info from the arguments
_img_path = args['query']
_index_file = args['index']
_dataset = args['dataset']


def get_results(img_path, index_file):
    # init dict of results and retrieve query features

    results = {}
    query_features = get_features(img_path)

    # print(query_features.size)
    searcher = Searcher()
    h = h5py.File(index_file, 'r')

    for file_name, features in h.items():
        features = np.array(features)
        # print(len(features))
        dist = searcher.euclidian_distance(query_features, features)
        results[file_name] = dist
    # sort the results by distance from query image
    results = sorted([(v, k) for (k, v) in results.items()])
    return results

# @Dinesh to complete this
def get_results_faiss(img_path, index_file):


    return results


# load the query image and show it

query_image = cv2.imread(_img_path)
query_image = cv2.resize(query_image, (400, 166))
cv2.imshow("Query", query_image)
print("query: %s" % _img_path)
start = datetime.datetime.now()
results = get_results(_img_path, _index_file)

total_time = (datetime.datetime.now() - start).total_seconds()
print("Time taken : %f seconds" %(total_time))

# initialize the two groups to display our results --
groupA = np.zeros((166 * 4, 400, 3), dtype="uint8")
groupB = np.zeros((166 * 4, 400, 3), dtype="uint8")

# loop over the top 8 results
for j in range(0, 8):
    # grab the result (we are using row-major order) and
    # load the result image
    (score, imageName) = results[j]
    path = _dataset + os.path.sep + "ukbench" + "%s" % imageName
    print(path)
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
cv2.imshow("Results 1-4", groupA)
cv2.imshow("Results 5-8", groupB)
cv2.waitKey(0)
