# import os
# import cv2
# import h5py
# import glob
# import datetime
# import argparse
import numpy as np
import faiss_index
# from feature_extractor import get_feature_file
# Parse the arguments
'''
ap = argparse.ArgumentParser()
ap.add_argument('-q', '--query', required=True, help='Path to the query image')
ap.add_argument('-i', '--index', required=True, help='Path to the index file')
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory that contains images to be indexed")
args = vars(ap.parse_args())

# get info from the arguments
_img_path = args['query']
_index_file = args['index']
_dataset = args['dataset']
'''

# performance tester
_index_file = '/home/dinesh/Documents/pca6632dim_features_norm_hsv_vgg_ukbench_sklearn.hdf5'
_dataset = '/media/dinesh/dinesh/C/Documents/BTP/ukbench/ukbench/full'
_img_path = 0
itms,idx = faiss_index.build_index(_index_file)
print('indexing successfully done...\n')

'''

    Deprecated function using L2 without indexing

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
    results = sorted([(v, k) for (k, `v) in results.items()])
    return results
'''

# code for finding top 8 out of 100
# def get_better_results(itms,results):
def get_better_results(results):
    # (score,index)
    better_results = np.zeros((len(results),2))
    # query_image_feature = np.float32(itms[int(_img_path[_img_path.rfind('h')+1:_img_path.rfind('.')])])
    global _img_path
    query_image_feature = np.float32(itms[_img_path])
    _img_path += 1
    # print _img_path
    result_feature = (np.float32(itms[idx]) for score,idx in results)
    from itertools import starmap,izip
    # distance = lambda x,y:(np.sqrt(np.sum(np.square(abs(x-y)))))
    distance = lambda x,y : 1 - np.inner(x,y)/(np.sqrt(np.inner(x,x))*np.sqrt(np.inner(y,y)))
    temp = izip((query_image_feature for i in xrange(len(results))), result_feature)
    better_results[:,0] = np.fromiter(starmap(distance,temp),'float32')
    # better_results[:,1] = np.fromiter((idx for score,idx in results),'int')
    better_results[:,1] = np.array(results)[:,1]
    better_results = better_results[better_results[:,0].argsort()]
    # return better_results[:8]
    return better_results[:4, 1]

def get_results_faiss():
    
    # print 'trying to fetch results for',_img_path,'...\n'
    # i = int(_img_path[_img_path.rfind('h')+1:_img_path.rfind('.')])
    # itms = get_feature_file(_index_file)
    # # print itms.shape
    # query_features = itms[i]
    # query = np.zeros((1,6272))
    # query[0] = np.float32(query_features)
    # query = np.float32(query)
    # # 100 hence trying to find 8 out of 100
    # (score,j) = idx.search(query,100)
    # # (score,index)
    # results = zip(score[0], j[0])
    # results = get_better_results(itms,results)
    # return results

    # code for performance tester (multiple queries)
    dist_matrix, ind_matrix = idx.search(itms,100)
    gen_dist = (arr for arr in dist_matrix)
    gen_ind = (arr for arr in ind_matrix)
    results = map(zip,gen_dist,gen_ind)
    # return np.array(results)[:, :4, 1]
    better_results = map(get_better_results,(arr for arr in results))
    return better_results


'''
# load the query image and show it
query_image = cv2.imread(_img_path)
query_image = cv2.resize(query_image, (400, 166))
cv2.imshow("Query", query_image)
print("Query: %s" % _img_path)
start = datetime.datetime.now()

results = get_results_faiss()
print('results are ready to be processed...\n')
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
    imageName = '0000000' + str(imageName)
    imageName = imageName[:imageName.rfind('.')]
    imageName = imageName[imageName.__len__()-5:]
    path = _dataset + os.path.sep + "ukbench" + "%s.jpg" % imageName
    print(path)
    result = cv2.imread(path)
    result = cv2.resize(result, (400, 166))

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

'''