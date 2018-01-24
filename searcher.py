import numpy as np


class Searcher:
    # def __init__(self):
    # storing index of images
    # self.index = index

    def chi2_distance(self, f_vector_a, f_vector_b, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(f_vector_a, f_vector_b)])

        return d

    def euclidian_distance(self, f_vector_a, f_vector_b):
        d = np.sum([abs(a-b) for (a,b) in zip(f_vector_a, f_vector_b)])
        return d

    '''
    ->shifted this part to main.py<-
    
    def search(self, queryFeatures):
        # init dict of results
        results = {}

        # loop over the index
        for (k, features) in self.index.items():
            # compute the chi-squared distance between the features
            # in our index and our query features -- using the
            # chi-squared distance
            d = self.euclidian_distance(features, queryFeatures)

            results[k] = d

        results = sorted([(v, k) for (k, v) in results.items()])
        return results'''