import argparse
import glob
from feature_extractor import get_features
import h5py


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
index_file = args['index']

# f = open(index_file, "wb") uncomment for using cPickle

for imagePath in glob.glob(args["dataset"] + "/*.*"):
    # extract our unique image ID (i.e. the filename)
    with h5py.File(index_file, 'a') as h:
        k = imagePath[imagePath.rfind('/') + 1:]
        features = get_features(imagePath, args["layer"])
        h.create_dataset(k, data=features)

''' uncomment for cPickle
# use glob to grab the image paths and loop over them
for imagePath in glob.glob(args["dataset"] + "/*.*"):
    # extract our unique image ID (i.e. the filename)
    k = imagePath[imagePath.rfind('/') + 1:]
    # print(k, imagePath, end=' ')
    features = get_features(imagePath, args["layer"])
    with open(index_file, 'ab') as f:
        cPickle.dump({k : features}, f)

f.close()
'''