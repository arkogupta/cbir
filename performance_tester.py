import argparse
import glob
from main import get_results
import os

ap = argparse.ArgumentParser()
# ap.add_argument('-q', '--query', required=True, help='Path to the query image')
ap.add_argument('-i', '--index', required=True, help='Path to the index file')
ap.add_argument("-l", "--layer", required=True, help="Name of the layer from which features are to be extracted")
ap.add_argument('-d', '--dataset', required=True, help='Path to dataset')
args = vars(ap.parse_args())

# img_path = args['query']
index_file = args['index']
layer_name = args['layer']
dataset = args['dataset']

score = 0
image_count = 0
for image_path in glob.glob(dataset + os.path.sep + "*.*"):
    # print(image_path)
    image_id = int(image_path[image_path.rfind('h') + 1: image_path.rfind('.')])
    # print(image_id)
    results = get_results(image_path, index_file, layer_name)
    base = (image_id//4)*4
    group_ids = set()
    for i in range(0,4):
        group_ids.add(base + i)

    for i in range(0,4):
        (dist, imageName) = results[i]
        result_image_id = int(imageName[imageName.rfind('h') + 1: image_path.rfind('.')])
        if result_image_id in group_ids:
            score += 1
    image_count += 1

score = score/image_count
print("Accuracy : %.3f" % score)

