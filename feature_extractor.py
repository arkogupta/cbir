from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from itertools import chain
import numpy as np
import h5py


layer = 'block5_pool'
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)

def get_neural_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features.flatten()
    return features

def get_feature_file(feature_file):
    h = h5py.File(feature_file,'r')
    # hardcoded
    m,dim = h.__len__(),6272
    xb = np.fromiter(chain.from_iterable(np.array(feature) for imageName,feature in h.iteritems()),'float32')
    xb.shape = m,dim    
    return np.float32(xb)
    