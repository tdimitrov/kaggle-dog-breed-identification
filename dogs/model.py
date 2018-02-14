from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.regularizers import l2
import numpy as np

def get_resnet50():
    model = ResNet50(include_top=False, weights='imagenet')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_dense(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(384, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile("sgd", "categorical_crossentropy")
    return model

def extract_features(pretrained_model, data, out_fname):
    print("Extracting features to %s...\n" % (out_fname))
    features = pretrained_model.predict(data, verbose=True)
    np.save(out_fname, features)
    print("Done")
    return True
