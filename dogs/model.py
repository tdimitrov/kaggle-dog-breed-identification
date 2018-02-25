from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Input, Flatten, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.regularizers import l2
import numpy as np


def get_model(num_classes):
    inp = Input(shape=(224, 224, 3))
    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=inp)
    
    # Freeze all layers
    for l in resnet.layers:
        l.trainable = False

    d = Flatten()(resnet.output)
    d = Dense(384, activation='relu', name='dense_hidden_1')(d)
    d = Dropout(0.6)(d)
    d = Dense(num_classes, activation='softmax', name='dense_out_1')(d)

    model = Model(inputs=inp, outputs=d)
    model.compile("sgd", "categorical_crossentropy")
    return model
