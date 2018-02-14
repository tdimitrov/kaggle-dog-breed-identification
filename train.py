import numpy as np
import dogs
import params
import os
import sys


labels = np.load(params.LABELS_NPY)
classes = np.load(params.CLASSES_NPY)
features = np.load(params.TRAIN_FEATURES_NPY)

# binary labels
y = np.zeros((features.shape[0], classes.shape[0]))
for i in range(y.shape[0]):
    y[i] = labels[i] == classes

model = dogs.model.get_dense(features[0].shape, np.unique(labels).shape[0])
model.fit(features, y, epochs=params.EPOCHS, sample_weight=None, verbose=True, validation_split=0.3)
print("Saving model to %s" % params.MODEL_H5)
model.save(params.MODEL_H5)
print("Done")
