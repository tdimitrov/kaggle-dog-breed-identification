import numpy as np
import dogs
import params
import os
import sys


y = np.load(params.Y_TRAIN_NPY)
classes = np.load(params.CLASSES_NPY)
features = np.load(params.TRAIN_FEATURES_NPY)

model = dogs.model.get_dense(features[0].shape, classes.shape[0])

# Train on original images
model.fit(features, y, epochs=params.EPOCHS, sample_weight=None, verbose=True, validation_split=0.3)

#Train on augmented images
adg = dogs.data.AugDataGenerator(params.AUG_FEATURES_BASENAME, params.AUG_Y_BASENAME)
steps = adg.get_steps()
g = adg.generate()
model.fit_generator(g, steps_per_epoch=steps, epochs=params.EPOCHS, verbose=1)

print("Saving model to %s" % params.MODEL_H5)
model.save(params.MODEL_H5)
print("Done")
