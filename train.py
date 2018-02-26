import numpy as np
import dogs
import params
import os
import sys
import glob
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping


classes = np.load(params.CLASSES_NPY)

num_classes = classes.shape[0]
num_train = len(glob.glob(params.IMAGES_TRAIN_DIR + '/*/*.jpg'))
num_cv = len(glob.glob(params.IMAGES_CV_DIR + '/*/*.jpg'))
batch_size = 32

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, 
                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
train_generator = datagen.flow_from_directory(directory=params.IMAGES_TRAIN_DIR, target_size=(224, 224), 
                                            classes=classes.tolist(), class_mode="categorical", batch_size=batch_size)

cv_generator = datagen.flow_from_directory(directory=params.IMAGES_CV_DIR, target_size=(224, 224), 
                                            classes=classes.tolist(), class_mode="categorical", batch_size=batch_size)

# Setup early stopping
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')

# Combine ResNet50 with the Dense NN
if os.path.isfile(params.MODEL_H5):
    print("Loading %s" % params.MODEL_H5)
    model = load_model(params.MODEL_H5)
else:
    print("Training model from scratch")
    model = dogs.model.get_model(num_classes)

model.fit_generator(train_generator, steps_per_epoch=num_train//batch_size, epochs=params.EPOCHS, verbose=1, 
                        validation_data=cv_generator, validation_steps=num_cv//batch_size, callbacks=[early_stop])

print("Saving model to %s" % params.MODEL_H5)
model.save(params.MODEL_H5)
print("Done")
