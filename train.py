import numpy as np
import dogs
import params
import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


classes = np.load(params.CLASSES_NPY)
num_classes = classes.shape[0]


datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, 
                                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
train_generator = datagen.flow_from_directory(directory=params.IMAGES_TRAIN_DIR, target_size=(224, 224), 
                                            classes=classes.tolist(), class_mode="categorical", batch_size=32)

cv_generator = datagen.flow_from_directory(directory=params.IMAGES_CV_DIR, target_size=(224, 224), 
                                            classes=classes.tolist(), class_mode="categorical", batch_size=int(10222/100))

# Setup early stopping
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto')

# Combine ResNet50 with the Dense NN
model = dogs.model.get_model(num_classes)
model.fit_generator(train_generator, steps_per_epoch=int(10222/32), epochs=params.EPOCHS, verbose=1, 
                        validation_data=cv_generator, validation_steps=int(10222/100), callbacks=[early_stop])

print("Saving model to %s" % params.MODEL_H5)
model.save(params.MODEL_H5)
print("Done")
