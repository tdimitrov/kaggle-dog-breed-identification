import dogs.model as km
import numpy as np
import params

train_data = np.load(params.TRAIN_IMG_NPY)
test_data = np.load(params.TEST_IMG_NPY)
labels = np.load(params.LABELS_NPY)

m = km.get_resnet50()

km.extract_features(m, train_data, params.TRAIN_FEATURES_NPY)
km.extract_features(m, test_data, params.TEST_FEATURES_NPY)