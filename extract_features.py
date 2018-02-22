import dogs as kd
import params
import numpy as np

train_data = np.load(params.X_TRAIN_NPY)
test_data = np.load(params.X_TEST_NPY)
labels = np.load(params.Y_TRAIN_NPY)

kd.helpers.create_dir(params.FEATURES_DIR)

m = kd.model.get_resnet50()

# Convert original dataset
kd.model.extract_features(m, train_data, params.TRAIN_FEATURES_NPY)
kd.model.extract_features(m, test_data, params.TEST_FEATURES_NPY)

# Convert augmented images
x_list = kd.helpers.get_aug_x_files(params.AUG_X_BASENAME)

for fname_x in x_list:
    id = kd.helpers.get_aug_file_id(fname_x, params.AUG_X_BASENAME) # get id from fname
    x = np.load(fname_x)

    fname_dest = "%s%d.npy" % (params.AUG_FEATURES_BASENAME, id)    # fname for the features
    
    kd.model.extract_features(m, x, fname_dest)
    