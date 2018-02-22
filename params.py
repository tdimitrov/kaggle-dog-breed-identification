LABELS_CSV='./data/input/labels.csv'
IMAGES_TRAIN_DIR='./data/input/train'
IMAGES_TEST_DIR='./data/input/test'

PROCESSED_DIR = './data/processed'
Y_TRAIN_NPY = "%s/labels.npy" % PROCESSED_DIR
X_TRAIN_NPY = "%s/train-images.npy" % PROCESSED_DIR
CLASSES_NPY = "%s/classes.npy" % PROCESSED_DIR
X_TEST_NPY = "%s/test-images.npy" % PROCESSED_DIR
X_TEST_IDS_NPY = "%s/test-images-ids.npy" % PROCESSED_DIR

FEATURES_DIR = "./data/features"
TRAIN_FEATURES_NPY = "%s/original_train.npy" % FEATURES_DIR
TEST_FEATURES_NPY = "%s/test.npy" % FEATURES_DIR
AUG_FEATURES_BASENAME = "%s/augmented_" % FEATURES_DIR


MODEL_H5 = "%s/model-weights.h5" % PROCESSED_DIR

AUG_DIR = "./data/augmented"
AUG_X_BASENAME = "%s/x_" % AUG_DIR
AUG_Y_BASENAME = "%s/y_" % AUG_DIR


EPOCHS = 100


SUBMISSION_CSV = "data/submission.csv"
