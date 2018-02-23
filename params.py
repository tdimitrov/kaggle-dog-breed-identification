LABELS_CSV = './data/input/labels.csv'
IMAGES_TRAIN_DIR = './data/input/train'
IMAGES_TEST_DIR = './data/input/test'
IMAGES_CV_DIR = './data/input/cv'

PROCESSED_DIR = './data/processed'
X_TEST_NPY = "%s/test-images.npy" % PROCESSED_DIR
X_TEST_IDS_NPY = "%s/test-images-ids.npy" % PROCESSED_DIR
CLASSES_NPY = "%s/classes.npy" % PROCESSED_DIR
MODEL_H5 = "%s/model-weights.h5" % PROCESSED_DIR

EPOCHS = 100
SUBMISSION_CSV = "data/submission.csv"
