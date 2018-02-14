LABELS_CSV='./data/input/labels.csv'
IMAGES_TRAIN_DIR='./data/input/train'
IMAGES_TEST_DIR='./data/input/test'

OUT_DIR = './data/processed'
LABELS_NPY = "%s/labels.npy" % OUT_DIR
TRAIN_IMG_NPY = "%s/train-images.npy" % OUT_DIR
CLASSES_NPY = "%s/classes.npy" % OUT_DIR
TEST_IMG_NPY = "%s/test-images.npy" % OUT_DIR
TEST_IMG_IDS_NPY = "%s/test-images-ids.npy" % OUT_DIR
TRAIN_FEATURES_NPY = "%s/train-features-resnet50.npy" % OUT_DIR
TEST_FEATURES_NPY = "%s/test-features-resnet50.npy" % OUT_DIR
MODEL_H5 = "%s/model-weights.h5" % OUT_DIR

EPOCHS = 100

SUBMISSION_CSV = "data/submission.csv"
