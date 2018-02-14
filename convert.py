import dogs as kd
import os
import sys
import params

if os.path.exists(params.OUT_DIR) == False:
    try:
        os.mkdir(params.OUT_DIR)
        print("Created dir %s" % params.OUT_DIR)
    except FileExistsError as e:
        print("%s exists, but it's not a directory. Exiting." % params.OUT_DIR)
        sys.exit(1)

kd.data.convert_train_set(params.LABELS_CSV, params.IMAGES_TRAIN_DIR, params.LABELS_NPY, params.TRAIN_IMG_NPY, params.CLASSES_NPY)
kd.data.convert_test_set(params.IMAGES_TEST_DIR, params.TEST_IMG_NPY, params.TEST_IMG_IDS_NPY)