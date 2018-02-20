import dogs as kd
import os
import sys
import params

if os.path.exists(params.PROCESSED_DIR) == False:
    try:
        os.mkdir(params.PROCESSED_DIR)
        print("Created dir %s" % params.PROCESSED_DIR)
    except FileExistsError as e:
        print("%s exists, but it's not a directory. Exiting." % params.PROCESSED_DIR)
        sys.exit(1)

if os.path.exists(params.TRAIN_DIR) == False:
    try:
        os.mkdir(params.TRAIN_DIR)
        print("Created dir %s" % params.TRAIN_DIR)
    except FileExistsError as e:
        print("%s exists, but it's not a directory. Exiting." % params.TRAIN_DIR)
        sys.exit(1)

kd.data.convert_train_set(params.LABELS_CSV, params.IMAGES_TRAIN_DIR, params.Y_TRAIN_NPY, params.X_TRAIN_NPY, 
                            params.CLASSES_NPY)

#kd.data.convert_test_set(params.IMAGES_TEST_DIR, params.X_TEST_NPY, params.X_TEST_IDS_NPY)

kd.data.augment(params.X_TRAIN_NPY, params.Y_TRAIN_NPY, params.AUG_X_BASENAME, params.AUG_Y_BASENAME, 
                    params.AUG_BATCH_SIZE, params.AUG_BATCH_COUNT)