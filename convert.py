import dogs as kd
import os
import sys
import params

def create_dir(dir_name):
    if os.path.exists(dir_name) == False:
        try:
            os.mkdir(dir_name)
            print("Created dir %s" % dir_name)
        except FileExistsError:
            print("%s exists, but it's not a directory. Exiting." % dir_name)
            sys.exit(1)

create_dir(params.PROCESSED_DIR)
create_dir(params.TRAIN_DIR)

kd.data.convert_train_set(params.LABELS_CSV, params.IMAGES_TRAIN_DIR, params.Y_TRAIN_NPY, params.X_TRAIN_NPY, 
                            params.CLASSES_NPY)

kd.data.convert_test_set(params.IMAGES_TEST_DIR, params.X_TEST_NPY, params.X_TEST_IDS_NPY)

kd.data.augment(params.X_TRAIN_NPY, params.Y_TRAIN_NPY, params.AUG_X_BASENAME, params.AUG_Y_BASENAME, 
                    params.AUG_BATCH_SIZE, params.AUG_BATCH_COUNT)