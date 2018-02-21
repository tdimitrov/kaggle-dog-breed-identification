import dogs as kd
import params

kd.helpers.create_dir(params.PROCESSED_DIR)

kd.data.convert_train_set(params.LABELS_CSV, params.IMAGES_TRAIN_DIR, params.Y_TRAIN_NPY, params.X_TRAIN_NPY, params.CLASSES_NPY)
kd.data.convert_test_set(params.IMAGES_TEST_DIR, params.X_TEST_NPY, params.X_TEST_IDS_NPY)
