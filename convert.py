import dogs as kd
import params

kd.helpers.create_dir(params.PROCESSED_DIR)


kd.data.categorise(params.LABELS_CSV, params.IMAGES_TRAIN_DIR, params.CLASSES_NPY)
kd.data.split_cv(params.IMAGES_TRAIN_DIR, params.IMAGES_CV_DIR, params.LABELS_CSV)
kd.data.convert_test_set(params.IMAGES_TEST_DIR, params.X_TEST_NPY, params.X_TEST_IDS_NPY)
