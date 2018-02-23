from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
from glob import glob
import os
import dogs.helpers as dh


t_size = (224, 224, 3)  # ResNet50's default input is 224x224

def categorise(labels_csv, images_dir, out_classes_fname):
    """Convert training images to numpy array"""

    labels = pd.read_csv(labels_csv)
    classes = labels['breed'].unique()
    
    for c in classes:
        dir_name = "%s/%s" % (images_dir, c)
        dh.create_dir(dir_name)
    
    for r in labels.itertuples():
        src = "%s/%s.jpg" % (images_dir, r[1])
        dest = "%s/%s/%s.jpg" % (images_dir, r[2], r[1])
        os.rename(src, dest)
        print("mv %s %s" % (src, dest))
    
    print("Saving classes to %s" % out_classes_fname)
    np.save(out_classes_fname, classes)
    print("Done")
    return True


def split_cv(train_dir, cv_dir, labels_csv):
    labels = pd.read_csv(labels_csv)
    classes = labels['breed'].unique()

    dh.create_dir(cv_dir)
    for c in classes:
        # create the category in cv dir
        dir_name = "%s/%s" % (cv_dir, c)
        dh.create_dir(dir_name)

        # get list of all files in train dir and move 1% in cv dir
        files = glob("%s/%s/*.jpg" % (train_dir, c))
        count = max(1, int(len(files)/100))   
        for i in range(count):
            dst = "%s/%s/%s" % (cv_dir, c, os.path.basename(files[i]))
            os.rename(files[i], dst)
            print("mv %s %s" % (files[i], dst))


def convert_test_set(images_dir, out_test_img_fname, out_test_img_ids_fname):
    """Convert test (submission) images to numpy array"""
    images = glob(images_dir + "/*.jpg")
    img_count = len(images)
    dataset = np.zeros(shape=(img_count, t_size[0], t_size[1], t_size[2]), dtype=np.uint8)

    print("Converting test (submission) images to numpy array")
    img_ids = []
    for i, fname in enumerate(images):
        dataset[i] = img_to_array(load_img(fname, target_size=t_size))
        # get the filename of the image without the extension
        # it's used in the submission as image id
        img_id = os.path.basename(fname)
        img_id = os.path.splitext(img_id)[0]
        img_ids.append(img_id)

        if i % 100 == 0:
            print("Read %d of %d images" % (i, img_count), end='\r')

    print("Saving test images to %s" % out_test_img_fname)
    np.save(out_test_img_fname, dataset)
    print("Saving test images IDs to %s" % out_test_img_ids_fname)
    np.save(out_test_img_ids_fname, np.array(img_ids).reshape(-1,1))
    print("Done")
    return True
