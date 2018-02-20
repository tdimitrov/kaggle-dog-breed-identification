from keras.preprocessing.image import img_to_array, load_img
from keras.layers import Input
from keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
from glob import glob
import os.path


t_size = (224, 224, 3)  # ResNet50's default input is 224x224

def convert_train_set(labels_csv, images_dir, out_labels_fname, out_train_img_fname, out_classes_fname):
    """Convert training images to numpy array"""
    labels = pd.read_csv(labels_csv)

    classes = labels['breed'].unique()
    img_count = labels.count()[0]
    
    dataset = np.zeros(shape=(img_count, t_size[0], t_size[1], t_size[2]), dtype=np.uint8)

    print("Converting training images to numpy array")
    for img in labels.itertuples():
        fname = "%s/%s.jpg" % (images_dir, img[1])
        
        dataset[img[0]] = img_to_array(load_img(fname, target_size=t_size))
        if img[0] % 100 == 0:
            print("Read %d of %d images" % (img[0], img_count), end='\r')

    print("Saving labels to %s" % out_labels_fname)
    y = np.zeros((img_count, classes.shape[0]))
    for i in range(img_count):
        y[i] = labels['breed'][i] == classes
    np.save(out_labels_fname, y)

    print("Saving train images to %s" % out_train_img_fname)
    np.save(out_train_img_fname, dataset)
    
    print("Saving classes to %s" % out_classes_fname)
    np.save(out_classes_fname, classes)
    
    print("Done")
    return True


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
