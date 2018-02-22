import os
import sys
import glob

def create_dir(dir_name):
    if os.path.exists(dir_name) == False:
        try:
            os.mkdir(dir_name)
            print("Created dir %s" % dir_name)
        except FileExistsError:
            print("%s exists, but it's not a directory. Exiting." % dir_name)
            sys.exit(1)

def get_aug_x_files(aug_x_basename):
    pattern = "%s*" % (aug_x_basename)
    return glob.glob(pattern)

def get_aug_file_id(aug_filename, aug_basename):
    return int(aug_filename.replace(aug_basename, '').replace('.npy', ''))  # get id from fname