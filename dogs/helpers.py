import os
import sys

def create_dir(dir_name):
    if os.path.exists(dir_name) == False:
        try:
            os.mkdir(dir_name)
            print("Created dir %s" % dir_name)
        except FileExistsError:
            print("%s exists, but it's not a directory. Exiting." % dir_name)
            sys.exit(1)