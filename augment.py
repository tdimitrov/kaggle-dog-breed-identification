import dogs as kd
import params
import numpy as np
import os

kd.helpers.create_dir(params.AUG_DIR)

x = np.load(params.X_TRAIN_NPY)
y = np.load(params.Y_TRAIN_NPY)

m = x.shape[0]  #10222
batch_per_run = 3000
runs = int(m / batch_per_run) + 1

for i in range(runs):
    if i == runs - 1:
        b = i*batch_per_run
        e = m
    else:
        b = i*batch_per_run
        e = b + batch_per_run
    
    print("Batch %d of %d" % (i+1, runs))

    try:
        pid = os.fork()
    except OSError:
        print("Can't fork")
        exit()
    
    if pid == 0:
        # The function is called in a fork due to a memory leak caused either by a bug in 
        # keras.preprocessing.image.ImageDataGenerator or misuse of the class by me.
        # In nutshell - if the generator is reinitialised the memory for the input images is not deallocated,
        # which caused the host system to run out of memory.
        # Running each time in a separate process guarantees that all memory is deallocated.
        # The issue was experienced with keras 2.1.3 and python 3.6.3
        kd.data.augment(x[b:e], y[b:e], params.AUG_X_BASENAME, params.AUG_Y_BASENAME, params.BATCH_SIZE, i*10, 10)
        exit(0)
    else:
        os.waitpid(0, 0)