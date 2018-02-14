import dogs
import params
from keras.models import load_model
import numpy as np
import pandas as pd


model = load_model(params.MODEL_H5)
X = np.load(params.TEST_FEATURES_NPY)
ids = np.load(params.TEST_IMG_IDS_NPY)
classes = np.load(params.CLASSES_NPY)

pred = model.predict(X, verbose=True)

data = np.hstack((ids, pred))
labels = np.hstack((["id"], classes))

submission = pd.DataFrame(data=data, columns=labels)

print("Saving submission to %s" % params.SUBMISSION_CSV)
submission.to_csv(params.SUBMISSION_CSV, index=False)
print("Done")