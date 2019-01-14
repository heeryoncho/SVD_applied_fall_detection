import numpy as np
import pandas as pd
import cPickle as pickle

# Load train and test data

dir_path = '../raw_dataset/UniMiB/data/'

X = pd.read_csv(dir_path + "acc_data.csv", header=None)
y = pd.read_csv(dir_path + "acc_labels.csv", header=None)

n_classes = 17
signal_rows = 151
signal_columns = 3

# Generate signal magnitude vector

# reshape data into (151, 3)
X_reshape = X.values.reshape(X.shape[0], signal_rows, signal_columns)
#print X_reshape.shape

# calculate signal magnitude vector (x, y, z)
X_smv = np.linalg.norm(X_reshape, axis=2)
#print X_smv.shape

pickle.dump(X_smv, open("data/X_unimib_smv.p","wb"))
pickle.dump(y, open("data/y_unimib_smv.p", "wb"))
