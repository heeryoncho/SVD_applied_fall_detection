import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Load train and test data

dir_path = '../raw_dataset/UniMiB/data/'

X = pd.read_csv(dir_path + "acc_data.csv", header=None)
y = pd.read_csv(dir_path + "acc_labels.csv", header=None)

n_classes = 17
signal_rows = 151
signal_columns = 3

# Generate SVD

X_raw = X.values.reshape(X.shape[0], signal_rows, signal_columns)
print X_raw.shape
scaler = StandardScaler()
svd = TruncatedSVD(n_components=1, random_state=2018)

n = 0
X_svd = np.empty([11771, 151])
for each in X_raw:
    scaled = scaler.fit_transform(each)
    X_svd[n] = svd.fit_transform(scaled).reshape((151,))
    n += 1

print X_svd[1]
print X_svd[1].shape

pickle.dump(X_svd, open("data/X_unimib_svd.p","wb"))
pickle.dump(y, open("data/y_unimib_svd.p", "wb"))
