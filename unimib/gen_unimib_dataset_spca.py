import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler

# Load train and test data

dir_path = '../raw_dataset/UniMiB/data/'

X = pd.read_csv(dir_path + "acc_data.csv", header=None)
y = pd.read_csv(dir_path + "acc_labels.csv", header=None)

n_classes = 17
signal_rows = 151
signal_columns = 3

# Generate Sparse PCA

X_raw = X.values.reshape(X.shape[0], signal_rows, signal_columns)
print X_raw.shape
scaler = StandardScaler()
spca = SparsePCA(n_components=1, random_state=2018, n_jobs=-1, method='cd')

n = 0
X_spca = np.empty([11771, 151])
for each in X_raw:
    #print n
    scaled = scaler.fit_transform(each)
    X_spca[n] = spca.fit_transform(scaled).reshape((151,))
    n += 1

print X_spca[1]
print X_spca[1].shape

pickle.dump(X_spca, open("data/X_unimib_spca.p","wb"))
pickle.dump(y, open("data/y_unimib_spca.p", "wb"))
