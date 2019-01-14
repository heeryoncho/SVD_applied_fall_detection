import pandas as pd
import cPickle as pickle

# Load raw train and test data

dir_path = '../raw_dataset/UniMiB/data/'

X = pd.read_csv(dir_path + "acc_data.csv", header=None)
y = pd.read_csv(dir_path + "acc_labels.csv", header=None)

X_raw = X.values

print type(X_raw)
print X.shape
print y.shape

pickle.dump(X_raw, open("data/X_unimib_raw.p","wb"))
pickle.dump(y, open("data/y_unimib_raw.p", "wb"))