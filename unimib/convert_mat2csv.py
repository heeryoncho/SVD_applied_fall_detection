import scipy.io
import pandas as pd

# Convert 'acc_data.mat' and 'acc_labels.mat' files in UniMiB data directory into CSV files.

dir_path = '../raw_dataset/UniMiB/data/'

in_data = dir_path + 'acc_data.mat'
in_labels = dir_path + 'acc_labels.mat'

out_data = dir_path + 'acc_data.csv'
out_labels = dir_path + 'acc_labels.csv'

mat = scipy.io.loadmat(in_data)
acc_data = mat['acc_data']
acc_data = acc_data.round(decimals=5)
pd.DataFrame(acc_data).to_csv(out_data, index=None, header=None)

mat = scipy.io.loadmat(in_labels)
acc_labels = mat['acc_labels']
pd.DataFrame(acc_labels).to_csv(out_labels, index=None, header=None)
