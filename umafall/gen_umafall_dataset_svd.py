import os
import numpy as np
import pandas as pd
import cPickle as pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dir_path = '../raw_dataset/UMAFall/UMA_ADL_FALL_Dataset/'
window_size = 450
slide_size = 50
begin = 50
end = 500

activity_dict = {"Bending":3, "Hopping":4, "Jogging":2, "LyingDown":7,
                 "Sitting":8, "Walking":1, "GoDownstairs":6, "GoUpstairs":5,
                 "backwardFall":11, "forwardFall":10, "lateralFall":9}
labels = []
X = []
for path, dirs, files in os.walk(dir_path):
    print "-------"
    dirs.sort()
    files.sort()
    #print files
    for file in files:
        #print file
        file_str_split = file.split("_")
        print file_str_split
        label_s = int(file_str_split[2])
        label_a = activity_dict[file_str_split[4]]
        file_str = os.path.join(path, file)
        df = pd.read_csv(file_str, header=None, sep=";", comment="%")
        print df.shape
        #print df.head(3)
        #print list(df)
        df_selected = df.loc[(df[5] == 0) & (df[6] == 0)]
        #print df_selected.shape
        #print df_selected.iloc[:, 2:5]
        stop = df_selected.shape[0] / 4
        idx = []
        for k in range(stop):
            idx.append(k * 4)
        #print len(idx)
        df_converted = df.iloc[idx, 2:5]
        #print df_converted.shape

        # calculate Truncated SVD(x, y, z)
        scaler = StandardScaler()
        svd = TruncatedSVD(n_components=1, random_state=2018)

        for j in range(5):
            scaled = scaler.fit_transform(df_converted[(begin + j * slide_size) : (end + j * slide_size)].values)
            X.append(svd.fit_transform(scaled).reshape((window_size,)))
            labels.append([label_a, label_s])

#print len(labels)

X = np.vstack(X)
print X.shape # (2655, 450)

df_label = pd.DataFrame(labels)
print df_label.shape # (2655, 2)

pickle.dump(X, open("data/X_umafall_svd.p","wb"))
pickle.dump(df_label, open("data/y_umafall_svd.p","wb"))

'''

# Load Singular Vector Decomposition data

X = pickle.load(open("data/X_umafall_svd.p", "rb"))

'''
