import os
import re
import numpy as np
import pandas as pd
import cPickle as pickle

data_dir = "data/"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dir_path = '../raw_dataset/SisFall/SisFall_dataset/'

window_size = 450
begin = 100
end = 550

labels = []
X = []
for path, dirs, files in os.walk(dir_path):
    print "-------"
    dirs.sort()
    files.sort()
    for file in files:
        if "SA" in file:
            label_a = 0
            label_s = 0
            #print os.path.join(path, file)
            print file
            tag = file.split("_")
            label_a = int(re.findall('\d+', tag[0])[0]); #print label_a
            label_s = int(re.findall('\d+', tag[1])[0]); #print label_s
            if 'F' in tag[0]:
                label_a += 19

            file_str = os.path.join(path, file)
            df = pd.read_csv(file_str, header=None)
            #print list(df)
            #print df.shape
            stop = df.shape[0] / 4
            idx = []
            for k in range(stop):
                idx.append(k * 4)
            #print len(idx)
            df_converted = df.iloc[idx, 0:3] * 0.00390625
            #print df_converted.shape

            # calculate Signal Magnitude Vector (x, y, z)
            if label_a in [1, 2, 3, 4]:
                for j in range(5):
                    X_smv = np.linalg.norm(df_converted[(begin + j*window_size) : (end + j*window_size)].values, axis=1)
                    X.append(X_smv.reshape((window_size,)))
                    labels.append([label_a, label_s])
            else:
                X_smv = np.linalg.norm(df_converted[begin:end].values, axis=1)
                X.append(X_smv.reshape((window_size,)))
                labels.append([label_a, label_s])
#print len(labels)

X = np.vstack(X)
print X.shape # (3900, 450)

df_label = pd.DataFrame(labels)
print df_label.shape # (3900, 2)

pickle.dump(X, open("data/X_sisfall_smv.p","wb"))
pickle.dump(df_label, open("data/y_sisfall_smv.p","wb"))


'''

# Load Signal Magnitude Vector data

X = pickle.load(open("data/X_sisfall_smv.p", "rb"))

'''
