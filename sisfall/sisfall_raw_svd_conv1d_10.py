import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cPickle as pickle
from keras.models import Model, load_model
from keras.initializers import Constant, TruncatedNormal
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout, Activation, BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras import losses
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load train and test data

X_svd = pickle.load(open("data/X_sisfall_svd.p", "rb"))
X_raw = pickle.load(open("data/X_sisfall_raw.p", "rb"))
X = np.concatenate((X_svd, X_raw), axis=1)

y = pickle.load(open("data/y_sisfall_raw.p", "rb"))

n_classes = 34
signal_rows = 1800
signal_columns = 1
n_subject = 23


# Below builds 1D CNN.

for i in range(n_subject):
    test = y.loc[y[1] == i+1]
    # print y_sub
    test_index = test.index.values
    # print len(test_index)

    train = y[~y.index.isin(test_index)]
    train_index = train.index.values

    y_values = y.ix[:, 0].values

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_values[train_index], y_values[test_index]
    y_train = np.eye(n_classes)[y_train - 1]
    print X_train.shape #

    # input layer
    input_signal = Input(shape=(signal_rows, 1))
    print K.int_shape(input_signal)

    # define initial parameters
    b_init = Constant(value=0.0)
    k_init = TruncatedNormal(mean=0.0, stddev=0.01, seed=2018)

    # first feature extractor
    conv11 = Conv1D(16, kernel_size=32, strides=1, padding='valid', bias_initializer=b_init,
                    kernel_initializer=k_init)(input_signal)
    bn11 = BatchNormalization()(conv11)
    actv11 = Activation('relu')(bn11)
    conv12 = Conv1D(32, kernel_size=32, strides=1, padding='valid', bias_initializer=b_init,
                    kernel_initializer=k_init)(actv11)
    bn12 = BatchNormalization()(conv12)
    actv12 = Activation('relu')(bn12)
    flat1 = Flatten()(actv12)

    # second feature extractor
    conv21 = Conv1D(16, kernel_size=42, strides=1, padding='valid', bias_initializer=b_init,
                    kernel_initializer=k_init)(input_signal)
    bn21 = BatchNormalization()(conv21)
    actv21 = Activation('relu')(bn21)
    conv22 = Conv1D(32, kernel_size=42, strides=1, padding='valid', bias_initializer=b_init,
                    kernel_initializer=k_init)(actv21)
    bn22 = BatchNormalization()(conv22)
    actv22 = Activation('relu')(bn22)
    flat2 = Flatten()(actv22)

    # third feature extractor
    conv31 = Conv1D(16, kernel_size=52, strides=1, padding='valid', bias_initializer=b_init,
                    kernel_initializer=k_init)(input_signal)
    bn31 = BatchNormalization()(conv31)
    actv31 = Activation('relu')(bn31)
    conv32 = Conv1D(32, kernel_size=52, strides=1, padding='valid', bias_initializer=b_init,
                    kernel_initializer=k_init)(actv31)
    bn32 = BatchNormalization()(conv32)
    actv32 = Activation('relu')(bn32)
    flat3 = Flatten()(actv32)

    # merge feature extractors
    merge = concatenate([flat1, flat2, flat3])

    # dropout & fully-connected layer
    do = Dropout(0.9, seed=2018)(merge)
    output = Dense(n_classes, activation='softmax', bias_initializer=b_init, kernel_initializer=k_init)(do)

    # result
    model = Model(inputs=input_signal, outputs=output)

    # summarize layers
    print(model.summary())

    new_dir = 'model/sisfall_raw_svd_conv1d_10/' + str(i+1) + '/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    fpath = new_dir + 'weights.{epoch:02d}-{val_acc:.2f}.hdf5'
    cp_cb = ModelCheckpoint(fpath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)

    adam = Adam(lr=0.0001)
    model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
    model.fit(np.expand_dims(X_train, axis=2), y_train, batch_size=32, epochs=50, verbose=2, validation_split=0.10,
              callbacks=[cp_cb], shuffle=True)

    del model
    K.clear_session()


# Below calculates average test data accuracy (LOSO CV).

acc = []

for i in range(n_subject):
    test = y.loc[y[1] == i+1]
    test_index = test.index.values

    train = y[~y.index.isin(test_index)]
    train_index = train.index.values

    y_values = y.ix[:, 0].values

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_values[train_index] - 1, y_values[test_index] - 1

    print "\n>>>>>>>>>>>>>>", str(i+1), "-fold <<<<<<<<<<<<<<<<"
    path_str = 'model/sisfall_raw_svd_conv1d_10/' + str(i+1) + '/'
    for path, dirs, files in os.walk(path_str):
        dirs.sort()
        files.sort()
        top_acc = []
        top_acc.append(files[-1])
        files = top_acc
        for file in files:
            print "========================================"
            print os.path.join(path, file)
            model = load_model(os.path.join(path, file))
            pred = model.predict(np.expand_dims(X_train, axis=2), batch_size=32)
            print "------ TRAIN ACCURACY: ", file, " ------"
            print accuracy_score(y_train, np.argmax(pred, axis=1))
            print confusion_matrix(y_train, np.argmax(pred, axis=1))
            pred = model.predict(np.expand_dims(X_test, axis=2), batch_size=32)
            print "------ TEST ACCURACY: ", file, " ------"
            print accuracy_score(y_test, np.argmax(pred, axis=1))
            print confusion_matrix(y_test, np.argmax(pred, axis=1))

            del model
            K.clear_session()

    acc.append(accuracy_score(y_test, np.argmax(pred, axis=1)))

print acc
print np.mean(acc)
print np.std(acc)


'''
-----
[0.6352941176470588, 0.5647058823529412, 0.5352941176470588, 0.6176470588235294, 0.6588235294117647, 0.6647058823529411, 0.6470588235294118, 0.6764705882352942, 0.7470588235294118, 0.5823529411764706, 0.5, 0.6764705882352942, 0.6352941176470588, 0.5882352941176471, 0.5636363636363636, 0.6588235294117647, 0.6745562130177515, 0.7235294117647059, 0.6294117647058823, 0.7228915662650602, 0.7058823529411765, 0.4764705882352941, 0.6235294117647059]
0.630788824628
0.0687332269035
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_raw_svd_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/1/weights.17-0.60.hdf5
------ TRAIN ACCURACY:  weights.17-0.60.hdf5  ------
0.854423592493
[[103   4   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  56   2  10]
 [  0   0   0 ...   0  96   3]
 [  1   0   0 ...   2   0  99]]
------ TEST ACCURACY:  weights.17-0.60.hdf5  ------
0.635294117647
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/2/weights.34-0.62.hdf5
------ TRAIN ACCURACY:  weights.34-0.62.hdf5  ------
0.941286863271
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  93   1   4]
 [  0   0   0 ...   1 108   0]
 [  1   0   0 ...   1   7  92]]
------ TEST ACCURACY:  weights.34-0.62.hdf5  ------
0.564705882353
[[1 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 1 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/3/weights.15-0.62.hdf5
------ TRAIN ACCURACY:  weights.15-0.62.hdf5  ------
0.850938337802
[[105   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  87   1   6]
 [  0   0   0 ...   0 100   3]
 [  1   0   0 ...   9   2  91]]
------ TEST ACCURACY:  weights.15-0.62.hdf5  ------
0.535294117647
[[0 5 0 ... 0 0 0]
 [0 2 3 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/4/weights.30-0.61.hdf5
------ TRAIN ACCURACY:  weights.30-0.61.hdf5  ------
0.923324396783
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  83   1   5]
 [  0   0   0 ...   0 107   0]
 [  1   0   0 ...   1   1  98]]
------ TEST ACCURACY:  weights.30-0.61.hdf5  ------
0.617647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 5]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/5/weights.24-0.61.hdf5
------ TRAIN ACCURACY:  weights.24-0.61.hdf5  ------
0.916621983914
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  84   1   5]
 [  0   0   0 ...   0 105   0]
 [  1   0   0 ...   4   4  93]]
------ TEST ACCURACY:  weights.24-0.61.hdf5  ------
0.658823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 3]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/6/weights.42-0.60.hdf5
------ TRAIN ACCURACY:  weights.42-0.60.hdf5  ------
0.953351206434
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 103   0   1]
 [  0   0   0 ...   1 106   0]
 [  0   0   0 ...   0   3 100]]
------ TEST ACCURACY:  weights.42-0.60.hdf5  ------
0.664705882353
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 0 0 5]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/7/weights.25-0.62.hdf5
------ TRAIN ACCURACY:  weights.25-0.62.hdf5  ------
0.917962466488
[[105   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  84   2   4]
 [  0   0   0 ...   0 105   1]
 [  0   0   0 ...   1   1 102]]
------ TEST ACCURACY:  weights.25-0.62.hdf5  ------
0.647058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/8/weights.45-0.61.hdf5
------ TRAIN ACCURACY:  weights.45-0.61.hdf5  ------
0.946380697051
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  89   0   2]
 [  0   0   0 ...   0 104   0]
 [  0   0   0 ...   1   0 102]]
------ TEST ACCURACY:  weights.45-0.61.hdf5  ------
0.676470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/9/weights.21-0.60.hdf5
------ TRAIN ACCURACY:  weights.21-0.60.hdf5  ------
0.890884718499
[[105   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  70   2   8]
 [  0   0   0 ...   0 104   0]
 [  1   0   0 ...   1   3  89]]
------ TEST ACCURACY:  weights.21-0.60.hdf5  ------
0.747058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/10/weights.34-0.61.hdf5
------ TRAIN ACCURACY:  weights.34-0.61.hdf5  ------
0.942895442359
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   6]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   1   1 103]]
------ TEST ACCURACY:  weights.34-0.61.hdf5  ------
0.582352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 1 3]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/11/weights.33-0.60.hdf5
------ TRAIN ACCURACY:  weights.33-0.60.hdf5  ------
0.920107238606
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  65   4   3]
 [  0   0   0 ...   0 108   0]
 [  1   0   0 ...   0   9  81]]
------ TEST ACCURACY:  weights.33-0.60.hdf5  ------
0.5
[[1 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/12/weights.44-0.61.hdf5
------ TRAIN ACCURACY:  weights.44-0.61.hdf5  ------
0.950134048257
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  89   0   4]
 [  0   0   0 ...   0 107   0]
 [  1   0   0 ...   1   3 102]]
------ TEST ACCURACY:  weights.44-0.61.hdf5  ------
0.676470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/13/weights.38-0.63.hdf5
------ TRAIN ACCURACY:  weights.38-0.63.hdf5  ------
0.943163538874
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  94   0   4]
 [  0   0   0 ...   0 103   1]
 [  0   0   0 ...   2   0 104]]
------ TEST ACCURACY:  weights.38-0.63.hdf5  ------
0.635294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 2 1 1]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/14/weights.35-0.61.hdf5
------ TRAIN ACCURACY:  weights.35-0.61.hdf5  ------
0.940750670241
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   4]
 [  0   0   0 ...   0 105   0]
 [  1   0   0 ...   0   2  95]]
------ TEST ACCURACY:  weights.35-0.61.hdf5  ------
0.588235294118
[[2 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/15/weights.15-0.58.hdf5
------ TRAIN ACCURACY:  weights.15-0.58.hdf5  ------
0.846050870147
[[ 92   6   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  82   2  11]
 [  0   0   0 ...   0 101   0]
 [  0   0   0 ...   5   5  94]]
------ TEST ACCURACY:  weights.15-0.58.hdf5  ------
0.563636363636
[[4 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 2]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/16/weights.41-0.61.hdf5
------ TRAIN ACCURACY:  weights.41-0.61.hdf5  ------
0.950402144772
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  96   0   2]
 [  0   0   0 ...   1 104   0]
 [  0   0   0 ...   4   1  97]]
------ TEST ACCURACY:  weights.41-0.61.hdf5  ------
0.658823529412
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/17/weights.38-0.60.hdf5
------ TRAIN ACCURACY:  weights.38-0.60.hdf5  ------
0.947199142321
[[105   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   6]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   2   1 103]]
------ TEST ACCURACY:  weights.38-0.60.hdf5  ------
0.674556213018
[[0 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/18/weights.42-0.62.hdf5
------ TRAIN ACCURACY:  weights.42-0.62.hdf5  ------
0.95308310992
[[102   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   1   4]
 [  0   0   0 ...   0 109   0]
 [  0   0   0 ...   1   7  95]]
------ TEST ACCURACY:  weights.42-0.62.hdf5  ------
0.723529411765
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/19/weights.42-0.61.hdf5
------ TRAIN ACCURACY:  weights.42-0.61.hdf5  ------
0.946112600536
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ... 102   0   1]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   6   3  95]]
------ TEST ACCURACY:  weights.42-0.61.hdf5  ------
0.629411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/20/weights.40-0.59.hdf5
------ TRAIN ACCURACY:  weights.40-0.59.hdf5  ------
0.945099089448
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  94   0   4]
 [  0   0   0 ...   0 106   1]
 [  1   0   0 ...   0   1 103]]
------ TEST ACCURACY:  weights.40-0.59.hdf5  ------
0.722891566265
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 2]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/21/weights.18-0.60.hdf5
------ TRAIN ACCURACY:  weights.18-0.60.hdf5  ------
0.854691689008
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  93 ...   0   0   0]
 ...
 [  0   0   0 ...  85   3   2]
 [  0   0   0 ...   0 103   3]
 [  1   0   0 ...  10   4  80]]
------ TEST ACCURACY:  weights.18-0.60.hdf5  ------
0.705882352941
[[2 0 0 ... 0 0 0]
 [5 0 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/22/weights.39-0.69.hdf5
------ TRAIN ACCURACY:  weights.39-0.69.hdf5  ------
0.936997319035
[[104   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  91   0   2]
 [  0   0   0 ...   0 109   0]
 [  0   0   0 ...   0   2 101]]
------ TEST ACCURACY:  weights.39-0.69.hdf5  ------
0.476470588235
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_svd_conv1d_10/23/weights.49-0.65.hdf5
------ TRAIN ACCURACY:  weights.49-0.65.hdf5  ------
0.958981233244
[[ 99   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0   7]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   1 103]]
------ TEST ACCURACY:  weights.49-0.65.hdf5  ------
0.623529411765
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 5]]
[0.6352941176470588, 0.5647058823529412, 0.5352941176470588, 0.6176470588235294, 0.6588235294117647, 0.6647058823529411, 0.6470588235294118, 0.6764705882352942, 0.7470588235294118, 0.5823529411764706, 0.5, 0.6764705882352942, 0.6352941176470588, 0.5882352941176471, 0.5636363636363636, 0.6588235294117647, 0.6745562130177515, 0.7235294117647059, 0.6294117647058823, 0.7228915662650602, 0.7058823529411765, 0.4764705882352941, 0.6235294117647059]
0.630788824628
0.0687332269035

Process finished with exit code 0

'''