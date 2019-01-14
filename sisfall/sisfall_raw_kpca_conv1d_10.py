import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cPickle as pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
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

X_kpca = pickle.load(open("data/X_sisfall_kpca.p", "rb"))
X_raw = pickle.load(open("data/X_sisfall_raw.p", "rb"))
X = np.concatenate((X_kpca, X_raw), axis=1)

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

    new_dir = 'sisfall/model/sisfall_raw_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_raw_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.6294117647058823, 0.6235294117647059, 0.5294117647058824, 0.6294117647058823, 0.7, 0.6411764705882353, 0.6352941176470588, 0.6294117647058823, 0.7176470588235294, 0.5176470588235295, 0.5352941176470588, 0.6529411764705882, 0.6352941176470588, 0.5529411764705883, 0.5636363636363636, 0.6235294117647059, 0.5976331360946746, 0.7, 0.6823529411764706, 0.7108433734939759, 0.7, 0.5529411764705883, 0.6176470588235294]
0.625130227225
0.0593793766675
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_raw_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/1/weights.49-0.60.hdf5
------ TRAIN ACCURACY:  weights.49-0.60.hdf5  ------
0.955764075067
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   1 104 ...   0   0   0]
 ...
 [  0   0   0 ...  94   0   8]
 [  0   0   0 ...   0 104   0]
 [  0   0   0 ...   0   0 106]]
------ TEST ACCURACY:  weights.49-0.60.hdf5  ------
0.629411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 0 5]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/2/weights.23-0.63.hdf5
------ TRAIN ACCURACY:  weights.23-0.63.hdf5  ------
0.916890080429
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  86   0   6]
 [  0   0   0 ...   0 107   1]
 [  0   0   0 ...   1   1  94]]
------ TEST ACCURACY:  weights.23-0.63.hdf5  ------
0.623529411765
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 3 1 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/3/weights.14-0.61.hdf5
------ TRAIN ACCURACY:  weights.14-0.61.hdf5  ------
0.848525469169
[[103   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  91   2   2]
 [  0   0   0 ...   0  94   2]
 [  1   0   0 ...   8   1  78]]
------ TEST ACCURACY:  weights.14-0.61.hdf5  ------
0.529411764706
[[1 4 0 ... 0 0 0]
 [0 2 3 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/4/weights.45-0.62.hdf5
------ TRAIN ACCURACY:  weights.45-0.62.hdf5  ------
0.950402144772
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   1 104 ...   0   0   0]
 ...
 [  0   0   0 ...  89   0   8]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   0   1 103]]
------ TEST ACCURACY:  weights.45-0.62.hdf5  ------
0.629411764706
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/5/weights.17-0.62.hdf5
------ TRAIN ACCURACY:  weights.17-0.62.hdf5  ------
0.863806970509
[[ 97   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  48   2  23]
 [  0   0   0 ...   0 103   0]
 [  1   0   0 ...   0   0  96]]
------ TEST ACCURACY:  weights.17-0.62.hdf5  ------
0.7
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 3]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/6/weights.36-0.61.hdf5
------ TRAIN ACCURACY:  weights.36-0.61.hdf5  ------
0.945308310992
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  99   1   3]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   2 100]]
------ TEST ACCURACY:  weights.36-0.61.hdf5  ------
0.641176470588
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 0 4]
 [0 0 0 ... 1 1 3]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/7/weights.45-0.61.hdf5
------ TRAIN ACCURACY:  weights.45-0.61.hdf5  ------
0.955495978552
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 103   0   2]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   2   1 101]]
------ TEST ACCURACY:  weights.45-0.61.hdf5  ------
0.635294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 3 0]
 [1 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/8/weights.44-0.62.hdf5
------ TRAIN ACCURACY:  weights.44-0.62.hdf5  ------
0.955227882038
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   1 104 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   4]
 [  0   0   0 ...   0 103   1]
 [  0   0   0 ...   0   0 103]]
------ TEST ACCURACY:  weights.44-0.62.hdf5  ------
0.629411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/9/weights.42-0.62.hdf5
------ TRAIN ACCURACY:  weights.42-0.62.hdf5  ------
0.950670241287
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   2]
 [  0   0   0 ...   0 104   0]
 [  0   0   0 ...   0   1  98]]
------ TEST ACCURACY:  weights.42-0.62.hdf5  ------
0.717647058824
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/10/weights.32-0.62.hdf5
------ TRAIN ACCURACY:  weights.32-0.62.hdf5  ------
0.940214477212
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  93   0   7]
 [  0   0   0 ...   0 105   1]
 [  0   0   0 ...   0   0 104]]
------ TEST ACCURACY:  weights.32-0.62.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/11/weights.49-0.61.hdf5
------ TRAIN ACCURACY:  weights.49-0.61.hdf5  ------
0.955227882038
[[105   3   1 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   2]
 [  0   0   0 ...   0 104   0]
 [  0   0   0 ...   0   0 102]]
------ TEST ACCURACY:  weights.49-0.61.hdf5  ------
0.535294117647
[[3 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/12/weights.29-0.62.hdf5
------ TRAIN ACCURACY:  weights.29-0.62.hdf5  ------
0.928954423592
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  73   2  16]
 [  0   0   0 ...   0 105   1]
 [  0   0   0 ...   0   3 101]]
------ TEST ACCURACY:  weights.29-0.62.hdf5  ------
0.652941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/13/weights.39-0.62.hdf5
------ TRAIN ACCURACY:  weights.39-0.62.hdf5  ------
0.935120643432
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   1 105 ...   0   0   0]
 ...
 [  0   0   0 ...  86   0   8]
 [  0   0   0 ...   0 107   1]
 [  0   0   0 ...   0   2 102]]
------ TEST ACCURACY:  weights.39-0.62.hdf5  ------
0.635294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/14/weights.24-0.60.hdf5
------ TRAIN ACCURACY:  weights.24-0.60.hdf5  ------
0.914477211796
[[100   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  80   0   8]
 [  0   0   0 ...   0 102   1]
 [  0   0   0 ...   0   0  96]]
------ TEST ACCURACY:  weights.24-0.60.hdf5  ------
0.552941176471
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/15/weights.46-0.61.hdf5
------ TRAIN ACCURACY:  weights.46-0.61.hdf5  ------
0.94859437751
[[104   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   2]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   2   0  96]]
------ TEST ACCURACY:  weights.46-0.61.hdf5  ------
0.563636363636
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 1 2]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/16/weights.15-0.62.hdf5
------ TRAIN ACCURACY:  weights.15-0.62.hdf5  ------
0.858981233244
[[ 96   7   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  59   3   5]
 [  0   0   0 ...   0 102   1]
 [  1   0   0 ...   2   3  72]]
------ TEST ACCURACY:  weights.15-0.62.hdf5  ------
0.623529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/17/weights.22-0.60.hdf5
------ TRAIN ACCURACY:  weights.22-0.60.hdf5  ------
0.892790136693
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  76   2  11]
 [  0   0   0 ...   0 107   0]
 [  1   0   0 ...   1   7  96]]
------ TEST ACCURACY:  weights.22-0.60.hdf5  ------
0.597633136095
[[0 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/18/weights.16-0.62.hdf5
------ TRAIN ACCURACY:  weights.16-0.62.hdf5  ------
0.85254691689
[[ 99   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   1]
 [  0   0   0 ...   0  90   2]
 [  1   0   0 ...  13   0  70]]
------ TEST ACCURACY:  weights.16-0.62.hdf5  ------
0.7
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/19/weights.49-0.61.hdf5
------ TRAIN ACCURACY:  weights.49-0.61.hdf5  ------
0.957104557641
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 102   0   2]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   1   0 103]]
------ TEST ACCURACY:  weights.49-0.61.hdf5  ------
0.682352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/20/weights.34-0.62.hdf5
------ TRAIN ACCURACY:  weights.34-0.62.hdf5  ------
0.944831280129
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0   4]
 [  0   0   0 ...   0 105   1]
 [  0   0   0 ...   0   1 101]]
------ TEST ACCURACY:  weights.34-0.62.hdf5  ------
0.710843373494
[[4 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 3]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/21/weights.39-0.61.hdf5
------ TRAIN ACCURACY:  weights.39-0.61.hdf5  ------
0.945576407507
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   6]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   1   0  99]]
------ TEST ACCURACY:  weights.39-0.61.hdf5  ------
0.7
[[1 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/22/weights.25-0.71.hdf5
------ TRAIN ACCURACY:  weights.25-0.71.hdf5  ------
0.919034852547
[[105   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   1 109 ...   0   0   0]
 ...
 [  0   0   0 ...  94   0   1]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...  12   0  84]]
------ TEST ACCURACY:  weights.25-0.71.hdf5  ------
0.552941176471
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 1]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_kpca_conv1d_10/23/weights.24-0.65.hdf5
------ TRAIN ACCURACY:  weights.24-0.65.hdf5  ------
0.915549597855
[[103   3   0 ...   0   0   0]
 [  5 105   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  85   2  13]
 [  0   0   0 ...   0 109   0]
 [  0   0   0 ...   1   7  94]]
------ TEST ACCURACY:  weights.24-0.65.hdf5  ------
0.617647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 5]]
[0.6294117647058823, 0.6235294117647059, 0.5294117647058824, 0.6294117647058823, 0.7, 0.6411764705882353, 0.6352941176470588, 0.6294117647058823, 0.7176470588235294, 0.5176470588235295, 0.5352941176470588, 0.6529411764705882, 0.6352941176470588, 0.5529411764705883, 0.5636363636363636, 0.6235294117647059, 0.5976331360946746, 0.7, 0.6823529411764706, 0.7108433734939759, 0.7, 0.5529411764705883, 0.6176470588235294]
0.625130227225
0.0593793766675

Process finished with exit code 0

'''