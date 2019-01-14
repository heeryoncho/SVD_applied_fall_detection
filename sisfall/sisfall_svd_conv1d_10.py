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

X = pickle.load(open("data/X_sisfall_svd.p", "rb"))
y = pickle.load(open("data/y_sisfall_svd.p", "rb"))

n_classes = 34
signal_rows = 450
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
    print X_train.shape # (3730, 450)

    # input layer
    input_signal = Input(shape=(signal_rows, 1))
    print K.int_shape(input_signal) # (None, 450, 1)

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

    new_dir = 'model/sisfall_svd_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_svd_conv1d_10/' + str(i+1) + '/'
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
[0.5529411764705883, 0.3588235294117647, 0.40588235294117647, 0.4823529411764706, 0.4764705882352941, 0.5352941176470588, 0.5, 0.48823529411764705, 0.5529411764705883, 0.4235294117647059, 0.49411764705882355, 0.5529411764705883, 0.48823529411764705, 0.4294117647058823, 0.5212121212121212, 0.5764705882352941, 0.4970414201183432, 0.6352941176470588, 0.5470588235294118, 0.5421686746987951, 0.5176470588235295, 0.38823529411764707, 0.5882352941176471]
0.502371298395
0.065777100022
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_svd_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/1/weights.30-0.49.hdf5
------ TRAIN ACCURACY:  weights.30-0.49.hdf5  ------
0.858981233244
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  87   4   8]
 [  0   0   0 ...   6  91   1]
 [  0   0   0 ...  11   6  75]]
------ TEST ACCURACY:  weights.30-0.49.hdf5  ------
0.552941176471
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 2 1]
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/2/weights.35-0.50.hdf5
------ TRAIN ACCURACY:  weights.35-0.50.hdf5  ------
0.87855227882
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  87   3   4]
 [  0   0   0 ...   6  88   3]
 [  0   0   0 ...   6   2  82]]
------ TEST ACCURACY:  weights.35-0.50.hdf5  ------
0.358823529412
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/3/weights.26-0.49.hdf5
------ TRAIN ACCURACY:  weights.26-0.49.hdf5  ------
0.848793565684
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  67   6   9]
 [  0   0   0 ...   3  90   7]
 [  0   0   0 ...   3   1  96]]
------ TEST ACCURACY:  weights.26-0.49.hdf5  ------
0.405882352941
[[1 4 0 ... 0 0 0]
 [1 2 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 4 0 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/4/weights.35-0.49.hdf5
------ TRAIN ACCURACY:  weights.35-0.49.hdf5  ------
0.872654155496
[[106   2   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  1   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  42  19  11]
 [  0   0   0 ...   1  98   4]
 [  0   0   0 ...   2   5  92]]
------ TEST ACCURACY:  weights.35-0.49.hdf5  ------
0.482352941176
[[5 0 0 ... 0 0 0]
 [4 1 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 3 0]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/5/weights.10-0.48.hdf5
------ TRAIN ACCURACY:  weights.10-0.48.hdf5  ------
0.649597855228
[[ 98   6   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0  95 ...   0   0   0]
 ...
 [  0   0   0 ...  33   6   4]
 [  0   0   0 ...   7  65   7]
 [  0   0   0 ...   6  10  40]]
------ TEST ACCURACY:  weights.10-0.48.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/6/weights.36-0.51.hdf5
------ TRAIN ACCURACY:  weights.36-0.51.hdf5  ------
0.872386058981
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   3]
 [  0   0   0 ...   8  79   7]
 [  0   0   0 ...  16   0  75]]
------ TEST ACCURACY:  weights.36-0.51.hdf5  ------
0.535294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/7/weights.28-0.50.hdf5
------ TRAIN ACCURACY:  weights.28-0.50.hdf5  ------
0.838605898123
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  73   4   9]
 [  0   0   0 ...   5  89   3]
 [  0   0   0 ...  11   3  82]]
------ TEST ACCURACY:  weights.28-0.50.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/8/weights.40-0.49.hdf5
------ TRAIN ACCURACY:  weights.40-0.49.hdf5  ------
0.895710455764
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  86   0   7]
 [  0   0   0 ...  10  79   6]
 [  0   0   0 ...   7   0  94]]
------ TEST ACCURACY:  weights.40-0.49.hdf5  ------
0.488235294118
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/9/weights.48-0.48.hdf5
------ TRAIN ACCURACY:  weights.48-0.48.hdf5  ------
0.911260053619
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  1   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  90   2   5]
 [  0   0   0 ...   4  90   4]
 [  0   0   0 ...   3   0  96]]
------ TEST ACCURACY:  weights.48-0.48.hdf5  ------
0.552941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 2 1 1]
 [0 0 0 ... 1 1 2]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/10/weights.27-0.49.hdf5
------ TRAIN ACCURACY:  weights.27-0.49.hdf5  ------
0.843699731903
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ...  67   3   7]
 [  0   0   0 ...   4  80  10]
 [  0   0   0 ...   5   1  82]]
------ TEST ACCURACY:  weights.27-0.49.hdf5  ------
0.423529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/11/weights.38-0.51.hdf5
------ TRAIN ACCURACY:  weights.38-0.51.hdf5  ------
0.89436997319
[[105   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  83   4   6]
 [  0   0   0 ...   7  95   3]
 [  0   0   0 ...   5   2  89]]
------ TEST ACCURACY:  weights.38-0.51.hdf5  ------
0.494117647059
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/12/weights.47-0.48.hdf5
------ TRAIN ACCURACY:  weights.47-0.48.hdf5  ------
0.911528150134
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  98   1   2]
 [  0   0   0 ...  11  81   4]
 [  0   0   0 ...   6   0  90]]
------ TEST ACCURACY:  weights.47-0.48.hdf5  ------
0.552941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/13/weights.43-0.51.hdf5
------ TRAIN ACCURACY:  weights.43-0.51.hdf5  ------
0.899731903485
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  89   1   7]
 [  0   0   0 ...   4  86   9]
 [  0   0   0 ...   2   0 100]]
------ TEST ACCURACY:  weights.43-0.51.hdf5  ------
0.488235294118
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 3]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/14/weights.25-0.48.hdf5
------ TRAIN ACCURACY:  weights.25-0.48.hdf5  ------
0.817694369973
[[105   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  70   7   8]
 [  0   0   0 ...   7  87   4]
 [  0   0   0 ...   8   7  76]]
------ TEST ACCURACY:  weights.25-0.48.hdf5  ------
0.429411764706
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 2 1 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/15/weights.13-0.48.hdf5
------ TRAIN ACCURACY:  weights.13-0.48.hdf5  ------
0.686479250335
[[100   6   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  1   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  61   2   4]
 [  0   0   0 ...  13  62   5]
 [  0   0   0 ...  17   7  41]]
------ TEST ACCURACY:  weights.13-0.48.hdf5  ------
0.521212121212
[[2 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/16/weights.47-0.51.hdf5
------ TRAIN ACCURACY:  weights.47-0.51.hdf5  ------
0.909115281501
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  84   3   7]
 [  0   0   0 ...   5  90   5]
 [  0   0   0 ...   3   0  98]]
------ TEST ACCURACY:  weights.47-0.51.hdf5  ------
0.576470588235
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 3 1 1]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/17/weights.22-0.49.hdf5
------ TRAIN ACCURACY:  weights.22-0.49.hdf5  ------
0.791208791209
[[103   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  78   3   8]
 [  0   0   0 ...  12  76   7]
 [  0   0   0 ...  12   2  87]]
------ TEST ACCURACY:  weights.22-0.49.hdf5  ------
0.497041420118
[[0 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 2]
 [0 0 0 ... 0 2 2]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/18/weights.34-0.50.hdf5
------ TRAIN ACCURACY:  weights.34-0.50.hdf5  ------
0.871045576408
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  87   2   4]
 [  0   0   0 ...   7  89   2]
 [  0   0   0 ...   7   1  90]]
------ TEST ACCURACY:  weights.34-0.50.hdf5  ------
0.635294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/19/weights.46-0.49.hdf5
------ TRAIN ACCURACY:  weights.46-0.49.hdf5  ------
0.902949061662
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  85   2   2]
 [  0   0   0 ...   5  86   3]
 [  0   0   0 ...   3   0  88]]
------ TEST ACCURACY:  weights.46-0.49.hdf5  ------
0.547058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 2 1 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/20/weights.31-0.47.hdf5
------ TRAIN ACCURACY:  weights.31-0.47.hdf5  ------
0.869309051955
[[106   2   0 ...   0   0   0]
 [  0 109   1 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  74   4  10]
 [  0   0   0 ...   6  84   6]
 [  0   0   0 ...   7   0  83]]
------ TEST ACCURACY:  weights.31-0.47.hdf5  ------
0.542168674699
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 4 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/21/weights.36-0.48.hdf5
------ TRAIN ACCURACY:  weights.36-0.48.hdf5  ------
0.867024128686
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  84   2   0]
 [  0   0   0 ...   6  89   0]
 [  0   0   0 ...  11   6  62]]
------ TEST ACCURACY:  weights.36-0.48.hdf5  ------
0.517647058824
[[1 0 0 ... 0 0 0]
 [3 2 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/22/weights.41-0.54.hdf5
------ TRAIN ACCURACY:  weights.41-0.54.hdf5  ------
0.899195710456
[[105   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  59   2   9]
 [  0   0   0 ...   2  92   2]
 [  0   0   0 ...   2   4  87]]
------ TEST ACCURACY:  weights.41-0.54.hdf5  ------
0.388235294118
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_conv1d_10/23/weights.45-0.46.hdf5
------ TRAIN ACCURACY:  weights.45-0.46.hdf5  ------
0.895710455764
[[103   2   0 ...   0   0   0]
 [  2 108   0 ...   0   0   0]
 [  1   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  96   1   0]
 [  0   0   0 ...   9  86   0]
 [  0   0   0 ...  14   4  67]]
------ TEST ACCURACY:  weights.45-0.46.hdf5  ------
0.588235294118
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 2 0 0]]
[0.5529411764705883, 0.3588235294117647, 0.40588235294117647, 0.4823529411764706, 0.4764705882352941, 0.5352941176470588, 0.5, 0.48823529411764705, 0.5529411764705883, 0.4235294117647059, 0.49411764705882355, 0.5529411764705883, 0.48823529411764705, 0.4294117647058823, 0.5212121212121212, 0.5764705882352941, 0.4970414201183432, 0.6352941176470588, 0.5470588235294118, 0.5421686746987951, 0.5176470588235295, 0.38823529411764707, 0.5882352941176471]
0.502371298395
0.065777100022

Process finished with exit code 0

'''
