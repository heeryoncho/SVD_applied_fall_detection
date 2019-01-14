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
X_spca = pickle.load(open("data/X_sisfall_spca.p", "rb"))
X = np.concatenate((X_svd, X_spca), axis=1)

y = pickle.load(open("data/y_sisfall_svd.p", "rb"))

n_classes = 34
signal_rows = 900
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

    new_dir = 'model/sisfall_svd_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_svd_spca_conv1d_10/' + str(i+1) + '/'
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
[0.5705882352941176, 0.36470588235294116, 0.37058823529411766, 0.47058823529411764, 0.45294117647058824, 0.5176470588235295, 0.5352941176470588, 0.5176470588235295, 0.5411764705882353, 0.4294117647058823, 0.47058823529411764, 0.5647058823529412, 0.4294117647058823, 0.4764705882352941, 0.5636363636363636, 0.5117647058823529, 0.514792899408284, 0.5235294117647059, 0.5588235294117647, 0.4819277108433735, 0.5470588235294118, 0.4, 0.5352941176470588]
0.493417055131
0.0602386223167
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_svd_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/1/weights.19-0.47.hdf5
------ TRAIN ACCURACY:  weights.19-0.47.hdf5  ------
0.783914209115
[[107   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  91 ...   0   0   0]
 ...
 [  0   0   0 ...  81   3   4]
 [  0   0   0 ...  11  78   4]
 [  0   0   0 ...  17   7  55]]
------ TEST ACCURACY:  weights.19-0.47.hdf5  ------
0.570588235294
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/2/weights.15-0.49.hdf5
------ TRAIN ACCURACY:  weights.15-0.49.hdf5  ------
0.73726541555
[[104   1   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0  99 ...   0   0   0]
 ...
 [  0   0   0 ...  31   6  31]
 [  0   0   0 ...   3  68  18]
 [  0   0   0 ...   1   4  87]]
------ TEST ACCURACY:  weights.15-0.49.hdf5  ------
0.364705882353
[[4 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/3/weights.33-0.48.hdf5
------ TRAIN ACCURACY:  weights.33-0.48.hdf5  ------
0.854423592493
[[107   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  95   2   0]
 [  0   0   0 ...  11  87   0]
 [  0   0   0 ...  24   5  61]]
------ TEST ACCURACY:  weights.33-0.48.hdf5  ------
0.370588235294
[[0 5 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 5 0 0]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/4/weights.46-0.49.hdf5
------ TRAIN ACCURACY:  weights.46-0.49.hdf5  ------
0.91018766756
[[106   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  93   0   0]
 [  0   0   0 ...  10  78   4]
 [  0   0   0 ...  10   0  81]]
------ TEST ACCURACY:  weights.46-0.49.hdf5  ------
0.470588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/5/weights.24-0.47.hdf5
------ TRAIN ACCURACY:  weights.24-0.47.hdf5  ------
0.822520107239
[[105   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  64   4  13]
 [  0   0   0 ...   6  76   6]
 [  0   0   0 ...   8   3  80]]
------ TEST ACCURACY:  weights.24-0.47.hdf5  ------
0.452941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/6/weights.45-0.48.hdf5
------ TRAIN ACCURACY:  weights.45-0.48.hdf5  ------
0.902680965147
[[106   2   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  85   3  10]
 [  0   0   0 ...   5  92   3]
 [  0   0   0 ...   3   1  99]]
------ TEST ACCURACY:  weights.45-0.48.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/7/weights.30-0.49.hdf5
------ TRAIN ACCURACY:  weights.30-0.49.hdf5  ------
0.86327077748
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  74   7   6]
 [  0   0   0 ...   4  95   3]
 [  0   0   0 ...   4   9  86]]
------ TEST ACCURACY:  weights.30-0.49.hdf5  ------
0.535294117647
[[5 0 0 ... 0 0 0]
 [1 3 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 2 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/8/weights.29-0.47.hdf5
------ TRAIN ACCURACY:  weights.29-0.47.hdf5  ------
0.841018766756
[[107   0   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ...  74   1   2]
 [  0   0   0 ...   7  75   7]
 [  0   0   0 ...  11   1  69]]
------ TEST ACCURACY:  weights.29-0.47.hdf5  ------
0.517647058824
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/9/weights.25-0.49.hdf5
------ TRAIN ACCURACY:  weights.25-0.49.hdf5  ------
0.835388739946
[[106   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  56   5   9]
 [  0   0   0 ...   2  84   2]
 [  0   0   0 ...   5   5  78]]
------ TEST ACCURACY:  weights.25-0.49.hdf5  ------
0.541176470588
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/10/weights.43-0.49.hdf5
------ TRAIN ACCURACY:  weights.43-0.49.hdf5  ------
0.9
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  58   0  15]
 [  0   0   0 ...   2  88   7]
 [  0   0   0 ...   1   0  97]]
------ TEST ACCURACY:  weights.43-0.49.hdf5  ------
0.429411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/11/weights.30-0.49.hdf5
------ TRAIN ACCURACY:  weights.30-0.49.hdf5  ------
0.852815013405
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  71   3   4]
 [  0   0   0 ...   7  89   0]
 [  0   0   0 ...  10   6  66]]
------ TEST ACCURACY:  weights.30-0.49.hdf5  ------
0.470588235294
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/12/weights.33-0.47.hdf5
------ TRAIN ACCURACY:  weights.33-0.47.hdf5  ------
0.870777479893
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  90   3   1]
 [  0   0   0 ...  11  88   1]
 [  0   0   0 ...  11   7  68]]
------ TEST ACCURACY:  weights.33-0.47.hdf5  ------
0.564705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/13/weights.43-0.49.hdf5
------ TRAIN ACCURACY:  weights.43-0.49.hdf5  ------
0.903753351206
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   1]
 [  0   0   0 ...  14  83   1]
 [  0   0   0 ...  12   0  85]]
------ TEST ACCURACY:  weights.43-0.49.hdf5  ------
0.429411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/14/weights.34-0.47.hdf5
------ TRAIN ACCURACY:  weights.34-0.47.hdf5  ------
0.869436997319
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  75   8   8]
 [  0   0   0 ...   2  97   4]
 [  0   0   0 ...   3  13  81]]
------ TEST ACCURACY:  weights.34-0.47.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/15/weights.50-0.48.hdf5
------ TRAIN ACCURACY:  weights.50-0.48.hdf5  ------
0.900401606426
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  73   1   4]
 [  0   0   0 ...   4  83   4]
 [  0   0   0 ...   3   0  79]]
------ TEST ACCURACY:  weights.50-0.48.hdf5  ------
0.563636363636
[[3 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/16/weights.29-0.49.hdf5
------ TRAIN ACCURACY:  weights.29-0.49.hdf5  ------
0.853887399464
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  53  10  11]
 [  0   0   0 ...   1  91   4]
 [  0   0   1 ...   3   3  84]]
------ TEST ACCURACY:  weights.29-0.49.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 1 3 1]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/17/weights.34-0.48.hdf5
------ TRAIN ACCURACY:  weights.34-0.48.hdf5  ------
0.869740016081
[[107   0   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  93   2   0]
 [  0   0   0 ...  12  88   0]
 [  0   0   0 ...  19   2  56]]
------ TEST ACCURACY:  weights.34-0.48.hdf5  ------
0.514792899408
[[0 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/18/weights.19-0.49.hdf5
------ TRAIN ACCURACY:  weights.19-0.49.hdf5  ------
0.780428954424
[[106   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  48  10  20]
 [  0   0   0 ...   4  83   9]
 [  0   0   0 ...   3   8  82]]
------ TEST ACCURACY:  weights.19-0.49.hdf5  ------
0.523529411765
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 3 2]
 [0 0 0 ... 0 1 3]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/19/weights.42-0.47.hdf5
------ TRAIN ACCURACY:  weights.42-0.47.hdf5  ------
0.893297587131
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  89   2   4]
 [  0   0   0 ...   8  84   7]
 [  0   0   0 ...   7   0  90]]
------ TEST ACCURACY:  weights.42-0.47.hdf5  ------
0.558823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 2 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/20/weights.31-0.48.hdf5
------ TRAIN ACCURACY:  weights.31-0.48.hdf5  ------
0.843063738618
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  68  11  10]
 [  0   0   0 ...   5  94   3]
 [  0   0   0 ...   4  12  84]]
------ TEST ACCURACY:  weights.31-0.48.hdf5  ------
0.481927710843
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 5 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/21/weights.25-0.49.hdf5
------ TRAIN ACCURACY:  weights.25-0.49.hdf5  ------
0.834852546917
[[103   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  55   4  17]
 [  0   0   0 ...   4  87   6]
 [  0   0   0 ...   2   6  84]]
------ TEST ACCURACY:  weights.25-0.49.hdf5  ------
0.547058823529
[[1 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/22/weights.39-0.54.hdf5
------ TRAIN ACCURACY:  weights.39-0.54.hdf5  ------
0.899731903485
[[106   0   0 ...   0   0   0]
 [  2 107   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  62   2   7]
 [  0   0   0 ...   1  89   3]
 [  0   0   0 ...   5   5  84]]
------ TEST ACCURACY:  weights.39-0.54.hdf5  ------
0.4
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 3]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_spca_conv1d_10/23/weights.42-0.48.hdf5
------ TRAIN ACCURACY:  weights.42-0.48.hdf5  ------
0.906166219839
[[105   0   0 ...   0   0   0]
 [  2 108   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  75   2  12]
 [  0   0   0 ...   3  92   5]
 [  0   0   0 ...   3   4  93]]
------ TEST ACCURACY:  weights.42-0.48.hdf5  ------
0.535294117647
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 0 3]]
[0.5705882352941176, 0.36470588235294116, 0.37058823529411766, 0.47058823529411764, 0.45294117647058824, 0.5176470588235295, 0.5352941176470588, 0.5176470588235295, 0.5411764705882353, 0.4294117647058823, 0.47058823529411764, 0.5647058823529412, 0.4294117647058823, 0.4764705882352941, 0.5636363636363636, 0.5117647058823529, 0.514792899408284, 0.5235294117647059, 0.5588235294117647, 0.4819277108433735, 0.5470588235294118, 0.4, 0.5352941176470588]
0.493417055131
0.0602386223167

Process finished with exit code 0

'''