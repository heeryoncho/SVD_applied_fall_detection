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

X_spca = pickle.load(open("data/X_umafall_spca.p", "rb"))
X_kpca = pickle.load(open("data/X_umafall_kpca.p", "rb"))
X = np.concatenate((X_spca, X_kpca), axis=1)

y = pickle.load(open("data/y_umafall_spca.p", "rb"))

n_classes = 11
signal_rows = 450 * 2
signal_columns = 1
n_subject = 17


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
    print X_train.shape # (2465, 900)

    # input layer
    input_signal = Input(shape=(signal_rows, 1))
    print K.int_shape(input_signal) # (None, 900, 1)

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

    new_dir = 'model/umafall_kpca_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_kpca_spca_conv1d_10/' + str(i+1) + '/'
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
[0.4473684210526316, 0.5866666666666667, 0.4540540540540541, 0.5157894736842106, 0.5142857142857142, 0.46, 0.7545454545454545, 0.4, 0.5888888888888889, 0.5428571428571428, 0.54, 0.51875, 0.49473684210526314, 0.5454545454545454, 0.5142857142857142, 0.5183333333333333, 0.48333333333333334]
0.522314681444
0.0749019969049
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_kpca_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/1/weights.18-0.53.hdf5
------ TRAIN ACCURACY:  weights.18-0.53.hdf5  ------
0.72738336714
[[236   0   0   2   0   0   1   0   0   1   0]
 [  5 101   0  17   0   0   0   0   1   1   0]
 [  1   0 198   0   2   1   3  17  24   3   1]
 [  0   0   0 218   0   0   0   0   0   2   0]
 [  2   0   0   1  87   1   0   0   7   2   0]
 [  2   0   0   4   2  86   0   0   6   0   0]
 [  0   0  13   0   0   0 128  68  30   0   1]
 [  2   0  52   0   0   0  17 154  18   2   0]
 [  3   0   8   0   3   2   4   6 236  17  11]
 [  1   1   7   0   4   0   6   6 111 169  15]
 [  3   0   6   3   1   3   2   0 102  35 180]]
------ TEST ACCURACY:  weights.18-0.53.hdf5  ------
0.447368421053
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  2  0 13  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  4  3  0  0]
 [ 0  0  0 13  0  2  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  5  0  0]
 [ 0  0  1  0  0  0  9  4  1  0  0]
 [ 2  0  5  0  3  0  0  0 14  2  4]
 [ 0  0  2  0  2  0  0  3 14 15  4]
 [ 2  0  0  0  0  0  0  0 14 10  4]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/2/weights.47-0.53.hdf5
------ TRAIN ACCURACY:  weights.47-0.53.hdf5  ------
0.852295409182
[[235   0   0   3   0   0   1   1   0   0   0]
 [  1 115   0   9   0   0   0   0   0   0   0]
 [  0   0 213   0   1   1   4  11  14   4   2]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  97   0   0   0   1   1   1]
 [  0   0   0   0   0  98   0   0   2   0   0]
 [  0   0   5   0   0   0 181  44   7   3   0]
 [  1   0  26   0   0   0  31 177   9   1   0]
 [  0   0   6   0   2   1   6   6 244  26  14]
 [  0   0   3   0   4   0   5  10  22 277   9]
 [  0   0   4   3   1   3   0   2  39  20 278]]
------ TEST ACCURACY:  weights.47-0.53.hdf5  ------
0.586666666667
[[15  0  0  0  0  0  0  0  0]
 [12  3  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  1  1  0  0]
 [ 0  2  0 10  0  0  0  0  3]
 [ 0  0  0  0  6  9  0  0  0]
 [ 0  0 10  0  0  5  0  0  0]
 [ 0  0  0  0  0  0  7  1  7]
 [ 0  0  0  0  0  0  3 18  9]
 [ 0  0  0  0  0  0  2  2 11]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/3/weights.23-0.54.hdf5
------ TRAIN ACCURACY:  weights.23-0.54.hdf5  ------
0.767206477733
[[237   0   0   0   0   0   3   0   0   0   0]
 [  7 115   0   3   0   0   0   0   0   0   0]
 [  0   0 211   0   3   0   5   3  19   2   2]
 [ 14   2   0 202   0   0   0   0   0   1   1]
 [  4   0   0   0  89   0   0   0   4   2   1]
 [ 10   0   0   0   3  74   0   0   9   2   2]
 [  0   0  26   0   0   0 179  25   8   2   0]
 [  2   0  73   0   0   0  47 112   9   1   1]
 [  2   0  12   0   2   1   8   5 229  18  13]
 [  5   0  16   0   4   0  14   1  47 228  15]
 [ 10   0  11   2   0   0   4   0  63  26 219]]
------ TEST ACCURACY:  weights.23-0.54.hdf5  ------
0.454054054054
[[15  0  0  0  0  0  0  0  0  0]
 [ 8  5  0  0  0  1  0  0  1  0]
 [ 0  0 15  0  0  0  4  1  0  0]
 [ 0  3  0  9  2  0  0  0  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 11  0  4  0  0]
 [ 0  0  0  0  0  6  9  0  0  0]
 [ 6  0  0  0  0  0  1  6  6 11]
 [ 1  0  2  0  0  0  0 12  3 12]
 [ 3  0  0  0  0  0  0  8  8 11]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/4/weights.45-0.54.hdf5
------ TRAIN ACCURACY:  weights.45-0.54.hdf5  ------
0.842596348884
[[232   0   0   0   0   0   2   0   0   0   1]
 [  0 118   0   7   0   0   0   0   0   0   0]
 [  0   0 223   0   0   0   4  11  10   0   2]
 [  0   2   0 218   0   0   0   0   0   0   0]
 [  1   0   0   0  94   0   0   0   2   1   2]
 [  0   0   0   0   0  97   0   0   2   0   1]
 [  0   0  20   0   0   0 167  37   4   2   0]
 [  1   0  36   0   0   0  20 184   4   0   0]
 [  0   0  11   0   1   1   6  11 223  10  27]
 [  0   0   9   0   3   0  10   9  30 238  36]
 [  0   0   7   1   0   3   4   3  32   2 283]]
------ TEST ACCURACY:  weights.45-0.54.hdf5  ------
0.515789473684
[[15  2  0  0  0  0  0  0  0  3]
 [ 7  4  0  1  0  0  0  0  0  3]
 [ 0  0 12  0  0  0  0  2  0  1]
 [13  0  0  2  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  3  0  0 15  6  1  0  0]
 [ 0  0  2  0  0  5  8  0  0  0]
 [ 3  0  2  0  2  0  1 12  0 10]
 [ 1  0  0  0  0  0  0  9  6  9]
 [ 0  1  0  0  0  0  0  5  0 24]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/5/weights.38-0.54.hdf5
------ TRAIN ACCURACY:  weights.38-0.54.hdf5  ------
0.859215686275
[[238   0   0   0   0   0   2   0   0   0   0]
 [  1 131   0   7   0   0   0   0   0   1   0]
 [  0   0 187   0   0   0   8  34  18   3   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  94   0   0   0   4   2   0]
 [  0   0   0   0   0  97   0   0   2   0   1]
 [  0   0   1   0   0   1 204  27   4   2   1]
 [  1   0  12   0   0   0  37 190   5   0   0]
 [  0   0  11   0   2   0   6   3 248  25  25]
 [  0   0   5   0   4   0  10   7  20 302  12]
 [  3   0   0   2   0   3   0   1  30  16 280]]
------ TEST ACCURACY:  weights.38-0.54.hdf5  ------
0.514285714286
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  1  0  0  0  1]
 [ 0  5  0 10  0  0  0  0  0]
 [ 0  0  0  0  8  2  5  0  0]
 [ 0  0  1  0  9  5  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  7  0  3  2  9  6  3]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/6/weights.49-0.51.hdf5
------ TRAIN ACCURACY:  weights.49-0.51.hdf5  ------
0.851439539347
[[247   0   0   0   0   0   2   0   0   1   0]
 [  0 128   0  10   0   0   1   0   0   0   1]
 [  0   0 221   0   0   0   3  21   8   5   2]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   0   0  96   0   0   0   1   1   2]
 [  0   0   0   0   1  97   0   0   0   0   2]
 [  0   0   6   0   0   0 171  70   3   4   1]
 [  0   0  30   0   0   0  13 205   2   0   0]
 [  0   0   8   0   2   0   8   7 205  28  52]
 [  0   0   3   0   2   0  11   6   9 285  34]
 [  1   0   3   2   0   1   0   0  12   8 328]]
------ TEST ACCURACY:  weights.49-0.51.hdf5  ------
0.46
[[3 0 2 0 0 0 0 0]
 [0 4 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [2 0 0 2 4 1 0 1]
 [0 0 1 0 0 5 1 3]
 [0 2 1 0 0 3 1 3]
 [0 1 0 0 0 3 0 6]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/7/weights.26-0.52.hdf5
------ TRAIN ACCURACY:  weights.26-0.52.hdf5  ------
0.781139489194
[[234   0   0   2   0   0   2   0   0   1   1]
 [  0 127   0  11   0   0   2   0   0   0   0]
 [  0   0 189   0   2   0   8  20  20  10   1]
 [  0   1   0 218   0   1   0   0   0   0   0]
 [  0   0   0   0  80   0   0   0   3   2   0]
 [  1   0   0   1   1  77   0   0   4   1   0]
 [  0   0   4   0   2   0 178  35  11   5   0]
 [  0   0  42   0   0   0  51 131  14   5   2]
 [  0   0   7   0   3   0   6   5 254  37   8]
 [  0   0   2   0   6   0  11   4  38 295   4]
 [  1   1   4   3   1   1   3   0  90  56 205]]
------ TEST ACCURACY:  weights.26-0.52.hdf5  ------
0.754545454545
[[14  0  0  0  0  0  0  1  0  0]
 [ 0 11  0  0  0  0  1  3  0  0]
 [ 0  0 10  0  1  0  0  1  3  0]
 [ 3  0  0 12  0  0  0  0  0  0]
 [ 1  0  0  1  7  0  0  3  0  3]
 [ 0  1  0  0  0 17  0  1  1  0]
 [ 0  0  0  0  0  3 12  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/8/weights.32-0.53.hdf5
------ TRAIN ACCURACY:  weights.32-0.53.hdf5  ------
0.808203125
[[238   0   0   0   0   0   2   0   0   0   0]
 [  1 137   0   1   0   0   1   0   0   0   0]
 [  0   0 208   0   4   1  10   7  16   2   2]
 [  4  10   0 206   0   0   0   0   0   0   0]
 [  0   0   0   0  89   0   0   0   0   1   0]
 [  1   0   0   0   0  88   0   0   1   0   0]
 [  0   0   3   0   1   1 211  18   5   1   0]
 [  2   0  38   0   1   0  70 126   5   2   1]
 [  2   0   6   0   4   2  15   1 227  16  47]
 [  3   2   5   0   5   0  21   4  37 250  33]
 [  3   1   5   1   4   6   4   0  35  17 289]]
------ TEST ACCURACY:  weights.32-0.53.hdf5  ------
0.4
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  7  0  1  0  4  1  2  0]
 [ 2 11  0  2  0  0  0  0  0  0]
 [ 0  0  0  0  2  5  0  0  1  2]
 [ 0  0  0  0  0  4  0  0  2  4]
 [ 0  0  1  0  0  0  6  8  0  0]
 [ 0  0 10  0  0  0  2  2  1  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/9/weights.36-0.50.hdf5
------ TRAIN ACCURACY:  weights.36-0.50.hdf5  ------
0.792323232323
[[237   0   0   0   0   0   2   0   0   1   0]
 [  0 118   0   5   0   0   1   0   0   1   0]
 [  0   0 214   0   2   0   1  21   6   5   1]
 [  2   1   0 217   0   0   0   0   0   0   0]
 [  0   0   1   0  95   0   0   0   1   2   1]
 [  3   0   0   0   1  87   0   0   2   1   6]
 [  0   0  21   0   1   0  87 120   6   5   0]
 [  2   0  48   0   0   0   8 184   3   0   0]
 [  0   0  18   0   5   0   2  10 203  25  27]
 [  1   0  10   0   3   0   3  12  22 262  17]
 [  3   0  14   1   1   1   1   3  38  16 257]]
------ TEST ACCURACY:  weights.36-0.50.hdf5  ------
0.588888888889
[[14  0  0  0  0  0  0  0  1  0  0]
 [ 0 15  0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  0  1  0  0]
 [ 0 11  0  4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0  9  4  1  0  0]
 [ 0  0  0  0  0  0  1 14  0  0  0]
 [ 0  0  4  0  4  0  1  0  9  4  8]
 [ 0  1  0  0  0  0  1  0  4 12 12]
 [ 0  3  0  0  1  1  0  0  6  4 15]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/10/weights.45-0.51.hdf5
------ TRAIN ACCURACY:  weights.45-0.51.hdf5  ------
0.836862745098
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 131   0   8   0   0   1   0   0   0   0]
 [  0   0 219   0   0   0   3  16  11   0   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  83   0   0   0   1   0   1]
 [  0   0   0   0   1  83   0   0   0   0   1]
 [  0   0   7   0   0   0 177  50   4   1   1]
 [  1   0  31   0   0   0  24 187   2   0   0]
 [  0   0  15   0   3   1   8  14 251  11  17]
 [  0   0  10   0   3   0  13   7  47 266  14]
 [  0   0  10   1   2   2   4   0  57  10 279]]
------ TEST ACCURACY:  weights.45-0.51.hdf5  ------
0.542857142857
[[13  0  1  0  1  0  0  0  0  0]
 [ 0  2  0  2  0  0  0  8  1  2]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 1  0  0  4  7  0  0  1  0  2]
 [ 0  0  0  0  0  5 10  0  0  0]
 [ 0  9  0  0  0  0  0  5  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/11/weights.10-0.53.hdf5
------ TRAIN ACCURACY:  weights.10-0.53.hdf5  ------
0.653228962818
[[232   1   0   3   0   0   2   1   0   1   0]
 [  7 116   0  14   0   0   2   0   0   1   0]
 [  1   0 157   0   1   1   1  47  25   5   7]
 [ 11  19   0 187   0   0   0   0   0   1   2]
 [ 11   0   1   2  51   8   0   0   7   1   9]
 [  5   0   0   8   0  67   0   0   4   0   6]
 [  0   0  12   0   0   0 115  90  18   5   0]
 [  2   0  36   0   0   0  14 180   9   3   1]
 [  4   1  12   0   0   1  11  12 217  24  38]
 [ 11   4  10   0   0   0  14  21  92 158  50]
 [  5   2  11   4   1   3   3   8 110  29 189]]
------ TEST ACCURACY:  weights.10-0.53.hdf5  ------
0.54
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  1  1  3  1  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 4  0  0  3  1  0  0  0  0  2]
 [ 2  0  0  1  7  0  0  0  0  0]
 [ 0  3  0  0  0  0 12  0  0  0]
 [ 0 12  0  0  0  0  0  3  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/12/weights.46-0.54.hdf5
------ TRAIN ACCURACY:  weights.46-0.54.hdf5  ------
0.83006012024
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 124   0  15   0   0   1   0   0   0   0]
 [  0   0 222   0   0   0   0  11   9   1   2]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   1   0  82   0   0   0   1   0   1]
 [  0   0   0   0   0  78   0   0   1   0   1]
 [  0   0  31   0   0   0 156  48   3   1   1]
 [  1   0  59   0   0   0  14 167   4   0   0]
 [  0   0  16   0   1   1   3   6 235  11  32]
 [  0   0  14   0   4   0   6   7  36 239  39]
 [  1   0  11   1   0   1   0   1  20   5 310]]
------ TEST ACCURACY:  weights.46-0.54.hdf5  ------
0.51875
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 16  0  0  0  0  0  1  0  3]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 9  0  2  0  4  0  0  0  0  0]
 [ 8  0  8  0  1  0  0  1  0  2]
 [ 0  3  0  0  0 11  1  0  0  0]
 [ 0  3  0  0  0  2 10  0  0  0]
 [ 0  0  0  0  0  0  0  4  3  8]
 [ 0  5  0  0  0  0  0  4  2  4]
 [ 0  0  0  0  0  0  0  5  1  9]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/13/weights.19-0.58.hdf5
------ TRAIN ACCURACY:  weights.19-0.58.hdf5  ------
0.701171875
[[243   0   0   2   0   0   0   0   0   0   0]
 [  1 128   0   4   0   0   0   0   1   1   0]
 [  0   0 204   0   0   0  10  10  32   2   2]
 [  5  15   0 207   0   0   0   0   0   0   3]
 [  6   1   0   0  80   1   0   0  12   0   0]
 [ 12   0   0   1   1  75   0   0   9   0   2]
 [  0   0  22   0   0   0 180  21  25   2   0]
 [  1   0  67   0   0   0  67 104  15   1   0]
 [  0   0  14   0   3   1  11   0 248   2  21]
 [  6   4  14   0   5   0  19   2 119 125  46]
 [  5   2  11   3   0   2   3   0 114   4 201]]
------ TEST ACCURACY:  weights.19-0.58.hdf5  ------
0.494736842105
[[ 5  0  0  0  2  1  2  0  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  1  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  1  0  2  0  2  0  0]
 [ 0  0  0  0  2  3  0  0  0]
 [ 0  0  1  0  0  0 16  0  3]
 [ 0  3  0  0  0  0  4  1 12]
 [ 0  2  0  0  0  0 12  0  6]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/14/weights.49-0.56.hdf5
------ TRAIN ACCURACY:  weights.49-0.56.hdf5  ------
0.859615384615
[[246   0   0   2   0   0   2   0   0   0   0]
 [  0 127   0  12   0   0   0   0   1   0   0]
 [  0   0 216   0   1   0   3  14  18   2   1]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   0   0  97   0   0   0   2   0   1]
 [  0   0   0   0   0  99   0   0   1   0   0]
 [  0   0   1   0   0   0 215  31   8   0   0]
 [  1   0  26   0   0   0  39 177   7   0   0]
 [  1   1   5   0   2   0   8   7 257  16  18]
 [  0   0   0   0   3   0  11   7  39 268  17]
 [  1   0   2   1   0   2   1   2  42   6 298]]
------ TEST ACCURACY:  weights.49-0.56.hdf5  ------
0.545454545455
[[5 0 0 0 0 0 0]
 [0 3 1 5 1 0 0]
 [0 0 0 0 0 0 0]
 [0 0 2 7 1 0 0]
 [0 1 0 1 3 0 0]
 [0 0 2 1 1 7 4]
 [0 0 1 0 3 1 5]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/15/weights.49-0.49.hdf5
------ TRAIN ACCURACY:  weights.49-0.49.hdf5  ------
0.843529411765
[[243   0   0   0   0   0   2   0   0   0   0]
 [  0 129   0   6   0   0   0   0   0   0   0]
 [  0   0 234   0   1   0   4  11   6   0   4]
 [  0   1   0 224   0   0   0   0   0   0   0]
 [  0   0   0   0  97   0   0   0   1   0   2]
 [  0   0   0   0   0  98   0   0   0   0   2]
 [  0   0   7   0   0   0 213  17   4   0   4]
 [  1   0  36   0   0   0  56 151   5   0   1]
 [  0   0  12   0   4   1  11   3 207  11  51]
 [  0   0   7   0   4   0  11   7  23 233  60]
 [  1   0   2   1   1   3   1   0  14   0 322]]
------ TEST ACCURACY:  weights.49-0.49.hdf5  ------
0.514285714286
[[ 6  0  0  0  3  0  0  1  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  1  0  0  4  0  0  0  0]
 [ 0  3  0  7  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  7  2  0  0  0]
 [ 0  0  0  0  0  2  8  0  0  0]
 [ 0  0  0  0  0  0  2  7  2  9]
 [ 1  0  0  0  0  0  0  4  5  5]
 [ 0  0  0  0  0  0  0  5  2 13]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/16/weights.31-0.50.hdf5
------ TRAIN ACCURACY:  weights.31-0.50.hdf5  ------
0.801459854015
[[212   0   0   0   0   0   3   0   0   0   0]
 [  1  93   0   5   0   0   1   0   0   0   0]
 [  0   0 193   0   0   0   4   6  11   5   1]
 [  2   2   0 190   0   0   0   0   0   1   0]
 [  2   0   0   0  61   0   0   0   2   0   0]
 [  1   0   0   0   0  64   0   0   2   3   0]
 [  0   0  18   0   0   0 172  11   6   3   0]
 [  3   0  60   0   0   0  49  96   7   0   0]
 [  2   0  13   0   0   0   4   1 186  26   3]
 [  1   0   5   0   1   0   9   2  17 223   2]
 [  3   0   9   2   1   1   3   0  48  46 157]]
------ TEST ACCURACY:  weights.31-0.50.hdf5  ------
0.518333333333
[[40  0  0  0  0  0  0  0  0  0  0]
 [12 27  0  1  0  0  0  0  0  0  0]
 [ 0  0 41  0  0  0  1  1  0  2  0]
 [ 4  0  0 35  1  0  0  0  0  0  0]
 [ 0  1  9  0 13  0  0  0  8  4  0]
 [ 1  0  0  0  5  8  0  0 10  5  1]
 [ 0  0  2  0  0  0 31 12  0  0  0]
 [ 0  0 17  0  0  0  6 22  0  0  0]
 [ 1  2  5  0  1  0 11  1 42 20  2]
 [ 5  1  5  0  0  0 11  4 22 46  6]
 [ 1  1  2  1  0  0  3  0 18 63  6]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_spca_conv1d_10/17/weights.28-0.42.hdf5
------ TRAIN ACCURACY:  weights.28-0.42.hdf5  ------
0.784242424242
[[237   0   0   0   0   0   3   0   0   0   0]
 [  0 122   0   1   0   0   2   0   0   0   0]
 [  0   0 215   0   0   0   3  14  13   3   2]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  88   0   0   1   8   2   1]
 [  1   0   0   0   1  86   0   0   8   3   1]
 [  0   0  16   0   0   0 165  52   6   1   0]
 [  1   0  46   0   0   0  27 163   8   0   0]
 [  0   0   6   0   2   0   6   7 222  28  19]
 [  0   2   8   0   0   0  17   6  49 232  16]
 [  1   0  15   1   0   0   5   1  54  67 191]]
------ TEST ACCURACY:  weights.28-0.42.hdf5  ------
0.483333333333
[[12  0  0  2  0  0  0  0  1  0  0]
 [ 0  6  0  9  0  0  0  0  0  0  0]
 [ 0  0  5  0  0  0  1  0  8  1  0]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0  9  3  0  2  0]
 [ 0  0  3  0  0  0  3  9  0  0  0]
 [ 0  0  4  0  0  0  1  2 17  5  1]
 [ 0  0  4  0  2  0  5  4  4  9  2]
 [ 0  0  0  2  0  3  0  0 11  9  5]]
[0.4473684210526316, 0.5866666666666667, 0.4540540540540541, 0.5157894736842106, 0.5142857142857142, 0.46, 0.7545454545454545, 0.4, 0.5888888888888889, 0.5428571428571428, 0.54, 0.51875, 0.49473684210526314, 0.5454545454545454, 0.5142857142857142, 0.5183333333333333, 0.48333333333333334]
0.522314681444
0.0749019969049

Process finished with exit code 0

'''