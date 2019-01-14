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

X_svd = pickle.load(open("data/X_umafall_svd.p", "rb"))
X_spca = pickle.load(open("data/X_umafall_spca.p", "rb"))
X = np.concatenate((X_svd, X_spca), axis=1)

y = pickle.load(open("data/y_umafall_svd.p", "rb"))

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

    new_dir = 'model/umafall_svd_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_svd_spca_conv1d_10/' + str(i+1) + '/'
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
[0.5368421052631579, 0.46, 0.5027027027027027, 0.6157894736842106, 0.580952380952381, 0.34, 0.7545454545454545, 0.5789473684210527, 0.5666666666666667, 0.4857142857142857, 0.58, 0.4625, 0.6, 0.5272727272727272, 0.5714285714285714, 0.49666666666666665, 0.4888888888888889]
0.538171605424
0.0851352496799
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_svd_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/1/weights.47-0.46.hdf5
------ TRAIN ACCURACY:  weights.47-0.46.hdf5  ------
0.802839756592
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 115   0  10   0   0   0   0   0   0   0]
 [  0   0 196   0   2   2   1  31  12   5   1]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  0   0   0   0  97   1   0   0   2   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0  12   0   1   0 129  85   6   5   2]
 [  0   0  25   0   0   0  11 201   5   2   1]
 [  0   0  13   0   2   2   4   9 194  31  35]
 [  0   0   8   0   0   2   4  11  30 243  22]
 [  0   0  11   1   5   5   1   3  26  36 247]]
------ TEST ACCURACY:  weights.47-0.46.hdf5  ------
0.536842105263
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  7  0  8  0  0  0  0  0  0  0]
 [ 0  0  6  0  0  0  0  7  2  0  0]
 [ 0  0  0 12  0  3  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  0 13  0  0  0  0]
 [ 0  0  0  0  0  0  6  9  0  0  0]
 [ 1  0  2  0  3  1  0  1 13  1  8]
 [ 1  0  1  0  1  1  0  3 14 16  3]
 [ 1  0  3  0  0  1  0  0  8  6 11]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/2/weights.24-0.54.hdf5
------ TRAIN ACCURACY:  weights.24-0.54.hdf5  ------
0.732135728543
[[237   0   0   0   0   0   3   0   0   0   0]
 [  2 117   0   6   0   0   0   0   0   0   0]
 [  0   0 189   0   3   0   4  16  28   2   8]
 [  5   2   0 208   2   0   0   0   0   0   3]
 [  0   0   2   0  86   1   1   0   8   1   1]
 [  0   0   0   2   4  81   0   0   2   1  10]
 [  0   0  29   0   1   0 155  46   9   0   0]
 [  0   0  44   0   0   0  41 143  14   2   1]
 [  1   0  11   0   3   1   9  11 208  13  48]
 [  0   0  23   0   0   0  10  15  58 137  87]
 [  2   0  20   1   2   1   5   1  38   7 273]]
------ TEST ACCURACY:  weights.24-0.54.hdf5  ------
0.46
[[15  0  0  0  0  0  0  0  0]
 [12  3  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  1  0  0]
 [ 0  1  0  7  0  0  0  0  7]
 [ 0  0  1  0  1 13  0  0  0]
 [ 0  0  9  0  0  5  1  0  0]
 [ 0  0  0  0  0  0  6  0  9]
 [ 0  0  1  0  0  1  4  5 19]
 [ 0  0  0  0  0  0  2  0 13]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/3/weights.31-0.53.hdf5
------ TRAIN ACCURACY:  weights.31-0.53.hdf5  ------
0.780971659919
[[237   0   0   0   0   0   3   0   0   0   0]
 [  0 123   0   2   0   0   0   0   0   0   0]
 [  0   0 181   0   3   1   8  33  12   2   5]
 [  1   1   0 217   0   0   0   0   0   0   1]
 [  1   0   0   0  92   1   0   0   3   2   1]
 [  0   0   1   0   0  95   0   0   0   0   4]
 [  0   0   8   0   0   0 166  63   2   0   1]
 [  0   0  29   0   0   0  30 178   7   0   1]
 [  0   0   9   0   2   1  12  12 193  14  47]
 [  0   0  19   0   0   1  20  11  32 169  78]
 [  4   0  13   2   4   2   7   5  19   1 278]]
------ TEST ACCURACY:  weights.31-0.53.hdf5  ------
0.502702702703
[[15  0  0  0  0  0  0  0  0]
 [ 4 11  0  0  0  0  0  0  0]
 [ 0  0  8  0  0 11  0  0  1]
 [ 0  3  0 12  0  0  0  0  0]
 [ 0  0  0  0 12  1  2  0  0]
 [ 0  0  0  0  8  7  0  0  0]
 [ 2  0  1  0  1  1  1  1 23]
 [ 2  0  0  0  2  0  4  5 17]
 [ 0  0  0  0  0  0  4  4 22]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/4/weights.48-0.51.hdf5
------ TRAIN ACCURACY:  weights.48-0.51.hdf5  ------
0.829614604462
[[232   0   0   0   0   0   2   0   1   0   0]
 [  0 123   0   2   0   0   0   0   0   0   0]
 [  0   0 211   0   1   0   5   3  20   9   1]
 [  0   1   0 218   0   0   0   0   0   1   0]
 [  0   0   0   0  95   0   0   0   3   2   0]
 [  0   0   0   0   0  97   0   0   2   1   0]
 [  0   0   4   0   0   0 190  24   4   7   1]
 [  0   0  47   0   0   0  44 131  10  11   2]
 [  0   0   4   0   0   1   5   5 220  30  25]
 [  0   0   3   0   0   0   7   4  33 273  15]
 [  0   0   4   1   2   1   2   1  31  38 255]]
------ TEST ACCURACY:  weights.48-0.51.hdf5  ------
0.615789473684
[[18  2  0  0  0  0  0  0  0  0  0]
 [ 2 12  0  1  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  0  0  1  1  1]
 [11  0  0  4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0 19  2  2  1  0]
 [ 0  0  2  0  0  0  5  8  0  0  0]
 [ 3  0  1  0  3  0  0  0 12  2  9]
 [ 1  0  0  0  0  0  0  1  3 15  5]
 [ 0  0  0  1  0  1  0  0  2  9 17]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/5/weights.47-0.55.hdf5
------ TRAIN ACCURACY:  weights.47-0.55.hdf5  ------
0.85568627451
[[237   0   0   0   0   0   2   0   0   0   1]
 [  0 137   0   3   0   0   0   0   0   0   0]
 [  1   0 211   0   1   1   5  18   9   3   1]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  0   0   0   0  96   0   0   1   2   0   1]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   3   0   0   0 199  33   3   2   0]
 [  0   0  28   0   0   0  29 178   7   2   1]
 [  1   0  10   0   0   1   7   9 227  21  44]
 [  1   0  10   0   0   0  12   2  22 280  33]
 [  0   0   2   1   2   1   2   0  17  12 298]]
------ TEST ACCURACY:  weights.47-0.55.hdf5  ------
0.580952380952
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  0  0  1]
 [ 0  4  0 11  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  1 10  1  1  1  0]
 [ 0  0  1  0  0  7  7  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  9  0  0  5  2  6  4  4]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/6/weights.13-0.51.hdf5
------ TRAIN ACCURACY:  weights.13-0.51.hdf5  ------
0.654126679463
[[246   0   0   0   0   0   4   0   0   0   0]
 [  0 127   0  10   0   1   2   0   0   0   0]
 [  0   0 147   0   3   1  24  30  43   7   5]
 [ 20  22   0 185   0   5   0   0   1   0   2]
 [  7   0   0   0  77   4   1   0   9   1   1]
 [  2   0   0   2   5  80   0   0   2   4   5]
 [  0   0   7   0   1   0 186  48  13   0   0]
 [  0   0  26   0   0   0  79 125  19   1   0]
 [  2   1   4   0   1   1  21   6 244  10  20]
 [  4   5  14   0   0   0  34   9 131 106  47]
 [  2   5   7   4   1   4  13   2 120  16 181]]
------ TEST ACCURACY:  weights.13-0.51.hdf5  ------
0.34
[[5 0 0 0 0 0 0 0 0 0]
 [0 3 0 0 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [2 0 0 0 0 5 3 0 0 0]
 [2 0 1 0 0 1 0 4 0 2]
 [0 0 0 0 0 1 0 7 2 0]
 [0 0 0 1 0 1 0 7 1 0]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/7/weights.43-0.52.hdf5
------ TRAIN ACCURACY:  weights.43-0.52.hdf5  ------
0.807465618861
[[238   0   0   0   0   0   0   2   0   0   0]
 [  0 128   0  12   0   0   0   0   0   0   0]
 [  0   0 186   0   2   0   0  24  26   7   5]
 [  0   0   0 219   0   1   0   0   0   0   0]
 [  1   0   0   0  81   0   0   0   2   1   0]
 [  0   0   0   0   1  73   0   0   2   0   9]
 [  0   0  14   0   2   0 140  61  12   6   0]
 [  0   0  33   0   0   0  13 180  14   3   2]
 [  0   0   6   0   2   0   4   6 258  19  25]
 [  0   0   7   0   0   0   3   8  54 262  26]
 [  0   0   7   1   3   0   0   1  50  13 290]]
------ TEST ACCURACY:  weights.43-0.52.hdf5  ------
0.754545454545
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  0  0  0  0  2  1  0]
 [ 0  0 11  1  0  0  0  0  2  1]
 [ 0  0  0 14  0  0  0  1  0  0]
 [ 0  0  0  2  9  0  0  0  1  3]
 [ 0  6  0  0  0  8  4  1  1  0]
 [ 0  0  0  0  0  1 14  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/8/weights.42-0.50.hdf5
------ TRAIN ACCURACY:  weights.42-0.50.hdf5  ------
0.8328125
[[238   0   0   0   0   0   2   0   0   0   0]
 [  1 126   0  13   0   0   0   0   0   0   0]
 [  0   0 207   0   1   1   5  12  15   6   3]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  84   0   1   1   2   2   0]
 [  0   0   0   0   0  88   0   0   0   1   1]
 [  0   0   3   0   0   0 201  31   4   1   0]
 [  0   0  27   0   0   0  29 179   8   2   0]
 [  1   0   7   0   0   1  10   8 242  29  22]
 [  0   0   7   0   0   0  15   4  29 285  20]
 [  1   0   7   1   1   2   7   1  48  34 263]]
------ TEST ACCURACY:  weights.42-0.50.hdf5  ------
0.578947368421
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  0  0  3  0  0]
 [ 4  1  0 10  0  0  0  0  0  0  0]
 [ 0  0  0  0  4  2  0  0  3  1  0]
 [ 0  0  0  0  0  6  0  0  1  2  1]
 [ 0  0  0  0  0  0  2 13  0  0  0]
 [ 0  0  8  0  0  0  0  6  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/9/weights.37-0.56.hdf5
------ TRAIN ACCURACY:  weights.37-0.56.hdf5  ------
0.812929292929
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 122   0   3   0   0   0   0   0   0   0]
 [  0   0 209   0   2   0   9   7  12   5   6]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  1   0   0   0  95   0   0   0   2   0   2]
 [  1   0   0   0   0  88   0   0   0   2   9]
 [  0   0   5   0   1   0 199  24   6   4   1]
 [  0   0  38   0   0   0  61 130   8   3   5]
 [  0   0   6   0   1   1  10   3 216  20  33]
 [  0   0   5   0   0   0  13   1  45 218  48]
 [  0   0   5   1   1   1   1   0  40   8 278]]
------ TEST ACCURACY:  weights.37-0.56.hdf5  ------
0.566666666667
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 13  0  2  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  0  2  0  0]
 [ 0 10  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  1 10  1  1  0  0]
 [ 0  0  0  0  0  5  7  3  0  0]
 [ 0  0  2  0  1  1  1 15  6  4]
 [ 1  1  0  0  0  1  0  4  6 17]
 [ 1  3  0  1  0  0  0  6  1 18]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/10/weights.27-0.52.hdf5
------ TRAIN ACCURACY:  weights.27-0.52.hdf5  ------
0.755294117647
[[236   0   0   0   1   0   3   0   0   0   0]
 [  0 134   0   3   0   0   2   0   0   0   1]
 [  0   0 211   0   2   0   8  11  14   2   2]
 [  0   3   0 213   0   2   0   0   0   1   1]
 [  0   0   0   0  79   0   0   0   4   0   2]
 [  0   0   0   0   4  76   1   0   1   1   2]
 [  0   0  18   0   1   0 170  46   5   0   0]
 [  0   0  47   0   0   0  46 147   4   0   1]
 [  0   0  11   0   4   1  19   8 222  15  40]
 [  1   0  21   0   0   0  18  12  78 177  53]
 [  0   1  20   1   1   3   8   1  61   8 261]]
------ TEST ACCURACY:  weights.27-0.52.hdf5  ------
0.485714285714
[[ 7  0  3  5  0  0  0  0  0  0]
 [ 0  1  0  2  1  0  0  8  1  2]
 [ 0  0 13  0  2  0  0  0  0  0]
 [ 0  0  0  9  5  0  0  0  0  1]
 [ 0  0  0  0 11  0  0  0  0  4]
 [ 0  1  0  0  0  9  5  0  0  0]
 [ 0  4  0  0  0  2  1  6  1  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/11/weights.39-0.53.hdf5
------ TRAIN ACCURACY:  weights.39-0.53.hdf5  ------
0.810176125245
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 134   0   6   0   0   0   0   0   0   0]
 [  0   0 209   0   0   0   3  17  10   1   5]
 [  2   0   0 217   0   0   0   0   0   1   0]
 [  2   0   2   0  83   1   0   0   2   0   0]
 [  0   0   0   0   0  86   0   0   0   0   4]
 [  0   0  19   0   0   0 178  37   5   0   1]
 [  0   0  36   0   0   0  40 160   7   0   2]
 [  1   0  10   0   1   1   9   5 244  21  28]
 [  0   0  16   0   0   0   9   8  38 229  60]
 [  1   0  15   1   0   3   1   2  38  12 292]]
------ TEST ACCURACY:  weights.39-0.53.hdf5  ------
0.58
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  2  0  2  4  1]
 [ 0  1  0 14  0  0  0  0  0  0  0]
 [ 2  0  0  0  8  0  0  0  0  0  0]
 [ 1  0  0  0  1  8  0  0  0  0  0]
 [ 0  0  1  0  0  0  0 13  1  0  0]
 [ 0  0 10  0  0  0  0  2  3  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/12/weights.29-0.51.hdf5
------ TRAIN ACCURACY:  weights.29-0.51.hdf5  ------
0.751903807615
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  0   0 194   0   3   0   7   5  21   4  11]
 [  5  23   0 190   0   0   0   0   0   0   2]
 [  0   3   1   0  76   1   0   0   2   2   0]
 [  0   0   0   0   0  76   0   0   0   0   4]
 [  0   0  23   0   2   0 187  13  10   4   1]
 [  0   0  58   0   0   0  74  94  11   3   5]
 [  2   3  11   0   1   1   8   2 198  21  58]
 [  0   1   7   0   0   0   9   3  50 188  87]
 [  2   4  12   1   1   1   1   0  25   8 295]]
------ TEST ACCURACY:  weights.29-0.51.hdf5  ------
0.4625
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  0  0  0  3  4]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [12  0  0  0  1  2  0  0  0  0  0]
 [ 7  3  0  7  0  3  0  0  0  0  0]
 [ 0  0  1  0  0  0 14  0  0  0  0]
 [ 0  0  4  0  0  0  6  5  0  0  0]
 [ 1  0  1  0  0  0  0  0  0  2 11]
 [ 0  1  2  0  0  0  0  0  0  4  8]
 [ 0  1  0  0  2  0  0  0  7  1  4]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/13/weights.32-0.48.hdf5
------ TRAIN ACCURACY:  weights.32-0.48.hdf5  ------
0.77734375
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 130   0   5   0   0   0   0   0   0   0]
 [  0   0 206   0   1   0   2  17  16   5  13]
 [  0   0   0 229   0   0   0   0   1   0   0]
 [  1   1   1   1  87   0   0   0   2   4   3]
 [  0   0   0   0   0  93   0   0   0   2   5]
 [  0   0  31   0   1   0 142  67   4   4   1]
 [  0   0  50   0   0   0  21 173   5   2   4]
 [  0   0  13   0   2   1   7  11 193  27  46]
 [  0   0  17   0   0   0   7  13  23 223  57]
 [  1   0  14   1   2   2   1   2  40  13 269]]
------ TEST ACCURACY:  weights.32-0.48.hdf5  ------
0.6
[[ 5  0  0  0  4  0  0  1  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  1  0  0]
 [ 0  1  0  4  0  0  0  0  0]
 [ 0  0  1  0  2  0  2  0  0]
 [ 0  0  0  0  1  4  0  0  0]
 [ 0  0  0  0  1  0  8  5  6]
 [ 0  0  0  0  0  0  0  9 11]
 [ 0  0  0  0  0  0  2  2 16]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/14/weights.47-0.52.hdf5
------ TRAIN ACCURACY:  weights.47-0.52.hdf5  ------
0.823461538462
[[248   0   0   0   0   0   2   0   0   0   0]
 [  1 134   0   5   0   0   0   0   0   0   0]
 [  1   0 212   0   0   0   5  24  10   2   1]
 [  0   0   0 234   0   0   0   0   0   1   0]
 [  3   0   4   0  82   7   0   0   3   1   0]
 [  0   0   1   0   0  99   0   0   0   0   0]
 [  0   0  10   0   0   0 161  81   3   0   0]
 [  0   0  27   0   0   0  13 207   1   2   0]
 [  2   0  11   0   0   1   5  15 235  31  15]
 [  0   0  16   0   0   0  10  13  23 275   8]
 [  1   0  19   1   0   2   1   1  31  45 254]]
------ TEST ACCURACY:  weights.47-0.52.hdf5  ------
0.527272727273
[[4 1 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 5 0 5 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 2 0 8 0 0 0]
 [0 0 1 0 2 1 0 1]
 [0 0 1 1 4 0 6 3]
 [0 0 1 2 0 1 1 5]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/15/weights.48-0.50.hdf5
------ TRAIN ACCURACY:  weights.48-0.50.hdf5  ------
0.85137254902
[[243   0   0   0   0   0   2   0   0   0   0]
 [  0 126   0   8   0   0   0   0   0   0   1]
 [  0   0 216   0   2   0   6  23   9   0   4]
 [  0   0   0 224   0   0   0   0   0   1   0]
 [  0   0   0   0  98   0   0   0   2   0   0]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   0   0   0   0 210  31   1   2   1]
 [  0   0  21   0   0   0  31 191   4   1   2]
 [  0   0   6   0   0   1  14   8 220  22  29]
 [  0   0   9   0   0   0  19   7  22 245  43]
 [  0   0   6   1   5   1   6   2  19   6 299]]
------ TEST ACCURACY:  weights.48-0.50.hdf5  ------
0.571428571429
[[ 8  0  0  0  2  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  4  1  0  0  0]
 [ 0  0  0 10  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  8  2  0  0  0]
 [ 0  0  0  0  0  3  7  0  0  0]
 [ 0  0  0  0  0  0  2 10  3  5]
 [ 2  0  0  0  0  0  0  3  6  4]
 [ 0  0  0  0  0  0  0  3  6 11]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/16/weights.29-0.52.hdf5
------ TRAIN ACCURACY:  weights.29-0.52.hdf5  ------
0.751338199513
[[213   0   0   0   0   0   2   0   0   0   0]
 [  2  97   0   1   0   0   0   0   0   0   0]
 [  1   0 194   0   3   0   3   5   9   1   4]
 [  7   1   0 184   1   1   0   0   0   0   1]
 [  2   0   0   0  60   0   0   0   2   1   0]
 [  0   0   0   0   0  66   0   0   0   1   3]
 [  0   0  40   0   1   0 127  36   6   0   0]
 [  1   0  68   0   0   0  27 113   3   0   3]
 [  8   0  16   0   0   1   2   5 161   7  35]
 [  7   0  21   0   0   0   3   6  37 118  68]
 [  4   0  27   1   2   2   0   0  21   2 211]]
------ TEST ACCURACY:  weights.29-0.52.hdf5  ------
0.496666666667
[[40  0  0  0  0  0  0  0  0  0  0]
 [24 15  0  0  0  0  0  0  0  0  1]
 [ 0  0 40  0  0  0  0  3  2  0  0]
 [ 5  0  0 35  0  0  0  0  0  0  0]
 [ 2  0  4  0 23  0  0  0  1  4  1]
 [ 0  0  2  1  4  4  0  0  6  2 11]
 [ 0  0  5  0  0  0 24 16  0  0  0]
 [ 1  0 22  0  0  0  4 18  0  0  0]
 [ 0  0 13  0  0  1  5  6 30 12 18]
 [ 1  0 15  0  0  0  6  4 17 24 33]
 [ 1  1  4  0  0  1  1  1 23 18 45]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_spca_conv1d_10/17/weights.49-0.42.hdf5
------ TRAIN ACCURACY:  weights.49-0.42.hdf5  ------
0.825858585859
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  0   0 218   0   0   0   3  18   8   1   2]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  1   0   0   0  97   0   0   0   1   0   1]
 [  0   0   0   0   0  93   0   0   2   4   1]
 [  0   0   7   0   1   0 180  48   3   0   1]
 [  0   0  32   0   0   0  28 176   5   3   1]
 [  0   0  13   0   2   1   7   3 207  28  29]
 [  1   2   9   0   0   0   8   5  28 237  40]
 [  0   1   5   4   3   0   1   1  29  38 253]]
------ TEST ACCURACY:  weights.49-0.42.hdf5  ------
0.488888888889
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  7  0  8  0  0  0  0  0  0  0]
 [ 0  0  6  0  2  0  0  1  4  2  0]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 11  4  0  0  0]
 [ 0  0  4  0  0  0  5  4  2  0  0]
 [ 0  0  3  0  0  0  1  4 14  4  4]
 [ 0  0  3  0  0  0  6  5 10  3  3]
 [ 1  0  0  1  2  1  0  0 10  2 13]]
[0.5368421052631579, 0.46, 0.5027027027027027, 0.6157894736842106, 0.580952380952381, 0.34, 0.7545454545454545, 0.5789473684210527, 0.5666666666666667, 0.4857142857142857, 0.58, 0.4625, 0.6, 0.5272727272727272, 0.5714285714285714, 0.49666666666666665, 0.4888888888888889]
0.538171605424
0.0851352496799

Process finished with exit code 0

'''