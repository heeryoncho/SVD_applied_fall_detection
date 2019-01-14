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

X_smv = pickle.load(open("data/X_umafall_smv.p", "rb"))
X_raw = pickle.load(open("data/X_umafall_raw.p", "rb"))
X = np.concatenate((X_smv, X_raw), axis=1)

y = pickle.load(open("data/y_umafall_raw.p", "rb"))

n_classes = 11
signal_rows = 1800
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

    new_dir = 'umafall/model/umafall_raw_smv_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_raw_smv_conv1d_10/' + str(i+1) + '/'
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
[0.5684210526315789, 0.42, 0.6162162162162163, 0.6105263157894737, 0.6095238095238096, 0.56, 0.7272727272727273, 0.7368421052631579, 0.6, 0.580952380952381, 0.58, 0.65625, 0.6, 0.5272727272727272, 0.580952380952381, 0.655, 0.25]
0.581131159757
0.108312508042
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_raw_smv_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/1/weights.23-0.60.hdf5
------ TRAIN ACCURACY:  weights.23-0.60.hdf5  ------
0.86490872211
[[238   0   2   0   0   0   0   0   0   0   0]
 [  1 123   0   1   0   0   0   0   0   0   0]
 [  0   0 229   0   0   0   0  21   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  94   0   0   0   0   0   0]
 [  0   0   5   1   6  88   0   0   0   0   0]
 [  0   0   1   0   0   0 199  28   7   2   3]
 [  0   0  48   0   0   0   3 190   0   2   2]
 [  2   0   5   0   0   0   3   0 230  34  16]
 [  0   0   0   0   0   0   6   3  25 255  31]
 [  0   0   0   0   0   0   8   0  27  34 266]]
------ TEST ACCURACY:  weights.23-0.60.hdf5  ------
0.568421052632
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  4  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  2  0  0  0]
 [ 0  0  0 14  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  5  0 10  0]
 [ 0  0  0  0  0  0  9  0  6  0]
 [ 0  0  0  0  0  0  0 24  6  0]
 [ 0  0  0  0  0  0  0  8 22 10]
 [ 0  0  0  0  0  0  0  0 30  0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/2/weights.33-0.59.hdf5
------ TRAIN ACCURACY:  weights.33-0.59.hdf5  ------
0.852694610778
[[236   0   1   3   0   0   0   0   0   0   0]
 [ 11   0   0 114   0   0   0   0   0   0   0]
 [  0   0 222   0   0   0   2  26   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  94   0   0   0   0   0   0]
 [  0   0   4   0   0  95   0   1   0   0   0]
 [  0   0   1   0   0   0 227   9   1   0   2]
 [  0   0  17   0   0   0   9 216   0   2   1]
 [  1   0   3   0   0   0   3   2 253  32  11]
 [  1   0   0   0   0   0   9   0   8 294  18]
 [  0   0   0   0   0   0   3   0  13  55 279]]
------ TEST ACCURACY:  weights.33-0.59.hdf5  ------
0.42
[[15  0  0  0  0  0  0  0  0]
 [ 4  0  0 11  0  0  0  0  0]
 [ 0  0  7  0  0  8  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 5  0  6  0  3  0  0  0  1]
 [ 0  0 12  0  3  0  0  0  0]
 [ 0  0  0  0  0  0  3  3  9]
 [ 0  0  0  0  0  0  2 18 10]
 [ 0  0  0  1  0  0 12  0  2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/3/weights.37-0.58.hdf5
------ TRAIN ACCURACY:  weights.37-0.58.hdf5  ------
0.840890688259
[[238   0   2   0   0   0   0   0   0   0   0]
 [  9   0   0 116   0   0   0   0   0   0   0]
 [  0   0 241   0   0   0   0   4   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   9   0  91   0   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   3   0   0   0 220  11   1   1   4]
 [  0   0  47   0   0   0   2 193   0   2   1]
 [  1   0   5   0   0   0   0   0 232  25  27]
 [  1   0   0   0   0   0   2   1  18 262  46]
 [  0   0   0   0   0   0   0   0  10  40 285]]
------ TEST ACCURACY:  weights.37-0.58.hdf5  ------
0.616216216216
[[15  0  0  0  0  0  0  0  0]
 [ 5  0  0 10  0  0  0  0  0]
 [ 0  0 18  0  0  2  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0 10  0  3  0  2]
 [ 0  0  0  0 10  5  0  0  0]
 [ 0  0  0  0  0  0 16  0 14]
 [ 0  0  0  0  0  0  3  7 20]
 [ 0  0  0  0  0  0  2  0 28]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/4/weights.29-0.63.hdf5
------ TRAIN ACCURACY:  weights.29-0.63.hdf5  ------
0.841784989858
[[234   0   1   0   0   0   0   0   0   0   0]
 [  9   2   0 114   0   0   0   0   0   0   0]
 [  0   0 238   0   0   0   0  12   0   0   0]
 [  0   0   0 219   1   0   0   0   0   0   0]
 [  0   0   7   0  93   0   0   0   0   0   0]
 [  0   0   5   0   3  92   0   0   0   0   0]
 [  0   0   2   0   0   0 195  20   4   5   4]
 [  0   0  41   0   0   0   4 197   0   2   1]
 [  2   0   0   0   0   0   2   0 240  37   9]
 [  0   0   0   0   0   0   6   1   4 309  15]
 [  0   0   0   0   0   0   0   3  12  64 256]]
------ TEST ACCURACY:  weights.29-0.63.hdf5  ------
0.610526315789
[[20  0  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 4  0  0  0 17  0  0  2  2]
 [ 0  0  1  0  3 11  0  0  0]
 [ 0  0  5  0  0  0  7 17  1]
 [ 0  0  0  0  0  0  2 17  6]
 [ 0  0  0  0  0  0  5 10 15]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/5/weights.43-0.60.hdf5
------ TRAIN ACCURACY:  weights.43-0.60.hdf5  ------
0.842352941176
[[238   0   2   0   0   0   0   0   0   0   0]
 [ 31   0   0 107   0   0   0   1   0   1   0]
 [  0   0 226   0   0   0   0  19   5   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  94   0   0   0   0   0   0]
 [  0   0   4   0   0  95   0   1   0   0   0]
 [  0   0   3   0   0   0 215  14   6   2   0]
 [  0   0  22   0   0   0   3 219   0   1   0]
 [  2   0   1   0   0   0   0   4 256  44  13]
 [  0   0   0   0   0   0   7   0   7 333  13]
 [  0   0   0   0   0   0   0   0  13  70 252]]
------ TEST ACCURACY:  weights.43-0.60.hdf5  ------
0.609523809524
[[15  0  0  0  0  0  0  0]
 [ 0 11  0  0  4  0  0  0]
 [ 0  0 15  0  0  0  0  0]
 [ 0  0  0 11  0  0  0  4]
 [ 0  6  0  0  9  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  1  4  1 21  3]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/6/weights.09-0.59.hdf5
------ TRAIN ACCURACY:  weights.09-0.59.hdf5  ------
0.732821497121
[[245   0   3   2   0   0   0   0   0   0   0]
 [  7   0   1 132   0   0   0   0   0   0   0]
 [  1   0 255   0   0   0   0   4   0   0   0]
 [  0   0   0 228   2   5   0   0   0   0   0]
 [  4   0  15   0  70  11   0   0   0   0   0]
 [  1   0   8   2   1  88   0   0   0   0   0]
 [  1   0   9   0   0   0 218   3   2  13   9]
 [  0   0  99   0   0   0  69  71   0   9   2]
 [  3   0   5   0   0   0  16   0 188  59  39]
 [  2   0   0   0   0   0  14   0  11 294  29]
 [  0   0   0   0   0   0   7   0  14  82 252]]
------ TEST ACCURACY:  weights.09-0.59.hdf5  ------
0.56
[[ 4  1  0  0  0  0  0]
 [ 0  5  0  0  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0  6  4  0  0  0  0]
 [ 0  0  0  0 10  0  0]
 [ 0  0  0  0  0  9  1]
 [ 0  0  0  0  0 10  0]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/7/weights.50-0.67.hdf5
------ TRAIN ACCURACY:  weights.50-0.67.hdf5  ------
0.893123772102
[[236   0   4   0   0   0   0   0   0   0   0]
 [  0 139   1   0   0   0   0   0   0   0   0]
 [  0   0 249   0   0   0   0   1   0   0   0]
 [  0   0   0 219   0   1   0   0   0   0   0]
 [  0   0   8   0  77   0   0   0   0   0   0]
 [  0   0   5   0   0  80   0   0   0   0   0]
 [  0   0   3   0   0   0 223   3   0   0   6]
 [  0   0  52   0   0   0   2 186   2   2   1]
 [  1   0   5   0   0   0   0   0 269  24  21]
 [  0   0   0   0   0   0   2   0  22 291  45]
 [  0   0   0   0   0   0   1   0  23  37 304]]
------ TEST ACCURACY:  weights.50-0.67.hdf5  ------
0.727272727273
[[15  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0  7  6  2  0  0  0  0]
 [ 2  0  0 11  2  0  0  0  0]
 [ 1  0  0  9  5  0  0  0  0]
 [ 0  0  0  0  0 18  0  2  0]
 [ 0  3  0  0  0  1  9  0  2]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/8/weights.23-0.63.hdf5
------ TRAIN ACCURACY:  weights.23-0.63.hdf5  ------
0.80625
[[238   0   2   0   0   0   0   0   0   0   0]
 [ 13   0   0 127   0   0   0   0   0   0   0]
 [  0   0 247   0   0   0   0   2   1   0   0]
 [  0   0   0 218   2   0   0   0   0   0   0]
 [  0   0   7   0  81   2   0   0   0   0   0]
 [  0   0   5   0   0  85   0   0   0   0   0]
 [  1   0   9   0   0   0 193  20   9   0   8]
 [  0   0  87   0   0   0   3 151   0   2   2]
 [  2   0   5   0   0   0   0   0 266  20  27]
 [  1   0   0   0   0   0   1   1  34 281  42]
 [  0   0   0   0   0   0   0   0  18  43 304]]
------ TEST ACCURACY:  weights.23-0.63.hdf5  ------
0.736842105263
[[15  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0]
 [ 0  0 15  0  0  0  0]
 [ 0  3  0  5  2  0  0]
 [ 0  1  0  0  9  0  0]
 [ 0  2  0  0  0  8  5]
 [ 0 12  0  0  0  0  3]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/9/weights.40-0.63.hdf5
------ TRAIN ACCURACY:  weights.40-0.63.hdf5  ------
0.861414141414
[[239   0   1   0   0   0   0   0   0   0   0]
 [ 21   1   0 103   0   0   0   0   0   0   0]
 [  0   0 248   0   0   0   0   2   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  94   0   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   1   0   0   0 232   2   1   1   3]
 [  0   0  43   0   0   0   9 189   0   2   2]
 [  1   0   5   0   0   0   1   0 225  33  25]
 [  0   0   0   0   0   0   3   0   9 305  13]
 [  0   0   0   0   0   0   1   1  12  37 284]]
------ TEST ACCURACY:  weights.40-0.63.hdf5  ------
0.6
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0 12  0  0  0  0  3  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0 11  0  4  0  0]
 [ 0  0  2  0 10  1  1  0  1]
 [ 0  0  0  0  0  0 24  0  6]
 [ 0  0  0  0  0  0  0 23  7]
 [ 0  0  0  0  0  0  4 19  7]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/10/weights.47-0.63.hdf5
------ TRAIN ACCURACY:  weights.47-0.63.hdf5  ------
0.853333333333
[[238   0   2   0   0   0   0   0   0   0   0]
 [ 16   1   0 123   0   0   0   0   0   0   0]
 [  0   0 228   0   0   0   0  22   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  79   0   0   0   0   0   0]
 [  0   0   5   0   0  80   0   0   0   0   0]
 [  0   0   0   0   0   0 221  12   2   2   3]
 [  0   0  10   0   0   0   1 231   0   2   1]
 [  1   0   5   0   0   0   2   0 266  29  17]
 [  1   0   0   0   0   0   9   0   9 311  30]
 [  0   0   0   0   0   0   0   0  13  51 301]]
------ TEST ACCURACY:  weights.47-0.63.hdf5  ------
0.580952380952
[[15  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0]
 [ 1  2  0 10  2  0  0  0]
 [ 0  0  0 14  1  0  0  0]
 [ 0  0  0  0  0  5  9  1]
 [ 0 15  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/11/weights.46-0.64.hdf5
------ TRAIN ACCURACY:  weights.46-0.64.hdf5  ------
0.901369863014
[[238   0   2   0   0   0   0   0   0   0   0]
 [  0 136   0   3   0   0   1   0   0   0   0]
 [  0   0 241   0   0   0   0   4   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   7   0  83   0   0   0   0   0   0]
 [  0   0   5   0   0  85   0   0   0   0   0]
 [  0   0   1   0   0   0 223   7   7   0   2]
 [  0   0  32   0   0   0   2 208   0   2   1]
 [  1   0   5   0   0   0   0   0 254  44  16]
 [  0   0   0   0   0   0   7   0  13 326  14]
 [  0   0   0   0   0   0   1   0  14  61 289]]
------ TEST ACCURACY:  weights.46-0.64.hdf5  ------
0.58
[[15  0  0  0  0  0  0]
 [ 0 20  0  0  0  0  0]
 [ 0  0 15  0  0  0  0]
 [ 0  1  0  5  4  0  0]
 [ 0  0  0  7  3  0  0]
 [ 0  1  0  0  0  0 14]
 [ 0 15  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/12/weights.36-0.54.hdf5
------ TRAIN ACCURACY:  weights.36-0.54.hdf5  ------
0.833667334669
[[238   0   2   0   0   0   0   0   0   0   0]
 [ 23   0   0 117   0   0   0   0   0   0   0]
 [  0   0 218   0   0   0   0  27   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  79   0   0   0   0   0   0]
 [  0   0   0   0   0  80   0   0   0   0   0]
 [  0   0   1   0   0   0 224   6   1   5   3]
 [  0   0  19   0   0   0   4 216   0   4   2]
 [  2   0   3   0   0   0   0   2 232  37  29]
 [  1   0   0   0   0   0   9   0   2 295  38]
 [  0   0   0   0   0   0   0   0   9  63 278]]
------ TEST ACCURACY:  weights.36-0.54.hdf5  ------
0.65625
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 20  0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 7  0  2  4  2  0  0  0  0  0]
 [ 0  5 13  0  2  0  0  0  0  0]
 [ 0  0  0  0  0 14  0  0  0  1]
 [ 0  8  0  0  0  2  5  0  0  0]
 [ 0  0  0  0  0  0  0  0 10  5]
 [ 0  0  0  0  0  0  0  0 15  0]
 [ 0  0  0  0  0  0  0  0  0 15]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/13/weights.41-0.61.hdf5
------ TRAIN ACCURACY:  weights.41-0.61.hdf5  ------
0.853515625
[[245   0   0   0   0   0   0   0   0   0   0]
 [ 15   1   0 119   0   0   0   0   0   0   0]
 [  0   0 260   0   0   0   0   0   0   0   0]
 [  0   0   0 230   0   0   0   0   0   0   0]
 [  0   0   7   0  93   0   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   4   0   0   0 231   8   1   1   5]
 [  0   0  59   0   0   0   5 189   0   2   0]
 [  1   0   5   0   0   0   0   0 239  34  21]
 [  0   0   0   0   0   0   2   0   9 314  15]
 [  0   0   0   0   0   0   0   0  10  47 288]]
------ TEST ACCURACY:  weights.41-0.61.hdf5  ------
0.6
[[ 7  0  3  0  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  5  0  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  0  0  1  0  4  0  0]
 [ 0  0  4  0  0  1  0  0  0]
 [ 0  0  0  0  0  0 14  2  4]
 [ 0  0  0  0  0  0  0  4 16]
 [ 0  0  0  0  0  0  0  0 20]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/14/weights.19-0.66.hdf5
------ TRAIN ACCURACY:  weights.19-0.66.hdf5  ------
0.842307692308
[[247   0   3   0   0   0   0   0   0   0   0]
 [  0 125   1  14   0   0   0   0   0   0   0]
 [  0   0 233   0   0   0   0  22   0   0   0]
 [  0   1   0 232   2   0   0   0   0   0   0]
 [  0   0   7   0  91   2   0   0   0   0   0]
 [  1   0   5   0   3  91   0   0   0   0   0]
 [  0   0   3   0   0   0 193  38  11   7   3]
 [  0   0  41   0   0   0   6 200   0   2   1]
 [  2   0   5   0   0   0   0   0 251  45  12]
 [  0   0   0   0   0   0   3   1  34 299   8]
 [  0   0   0   0   0   0   4   3  40  80 228]]
------ TEST ACCURACY:  weights.19-0.66.hdf5  ------
0.527272727273
[[5 0 0 0 0 0 0 0]
 [0 1 0 0 9 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 1 0 0 9 0 0 0]
 [0 0 0 1 0 3 0 1]
 [0 0 1 1 2 0 9 2]
 [0 0 0 0 0 4 4 2]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/15/weights.29-0.59.hdf5
------ TRAIN ACCURACY:  weights.29-0.59.hdf5  ------
0.82431372549
[[243   0   2   0   0   0   0   0   0   0   0]
 [ 22   0   0 113   0   0   0   0   0   0   0]
 [  1   0 233   0   0   0   0  26   0   0   0]
 [  0   0   0 225   0   0   0   0   0   0   0]
 [  0   0   6   0  94   0   0   0   0   0   0]
 [  0   0   5   0   1  94   0   0   0   0   0]
 [  0   0   3   0   0   0 203  12  10  13   4]
 [  0   0  32   0   0   0   8 204   0   5   1]
 [  2   0   4   0   0   0   0   1 246  30  17]
 [  1   0   0   0   0   0   1   1  24 294  24]
 [  0   0   0   0   0   0   2   0  22  55 266]]
------ TEST ACCURACY:  weights.29-0.59.hdf5  ------
0.580952380952
[[ 9  0  0  0  1  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  5  0  0  0  0  0  0  0]
 [ 0  0  0 10  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  4  0  0  6  0]
 [ 0  0  0  0  0  1  8  0  1  0]
 [ 0  0  0  0  0  0  0  8  6  6]
 [ 0  0  0  0  0  0  0  1  8  6]
 [ 0  0  0  0  0  0  0  0 11  9]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/16/weights.31-0.53.hdf5
------ TRAIN ACCURACY:  weights.31-0.53.hdf5  ------
0.819464720195
[[212   0   3   0   0   0   0   0   0   0   0]
 [  7   0   1  92   0   0   0   0   0   0   0]
 [  0   0 216   0   0   0   0   1   3   0   0]
 [  0   0   0 195   0   0   0   0   0   0   0]
 [  0   0   7   0  58   0   0   0   0   0   0]
 [  0   0   5   0   0  65   0   0   0   0   0]
 [  0   0   7   0   0   0 183   9   5   1   5]
 [  0   0  64   0   0   0   1 148   0   1   1]
 [  0   0   5   0   0   0   0   0 170  43  17]
 [  0   0   0   0   0   0   3   1   6 235  15]
 [  0   0   3   0   0   0   0   0   5  60 202]]
------ TEST ACCURACY:  weights.31-0.53.hdf5  ------
0.655
[[39  0  0  0  0  1  0  0  0  0  0]
 [ 1  0  0 39  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  0 13  0  0  0]
 [ 0  0  0 40  0  0  0  0  0  0  0]
 [ 0  0  6  0  7 21  0  1  0  0  0]
 [ 0  0  1  1  8 20  0  0  0  0  0]
 [ 0  0  2  0  0  0 40  3  0  0  0]
 [ 0  0 20  0  0  0  5 20  0  0  0]
 [ 1  0  0  0  0  0  2  0 70  0 12]
 [ 7  0  0  0  1  0  0  0 10 73  9]
 [ 0  0  0  0  0  0  0  0 22 21 52]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_smv_conv1d_10/17/weights.01-0.84.hdf5
------ TRAIN ACCURACY:  weights.01-0.84.hdf5  ------
0.496161616162
[[192   0   0  34   0   0  14   0   0   0   0]
 [  4   0   0 121   0   0   0   0   0   0   0]
 [190   0  17   0   0   0  24  18   0   1   0]
 [  4   0   0 216   0   0   0   0   0   0   0]
 [ 87   0   0  13   0   0   0   0   0   0   0]
 [ 81   0   0  19   0   0   0   0   0   0   0]
 [  5   0   2   0   0   0  80  30  14  70  39]
 [ 69   0   7   1   0   0  25 100   0  33  10]
 [ 10   0   0   1   0   0   2   0 197  50  30]
 [  0   0   0   5   0   0   1   3  15 269  37]
 [  0   0   0   4   0   0   0   2  28 144 157]]
------ TEST ACCURACY:  weights.01-0.84.hdf5  ------
0.25
[[ 0  0  0  9  4  0  1  0  1]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0 13  0  0  0  2]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  0  3  0 11  1]
 [ 0  0  0  0  2  0  0  7  6]
 [ 0  0  0  0  0  0  0 30  0]
 [ 0  0  0  0  0  0  1 29  0]
 [ 0  0  0  0  0  0  0 29  1]]
[0.5684210526315789, 0.42, 0.6162162162162163, 0.6105263157894737, 0.6095238095238096, 0.56, 0.7272727272727273, 0.7368421052631579, 0.6, 0.580952380952381, 0.58, 0.65625, 0.6, 0.5272727272727272, 0.580952380952381, 0.655, 0.25]
0.581131159757
0.108312508042

Process finished with exit code 0

'''