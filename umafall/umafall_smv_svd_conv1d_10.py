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
X_smv = pickle.load(open("data/X_umafall_smv.p", "rb"))
X = np.concatenate((X_svd, X_smv), axis=1)

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

    new_dir = 'model/umafall_smv_svd_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_smv_svd_conv1d_10/' + str(i+1) + '/'
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
[0.6105263157894737, 0.6133333333333333, 0.4648648648648649, 0.6368421052631579, 0.49523809523809526, 0.4, 0.7, 0.6526315789473685, 0.5611111111111111, 0.5238095238095238, 0.55, 0.4875, 0.5368421052631579, 0.6181818181818182, 0.580952380952381, 0.55, 0.4444444444444444]
0.554486922188
0.0775702744978
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_smv_svd_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/1/weights.43-0.53.hdf5
------ TRAIN ACCURACY:  weights.43-0.53.hdf5  ------
0.875050709939
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 120   0   5   0   0   0   0   0   0   0]
 [  0   0 229   0   0   0   5   5   9   2   0]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  0   0   0   0  97   0   1   0   2   0   0]
 [  0   0   0   0   0  99   0   0   1   0   0]
 [  0   0   1   0   0   0 228   9   1   0   1]
 [  0   0  31   0   0   0  28 180   4   1   1]
 [  0   0   8   0   0   1   6   7 231  12  25]
 [  0   0   5   0   0   1  12   6  44 220  32]
 [  0   0   3   0   1   1   1   1  30   2 296]]
------ TEST ACCURACY:  weights.43-0.53.hdf5  ------
0.610526315789
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  1  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  2  2  0  0  0]
 [ 0  0  0 13  0  2  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 15  0  0  0  0]
 [ 0  0  0  0  0  0 10  5  0  0  0]
 [ 0  0  3  0  1  0  0  0 18  1  7]
 [ 0  0  1  0  0  0  0  3 21  7  8]
 [ 0  0  0  0  0  1  0  0  9  2 18]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/2/weights.25-0.51.hdf5
------ TRAIN ACCURACY:  weights.25-0.51.hdf5  ------
0.830339321357
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  0   0 197   0   0   0   5   6  28   9   5]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  93   0   1   0   4   2   0]
 [  0   0   0   0   0  97   0   0   1   2   0]
 [  0   0   4   0   0   0 214  13   5   4   0]
 [  0   0  40   0   0   0  66 124  12   1   2]
 [  1   0   4   0   1   1  11   4 240  26  17]
 [  0   0   4   0   0   1   9   5  44 251  16]
 [  1   0   1   0   2   0   1   1  37  24 283]]
------ TEST ACCURACY:  weights.25-0.51.hdf5  ------
0.613333333333
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  1  1  0  0]
 [ 0  4  0 11  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  8  5  1  0  0]
 [ 0  0  9  0  0  1  4  1  0  0]
 [ 0  0  0  0  0  0  0  9  1  5]
 [ 1  0  0  0  1  0  0  5  7 16]
 [ 0  0  0  0  0  0  0  1  4 10]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/3/weights.50-0.52.hdf5
------ TRAIN ACCURACY:  weights.50-0.52.hdf5  ------
0.885425101215
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  1   0 220   0   0   0   2  11   2   5   4]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  96   0   2   0   1   0   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   1   0   0   0 213  24   0   1   1]
 [  0   0  18   0   0   0  43 181   1   1   1]
 [  0   0   7   0   0   0   6  10 191  24  52]
 [  0   0   3   0   0   0   9   4   9 285  20]
 [  0   0   1   0   1   0   0   2   3   9 319]]
------ TEST ACCURACY:  weights.50-0.52.hdf5  ------
0.464864864865
[[15  0  0  0  0  0  0  0  0]
 [ 1 11  0  0  1  0  0  2  0]
 [ 0  0 10  0  0  9  0  0  1]
 [ 0  9  0  5  0  0  0  0  1]
 [ 0  0  0  0 13  0  1  1  0]
 [ 0  0  0  0 11  4  0  0  0]
 [ 0  0  1  0  1  1  0  7 20]
 [ 2  0  1  0  1  0  5  3 18]
 [ 0  0  0  0  0  0  1  4 25]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/4/weights.45-0.56.hdf5
------ TRAIN ACCURACY:  weights.45-0.56.hdf5  ------
0.894523326572
[[233   0   0   0   0   0   1   1   0   0   0]
 [  0 120   0   5   0   0   0   0   0   0   0]
 [  1   0 210   0   2   0   4  24   4   5   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  97   0   1   0   1   0   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   0   0   0   0 207  22   1   0   0]
 [  0   0  12   0   0   0  14 216   1   2   0]
 [  0   0   7   0   1   1   8  14 215  23  21]
 [  0   0   2   0   0   1  12   7  14 287  12]
 [  0   0   3   0   4   0   0   3  10  14 301]]
------ TEST ACCURACY:  weights.45-0.56.hdf5  ------
0.636842105263
[[17  0  0  0  1  1  0  0  0  0  1]
 [ 0  6  0  9  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  0  2  0  0  2]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0 16  7  0  1  0]
 [ 0  0  2  0  0  0  4  9  0  0  0]
 [ 0  0  2  0  2  0  0  4 12  3  7]
 [ 0  0  1  0  0  0  1  1  2 14  6]
 [ 0  0  0  0  0  0  0  0  0  9 21]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/5/weights.11-0.51.hdf5
------ TRAIN ACCURACY:  weights.11-0.51.hdf5  ------
0.736078431373
[[234   0   0   0   0   0   4   1   0   1   0]
 [  0 134   0   4   0   0   1   0   0   1   0]
 [  2   0 204   0   0   0   8  13  16   3   4]
 [  0  10   0 205   0   1   0   0   3   0   1]
 [  3   0   2   0  84   0   2   0   4   2   3]
 [  2   0   0   2   0  88   1   0   2   2   3]
 [  0   0  15   0   0   0 193  25   6   1   0]
 [  0   0  62   0   0   0  59 109  13   1   1]
 [  3   0  15   0   0   1  23   3 218  29  28]
 [  2   0  20   0   0   2  28   5  81 165  57]
 [  5   0   4   0   2   0   9   0  56  16 243]]
------ TEST ACCURACY:  weights.11-0.51.hdf5  ------
0.495238095238
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  1  0  0  1]
 [ 0 10  0  5  0  0  0  0  0]
 [ 0  0  4  0  9  0  2  0  0]
 [ 0  0  2  0  6  7  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  7  1  4  3  3]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/6/weights.20-0.55.hdf5
------ TRAIN ACCURACY:  weights.20-0.55.hdf5  ------
0.787715930902
[[248   0   0   0   0   0   1   1   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  0   0 135   0   0   0   7  86  19   8   5]
 [  1  18   0 211   0   1   0   0   1   1   2]
 [  4   0   0   0  89   1   1   0   3   1   1]
 [  1   0   0   0   0  95   0   0   0   2   2]
 [  0   0   1   0   0   0 187  58   6   3   0]
 [  0   0   6   0   0   0  15 220   8   1   0]
 [  1   0   3   0   0   1  12  16 226  26  25]
 [  2   0   3   0   0   1   6  14  53 220  51]
 [  2   0   0   0   2   2   3   4  45  16 281]]
------ TEST ACCURACY:  weights.20-0.55.hdf5  ------
0.4
[[5 0 0 0 0 0 0]
 [0 4 0 0 1 0 0]
 [0 0 0 0 0 0 0]
 [1 0 2 5 2 0 0]
 [3 0 0 0 2 1 4]
 [0 0 2 0 4 2 2]
 [0 2 0 0 3 3 2]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/7/weights.36-0.53.hdf5
------ TRAIN ACCURACY:  weights.36-0.53.hdf5  ------
0.866011787819
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 137   0   3   0   0   0   0   0   0   0]
 [  1   0 221   0   0   0   2  14   8   3   1]
 [  0   0   0 218   0   2   0   0   0   0   0]
 [  0   0   0   0  81   0   0   1   2   1   0]
 [  0   0   0   0   0  83   0   0   2   0   0]
 [  0   0   4   0   0   0 197  28   3   2   1]
 [  0   0  27   0   0   0  19 194   2   3   0]
 [  0   0  13   0   0   0   6  10 234  32  25]
 [  0   0  10   0   0   0   6   8  28 279  29]
 [  0   0   5   0   1   0   2   3  14  18 322]]
------ TEST ACCURACY:  weights.36-0.53.hdf5  ------
0.7
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  0  7  0  4  0  0  1  2  1]
 [ 3  0  0 10  2  0  0  0  0  0]
 [ 0  0  0  1  3  0  0  2  2  7]
 [ 0  3  0  0  0 16  0  0  0  1]
 [ 0  0  0  0  0  4 11  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/8/weights.43-0.54.hdf5
------ TRAIN ACCURACY:  weights.43-0.54.hdf5  ------
0.8875
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 138   0   2   0   0   0   0   0   0   0]
 [  0   0 202   0   0   0   3  24  13   6   2]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  85   0   1   0   1   2   1]
 [  0   0   0   0   0  88   0   0   0   1   1]
 [  0   0   1   0   0   0 209  26   1   2   1]
 [  0   0  11   0   0   0  20 201   5   6   2]
 [  0   0   3   0   0   1   6   7 250  33  20]
 [  0   0   1   0   0   0   5   6  18 323   7]
 [  0   0   0   0   1   0   0   2  18  25 319]]
------ TEST ACCURACY:  weights.43-0.54.hdf5  ------
0.652631578947
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  0  1  2  1  0]
 [ 0  3  0 12  0  0  0  0  0  0  0]
 [ 0  0  0  0  3  0  0  0  2  1  4]
 [ 0  0  0  0  0  5  0  0  0  3  2]
 [ 0  0  0  0  0  0  6  9  0  0  0]
 [ 0  0  3  0  0  0  1 10  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/9/weights.39-0.53.hdf5
------ TRAIN ACCURACY:  weights.39-0.53.hdf5  ------
0.868686868687
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  1   0 180   0   0   0   7  39  13   5   5]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  97   0   1   0   1   0   1]
 [  0   0   0   0   0  98   0   0   0   1   1]
 [  0   0   1   0   0   0 218  18   1   2   0]
 [  0   0  10   0   0   0  38 191   1   3   2]
 [  0   0   3   0   2   1   9  12 202  43  18]
 [  0   0   1   0   0   2   9   7  13 288  10]
 [  0   0   1   0   2   1   1   2  12  22 294]]
------ TEST ACCURACY:  weights.39-0.53.hdf5  ------
0.561111111111
[[14  0  0  0  0  0  0  1  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  2  1  1  0]
 [ 0 10  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  1 12  1  1  0  0]
 [ 0  0  1  0  0  5  8  1  0  0]
 [ 0  0  0  0  0  1  3 10  9  7]
 [ 1  0  0  0  0  2  0  4 10 13]
 [ 1  0  0  0  0  0  0  6  7 16]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/10/weights.50-0.53.hdf5
------ TRAIN ACCURACY:  weights.50-0.53.hdf5  ------
0.893725490196
[[238   0   0   1   0   0   0   1   0   0   0]
 [  0 133   0   7   0   0   0   0   0   0   0]
 [  1   0 218   0   3   0   0  11   6  10   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  84   0   0   0   1   0   0]
 [  0   0   0   0   0  85   0   0   0   0   0]
 [  0   0   1   3   0   0 207  19   1   8   1]
 [  0   0  21   0   0   0  10 204   3   7   0]
 [  0   0   5   0   2   2   6   5 225  52  23]
 [  0   0   3   0   1   2   1   2   6 339   6]
 [  0   0   1   0   1   0   0   0   6  31 326]]
------ TEST ACCURACY:  weights.50-0.53.hdf5  ------
0.52380952381
[[11  0  0  4  0  0  0  0  0  0]
 [ 0  0  0  3  0  0  0  4  5  3]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0 13  1  0  0  0  0  1]
 [ 1  0  0  2  9  0  0  1  0  2]
 [ 0  0  0  0  0  6  9  0  0  0]
 [ 0  4  0  0  0  0  1  5  4  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/11/weights.48-0.51.hdf5
------ TRAIN ACCURACY:  weights.48-0.51.hdf5  ------
0.884931506849
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 136   0   4   0   0   0   0   0   0   0]
 [  0   0 214   0   0   0   3  12  10   5   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  88   0   0   0   2   0   0]
 [  0   0   0   0   0  90   0   0   0   0   0]
 [  0   0   2   0   0   0 206  27   2   3   0]
 [  0   0  18   0   0   0  16 198   7   6   0]
 [  0   0   3   0   0   1   8   9 248  40  11]
 [  0   0   3   0   0   2   0   5  15 330   5]
 [  0   0   1   0   1   2   0   1  18  49 293]]
------ TEST ACCURACY:  weights.48-0.51.hdf5  ------
0.55
[[15  0  0  0  0  0  0  0  0]
 [ 0  9  0  0  0  0  1  5  5]
 [ 0  0 15  0  0  0  0  0  0]
 [ 2  0  0  8  0  0  0  0  0]
 [ 2  0  0  1  7  0  0  0  0]
 [ 0  0  0  0  0  1 14  0  0]
 [ 0 12  0  0  0  0  0  3  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/12/weights.42-0.54.hdf5
------ TRAIN ACCURACY:  weights.42-0.54.hdf5  ------
0.882565130261
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 137   0   3   0   0   0   0   0   0   0]
 [  1   0 224   0   0   0   2  10   4   3   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  82   0   1   0   1   0   1]
 [  0   0   0   0   0  79   0   0   1   0   0]
 [  0   0   2   0   0   0 226  11   1   0   0]
 [  0   0  35   0   0   0  20 184   1   3   2]
 [  0   0  21   0   0   1   7   7 213  24  32]
 [  0   0  14   0   0   1  11   4  12 270  33]
 [  1   0   7   0   0   1   1   1   3   7 329]]
------ TEST ACCURACY:  weights.42-0.54.hdf5  ------
0.4875
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  1  1  4]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 9  0  2  1  3  0  0  0  0  0]
 [ 2  0 15  0  0  0  0  0  1  2]
 [ 0  2  0  0  0 13  0  0  0  0]
 [ 0  4  0  0  0  5  6  0  0  0]
 [ 0  0  0  0  0  0  0  1  6  8]
 [ 0  1  0  0  0  0  0  1  7  6]
 [ 0  1  0  1  0  1  0  4  2  6]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/13/weights.35-0.55.hdf5
------ TRAIN ACCURACY:  weights.35-0.55.hdf5  ------
0.862890625
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 135   0   0   0   0   0   0   0   0   0]
 [  0   0 233   0   0   0   2  16   5   3   1]
 [  0   0   0 228   0   0   0   0   0   1   1]
 [  0   0   1   0  98   0   0   0   1   0   0]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   4   0   0   0 210  33   0   1   2]
 [  0   0  30   0   0   0  24 197   2   1   1]
 [  0   0  19   0   0   1   7   8 213  17  35]
 [  0   0  13   0   0   0  12   6  28 240  41]
 [  0   0   9   0   1   0   1   2  16   5 311]]
------ TEST ACCURACY:  weights.35-0.55.hdf5  ------
0.536842105263
[[ 5  0  0  0  3  1  0  1  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  0  0  1]
 [ 0  1  0  4  0  0  0  0  0]
 [ 0  0  1  0  2  0  2  0  0]
 [ 0  0  0  0  2  3  0  0  0]
 [ 0  0  3  0  0  0  6  4  7]
 [ 0  0  0  0  0  0  0  6 14]
 [ 0  0  0  0  0  0  2  2 16]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/14/weights.45-0.50.hdf5
------ TRAIN ACCURACY:  weights.45-0.50.hdf5  ------
0.873846153846
[[248   0   0   0   0   0   2   0   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  1   0 222   0   0   0   6  14   8   4   0]
 [  0   0   0 234   0   0   0   0   0   1   0]
 [  0   0   0   0  96   0   1   0   1   1   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   0   0   0   0 238  13   1   2   1]
 [  0   0  21   0   0   0  54 168   1   5   1]
 [  1   0   7   0   0   1   7   6 228  43  22]
 [  0   0   4   0   0   1   7   3  14 309   7]
 [  0   0   1   0   2   0   2   2  18  40 290]]
------ TEST ACCURACY:  weights.45-0.50.hdf5  ------
0.618181818182
[[ 5  0  0  0  0  0  0]
 [ 0  4  0  5  0  1  0]
 [ 0  0  0  0  0  0  0]
 [ 0  2  0  8  0  0  0]
 [ 0  0  0  1  1  3  0]
 [ 0  0  2  1  1 10  1]
 [ 0  0  2  0  1  1  6]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/15/weights.48-0.55.hdf5
------ TRAIN ACCURACY:  weights.48-0.55.hdf5  ------
0.897254901961
[[243   0   0   0   0   0   2   0   0   0   0]
 [  0 134   0   1   0   0   0   0   0   0   0]
 [  1   0 232   0   0   0   4  15   5   3   0]
 [  0   0   0 224   0   0   0   0   0   1   0]
 [  0   0   0   0  96   0   2   0   2   0   0]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   0   0   0   0 235   8   0   2   0]
 [  0   0  14   0   0   0  51 183   1   1   0]
 [  0   0   6   0   0   1  19   9 212  23  30]
 [  0   0   2   0   0   0  17   3   8 307   8]
 [  0   0   0   0   2   0   3   2   6   9 323]]
------ TEST ACCURACY:  weights.48-0.55.hdf5  ------
0.580952380952
[[ 7  0  0  0  1  2  0  0  0  0  0]
 [ 0  1  0  4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  4  1  0  0  0]
 [ 0  0  0 10  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0  0]
 [ 0  0  0  0  0  0  3  7  0  0  0]
 [ 0  0  0  0  0  0  3  1  4  5  7]
 [ 1  0  0  0  0  0  0  0  2  7  5]
 [ 0  0  0  0  0  0  0  0  1  4 15]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/16/weights.43-0.50.hdf5
------ TRAIN ACCURACY:  weights.43-0.50.hdf5  ------
0.858880778589
[[213   0   0   0   0   0   1   1   0   0   0]
 [  0  96   0   4   0   0   0   0   0   0   0]
 [  1   0 148   0   0   0   2  43   2  21   3]
 [  0   0   0 194   0   0   0   0   0   1   0]
 [  0   0   0   0  63   0   0   0   1   0   1]
 [  0   0   0   0   0  67   0   0   0   2   1]
 [  0   0   0   0   0   0 188  21   0   0   1]
 [  0   0   4   0   0   0  18 187   0   3   3]
 [  0   0   3   0   0   0   9   9 134  38  42]
 [  0   0   2   0   0   0   5   1   7 226  19]
 [  0   0   0   0   3   0   0   1   5  12 249]]
------ TEST ACCURACY:  weights.43-0.50.hdf5  ------
0.55
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 0 40  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  7 21  1  0  4]
 [ 4  0  0 35  1  0  0  0  0  0  0]
 [ 1  0  0  0 23  0  2  0  0  9  0]
 [ 0  0  0  7  7  2  0  0  0  6  8]
 [ 0  0  0  0  0  0 25 19  1  0  0]
 [ 0  0  0  0  0  0 11 33  0  1  0]
 [ 0  0  2  0  0  0  9  4 20 26 24]
 [ 0  0  0  0  0  0 13  1  5 65 16]
 [ 1  0  0  0  0  0  3  0  6 50 35]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_svd_conv1d_10/17/weights.33-0.38.hdf5
------ TRAIN ACCURACY:  weights.33-0.38.hdf5  ------
0.832727272727
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  0   0 234   0   0   0   0   9   5   2   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   1  95   0   0   0   1   1   2]
 [  0   0   0   5   0  93   0   0   0   1   1]
 [  0   0  13   0   0   0 156  64   6   1   0]
 [  0   0  48   0   0   0   3 180  11   2   1]
 [  0   0   9   3   0   1   4   4 213  30  26]
 [  0   4   2  10   0   0   3   2  36 244  29]
 [  1   0   1   5   0   0   0   0  20  45 263]]
------ TEST ACCURACY:  weights.33-0.38.hdf5  ------
0.444444444444
[[ 8  2  0  5  0  0  0  0  0  0]
 [ 0  9  0  6  0  0  0  0  0  0]
 [ 1  0  7  0  0  0  0  2  4  1]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  4  0  5  4  0  2  0]
 [ 0  0  5  1  0  3  3  1  1  1]
 [ 0  0  3  0  1  0  2 14  4  6]
 [ 0  0  4  1  0  0  2 16  3  4]
 [ 0  0  0  1  1  0  0  8  4 16]]
[0.6105263157894737, 0.6133333333333333, 0.4648648648648649, 0.6368421052631579, 0.49523809523809526, 0.4, 0.7, 0.6526315789473685, 0.5611111111111111, 0.5238095238095238, 0.55, 0.4875, 0.5368421052631579, 0.6181818181818182, 0.580952380952381, 0.55, 0.4444444444444444]
0.554486922188
0.0775702744978

Process finished with exit code 0

'''