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
X_kpca = pickle.load(open("data/X_umafall_kpca.p", "rb"))
X = np.concatenate((X_svd, X_kpca), axis=1)

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

    new_dir = 'model/umafall_svd_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_svd_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.5578947368421052, 0.5466666666666666, 0.4540540540540541, 0.5526315789473685, 0.4857142857142857, 0.32, 0.7727272727272727, 0.5052631578947369, 0.5555555555555556, 0.49523809523809526, 0.59, 0.45625, 0.47368421052631576, 0.5818181818181818, 0.45714285714285713, 0.5233333333333333, 0.4777777777777778]
0.517985397896
0.0892150316863
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_svd_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/1/weights.42-0.49.hdf5
------ TRAIN ACCURACY:  weights.42-0.49.hdf5  ------
0.862880324544
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 117   0   8   0   0   0   0   0   0   0]
 [  1   0 222   0   0   0   5  13   3   3   3]
 [  1   0   0 216   0   2   0   0   0   1   0]
 [  0   0   0   0  97   0   1   0   1   0   1]
 [  0   0   0   0   0  99   0   0   0   1   0]
 [  0   0   7   0   0   0 187  42   3   1   0]
 [  0   0  28   0   0   0  26 190   1   0   0]
 [  0   0  13   0   0   1  11   8 206  24  27]
 [  2   0   9   0   0   0   9   8  11 262  19]
 [  0   0   7   2   0   2   2   2  17  10 293]]
------ TEST ACCURACY:  weights.42-0.49.hdf5  ------
0.557894736842
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  3  0  0  0  0  0  0  0]
 [ 0  0 10  0  0  0  0  4  0  1  0]
 [ 0  0  0 12  0  3  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  0 12  0  0  1  0]
 [ 0  0  0  0  0  0 11  4  0  0  0]
 [ 1  0  6  0  2  1  2  0  9  3  6]
 [ 2  0  3  0  0  0  1  4  5 19  6]
 [ 1  0  2  0  0  0  0  0 10  4 13]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/2/weights.31-0.52.hdf5
------ TRAIN ACCURACY:  weights.31-0.52.hdf5  ------
0.843912175649
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 119   0   6   0   0   0   0   0   0   0]
 [  0   0 224   0   0   0   5   2  11   5   3]
 [  0   0   0 217   0   2   0   0   0   1   0]
 [  0   0   1   0  96   0   0   0   1   1   1]
 [  0   0   0   0   0  98   0   0   1   1   0]
 [  0   0   9   0   0   0 192  32   4   3   0]
 [  0   0  37   0   0   0  40 153  11   4   0]
 [  0   2  12   0   2   1   9   4 223  31  21]
 [  0   0   8   0   1   0  12   3  21 277   8]
 [  1   0   8   3   0   3   4   0  23  31 277]]
------ TEST ACCURACY:  weights.31-0.52.hdf5  ------
0.546666666667
[[15  0  0  0  0  0  0  0  0]
 [10  5  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  1  0  0]
 [ 1  2  0  8  0  0  0  0  4]
 [ 0  0  3  0  3  9  0  0  0]
 [ 0  0 13  0  0  2  0  0  0]
 [ 0  0  0  0  0  0  6  3  6]
 [ 0  1  0  0  0  0  4 17  8]
 [ 0  0  0  0  0  0  2  1 12]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/3/weights.32-0.52.hdf5
------ TRAIN ACCURACY:  weights.32-0.52.hdf5  ------
0.85020242915
[[237   0   0   0   0   0   2   0   0   0   1]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  0   0 212   0   1   0   1   8  17   1   5]
 [  0   2   0 216   0   0   0   0   0   2   0]
 [  0   0   0   0  97   0   0   0   1   0   2]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0  16   0   0   0 159  55   8   2   0]
 [  0   0  32   0   0   0  14 187   9   1   2]
 [  0   0   6   0   2   1   3  10 214  19  35]
 [  1   0   6   0   0   0   2   6  22 244  49]
 [  1   0   2   3   1   1   0   0  12   4 311]]
------ TEST ACCURACY:  weights.32-0.52.hdf5  ------
0.454054054054
[[14  0  0  0  1  0  0  0  0  0]
 [ 1  8  0  0  0  0  0  0  3  3]
 [ 0  0 13  0  0  0  6  0  0  1]
 [ 0  9  0  5  0  0  0  0  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  9  1  3  2  0]
 [ 0  0  0  0  0  1 14  0  0  0]
 [ 3  0  0  1  0  0  1  2  3 20]
 [ 1  0  1  0  0  0  0  7  3 18]
 [ 0  0  0  0  0  0  0  7  7 16]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/4/weights.37-0.49.hdf5
------ TRAIN ACCURACY:  weights.37-0.49.hdf5  ------
0.845030425963
[[233   0   0   0   0   0   2   0   0   0   0]
 [  0 114   0  11   0   0   0   0   0   0   0]
 [  0   0 228   0   1   0   7   4   7   1   2]
 [  0   0   0 218   0   2   0   0   0   0   0]
 [  0   0   0   0  98   0   0   0   2   0   0]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0  13   0   0   0 197  15   4   1   0]
 [  0   0  49   0   0   0  49 143   4   0   0]
 [  0   1  12   1   3   1   9   1 230   8  24]
 [  1   0  11   0   2   0  12   3  30 234  42]
 [  1   0  10   4   1   3   2   0  23   2 289]]
------ TEST ACCURACY:  weights.37-0.49.hdf5  ------
0.552631578947
[[16  0  0  1  2  0  0  0  0  0  1]
 [ 7  3  0  5  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  0  0  2  0  1]
 [13  0  0  1  0  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  3  0  0  0 19  2  1  0  0]
 [ 0  0  2  0  0  0  6  7  0  0  0]
 [ 3  0  2  0  3  1  0  0 13  0  8]
 [ 1  0  0  0  0  0  1  0  3 11  9]
 [ 0  0  0  0  0  0  0  0  6  1 23]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/5/weights.35-0.54.hdf5
------ TRAIN ACCURACY:  weights.35-0.54.hdf5  ------
0.841568627451
[[238   0   0   0   0   0   1   0   0   1   0]
 [  0 139   0   1   0   0   0   0   0   0   0]
 [  0   0 201   0   0   0   4  17  23   3   2]
 [  4   1   0 211   0   2   0   0   0   2   0]
 [  0   0   0   0  97   0   0   0   2   1   0]
 [  0   0   0   0   0  99   0   0   1   0   0]
 [  0   0  15   0   0   0 164  47   6   8   0]
 [  0   0  33   0   0   0  26 174   9   3   0]
 [  0   1   8   0   0   0   6   7 251  27  20]
 [  1   0   5   0   0   0   2   7  29 307   9]
 [  1   0   0   1   0   1   0   0  50  17 265]]
------ TEST ACCURACY:  weights.35-0.54.hdf5  ------
0.485714285714
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  1  0  0]
 [ 0  9  0  6  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  4  0  1  6  0  2  2  0]
 [ 0  0  1  0  0  4  9  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 10  0  0  2  1 10  6  1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/6/weights.44-0.52.hdf5
------ TRAIN ACCURACY:  weights.44-0.52.hdf5  ------
0.863723608445
[[248   0   0   0   0   0   2   0   0   0   0]
 [  0 128   0  12   0   0   0   0   0   0   0]
 [  1   0 245   0   0   0   4   2   7   0   1]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   0   0  97   0   0   0   1   1   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0  23   0   0   0 204  23   5   0   0]
 [  0   0  50   0   0   0  23 173   4   0   0]
 [  1   1  21   1   0   1   8   3 237  14  23]
 [  0   0  19   0   0   0   7   2  20 284  18]
 [  0   0  16   4   0   3   1   0  28   3 300]]
------ TEST ACCURACY:  weights.44-0.52.hdf5  ------
0.32
[[5 0 0 0 0 0 0 0 0]
 [0 4 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [1 1 0 0 3 2 1 2 0]
 [2 1 1 0 0 0 1 1 4]
 [0 3 0 1 0 0 5 1 0]
 [0 3 0 0 0 0 4 0 3]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/7/weights.49-0.55.hdf5
------ TRAIN ACCURACY:  weights.49-0.55.hdf5  ------
0.880157170923
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 136   0   4   0   0   0   0   0   0   0]
 [  0   0 206   0   0   0   4  24  11   4   1]
 [  0   0   0 218   0   2   0   0   0   0   0]
 [  0   0   0   0  85   0   0   0   0   0   0]
 [  0   0   0   0   0  85   0   0   0   0   0]
 [  0   0   3   0   0   0 207  22   1   2   0]
 [  0   0  17   0   0   0  19 201   6   2   0]
 [  1   0   6   0   1   1   8   6 262  22  13]
 [  2   0   6   0   1   0   9   5  25 301  11]
 [  0   0   3   1   1   0   2   0  46  11 301]]
------ TEST ACCURACY:  weights.49-0.55.hdf5  ------
0.772727272727
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 13  0  0  0  0  1  1  0  0]
 [ 0  0 11  0  0  0  0  0  3  1]
 [ 2  0  0 11  0  0  0  2  0  0]
 [ 1  0  0  0  7  0  0  3  0  4]
 [ 0  2  0  0  0 15  2  1  0  0]
 [ 0  0  0  0  0  2 13  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/8/weights.50-0.54.hdf5
------ TRAIN ACCURACY:  weights.50-0.54.hdf5  ------
0.88671875
[[238   0   0   0   0   0   0   2   0   0   0]
 [  0 137   0   3   0   0   0   0   0   0   0]
 [  0   0 228   0   0   0   3   4   9   4   2]
 [  2   0   0 215   0   2   0   0   0   1   0]
 [  0   0   0   0  90   0   0   0   0   0   0]
 [  0   0   0   0   0  89   0   0   0   1   0]
 [  0   0  11   0   0   1 175  49   2   2   0]
 [  0   0  30   0   0   0   3 210   1   1   0]
 [  0   0   8   0   1   1   5   7 247  28  23]
 [  2   0   6   0   0   0   6   4   9 327   6]
 [  3   0   1   1   0   1   1   0  20  24 314]]
------ TEST ACCURACY:  weights.50-0.54.hdf5  ------
0.505263157895
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  1  0  2  0  1]
 [ 2  5  0  7  0  0  0  0  0  0  1]
 [ 0  0  0  0  5  2  0  0  2  0  1]
 [ 0  0  0  0  0  4  0  0  1  2  3]
 [ 0  0  1  0  0  0  3 11  0  0  0]
 [ 0  0 11  0  1  0  0  3  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/9/weights.50-0.50.hdf5
------ TRAIN ACCURACY:  weights.50-0.50.hdf5  ------
0.900202020202
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 119   0   5   0   0   0   0   0   0   1]
 [  0   0 223   0   0   0   8   9   5   4   1]
 [  0   0   0 219   0   1   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   3   0   0   0 221  14   0   2   0]
 [  0   0  26   0   0   0  20 196   0   3   0]
 [  0   0   8   0   2   0  11   5 219  33  12]
 [  1   0   4   0   0   0   7   2   9 301   6]
 [  0   0   4   1   0   2   1   0  20  15 292]]
------ TEST ACCURACY:  weights.50-0.50.hdf5  ------
0.555555555556
[[14  0  0  0  0  0  1  0  0  0  0]
 [ 0 14  0  1  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0  0]
 [ 0 11  0  4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0 11  2  1  0  0]
 [ 0  0  0  0  0  0  4  6  1  3  1]
 [ 0  0  2  0  3  1  1  0  9  5  9]
 [ 0  1  0  0  0  0  2  0  5 13  9]
 [ 0  3  1  0  0  1  0  0  6  5 14]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/10/weights.32-0.53.hdf5
------ TRAIN ACCURACY:  weights.32-0.53.hdf5  ------
0.832549019608
[[237   0   0   0   0   0   3   0   0   0   0]
 [  0 136   0   3   0   0   1   0   0   0   0]
 [  0   0 206   0   0   0   4  26   8   3   3]
 [  0   2   0 213   0   2   0   0   1   2   0]
 [  0   0   0   0  82   0   0   0   2   1   0]
 [  0   0   0   0   0  82   0   0   0   1   2]
 [  0   0   8   0   0   0 161  66   4   1   0]
 [  0   0  24   0   0   0  15 201   4   1   0]
 [  0   0  15   0   1   1  12   6 239  21  25]
 [  0   0  10   0   0   0  17  11  30 267  25]
 [  0   0  11   3   0   2   6   2  30  12 299]]
------ TEST ACCURACY:  weights.32-0.53.hdf5  ------
0.495238095238
[[12  0  1  1  1  0  0  0  0  0]
 [ 0  1  0  2  0  0  0  8  1  3]
 [ 0  0 13  0  2  0  0  0  0  0]
 [ 0  0  0 10  5  0  0  0  0  0]
 [ 1  0  0  0  8  0  0  1  1  4]
 [ 0  0  0  0  0  6  9  0  0  0]
 [ 0  5  0  0  0  2  2  6  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/11/weights.15-0.53.hdf5
------ TRAIN ACCURACY:  weights.15-0.53.hdf5  ------
0.748336594912
[[234   0   0   0   0   0   3   0   0   3   0]
 [  0 129   0   9   0   0   0   0   0   2   0]
 [  0   0 172   0   5   0  11  27  25   4   1]
 [  4   4   0 204   1   1   0   0   1   2   3]
 [  0   0   0   0  83   0   0   0   5   2   0]
 [  1   0   0   0   6  75   0   0   2   5   1]
 [  0   0   2   0   1   0 185  43   8   1   0]
 [  0   0  24   0   1   0  45 159  11   5   0]
 [  0   1   3   0   3   1  14   8 233  45  12]
 [  1   0   5   0   0   0  21   8  61 252  12]
 [  1   0   7   3   5   0   9   2  78  74 186]]
------ TEST ACCURACY:  weights.15-0.53.hdf5  ------
0.59
[[15  0  0  0  0  0  0  0  0]
 [ 0 11  0  1  0  2  0  3  3]
 [ 0  0 15  0  0  0  0  0  0]
 [ 0  0  0 10  0  0  0  0  0]
 [ 2  0  0  1  7  0  0  0  0]
 [ 0  1  0  0  0  1 12  1  0]
 [ 0  6  0  0  0  0  0  9  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/12/weights.12-0.51.hdf5
------ TRAIN ACCURACY:  weights.12-0.51.hdf5  ------
0.697795591182
[[237   0   0   0   0   0   3   0   0   0   0]
 [  7 120   0   9   1   0   0   0   0   3   0]
 [  1   0 179   0   7   0   4  23  23   4   4]
 [ 28   9   0 177   0   1   0   0   1   0   4]
 [  1   0   0   0  79   0   0   0   2   3   0]
 [  3   0   0   0   7  68   0   0   1   1   0]
 [  0   0  22   0   4   0 148  55   8   2   1]
 [  0   0  40   0   0   1  40 143  16   5   0]
 [  7   0   9   0   6   3  10  10 210  24  26]
 [ 10   0   9   0   5   1   9  13  65 193  40]
 [  6   0  15   3  10   3   3   2  84  37 187]]
------ TEST ACCURACY:  weights.12-0.51.hdf5  ------
0.45625
[[15  0  0  0  0  0  0  0  0  0]
 [ 2 13  0  1  0  0  0  3  1  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [14  0  0  0  1  0  0  0  0  0]
 [16  0  3  0  1  0  0  0  0  0]
 [ 0  3  0  0  0 11  1  0  0  0]
 [ 0  1  0  0  0  5  9  0  0  0]
 [ 3  0  0  0  1  0  1  3  3  4]
 [ 2  3  0  0  0  0  0  4  3  3]
 [ 2  0  0  1  1  0  0  8  0  3]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/13/weights.20-0.52.hdf5
------ TRAIN ACCURACY:  weights.20-0.52.hdf5  ------
0.76484375
[[244   0   0   0   0   0   0   0   1   0   0]
 [  0 128   0   4   0   0   0   0   0   3   0]
 [  0   0 197   0   3   0   8  19  26   6   1]
 [  7   4   0 211   0   4   0   0   1   2   1]
 [  1   0   0   0  92   2   0   0   2   2   1]
 [  1   0   0   0   0  96   0   0   1   2   0]
 [  0   0   4   0   1   0 193  42   7   3   0]
 [  0   0  25   0   0   1  57 156  11   5   0]
 [  1   0   5   0   2   1  12   4 234  34   7]
 [  1   0   4   0   2   2  10   5  59 251   6]
 [  1   1   7   3   1   7   5   3  88  73 156]]
------ TEST ACCURACY:  weights.20-0.52.hdf5  ------
0.473684210526
[[ 5  0  0  0  2  0  0  3  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  1  0  0]
 [ 0  2  0  3  0  0  0  0  0]
 [ 0  0  1  0  2  0  2  0  0]
 [ 0  0  0  0  3  2  0  0  0]
 [ 0  0  0  0  0  0  9  8  3]
 [ 0  2  0  0  0  0  1 12  5]
 [ 0  1  0  0  0  0 12  4  3]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/14/weights.42-0.52.hdf5
------ TRAIN ACCURACY:  weights.42-0.52.hdf5  ------
0.861538461538
[[248   0   0   0   0   0   2   0   0   0   0]
 [  0 128   0  12   0   0   0   0   0   0   0]
 [  0   0 220   0   0   0   4  11  14   2   4]
 [  0   0   0 232   0   2   0   0   0   1   0]
 [  0   0   0   0  97   0   0   0   1   1   1]
 [  0   0   0   0   0  99   0   0   1   0   0]
 [  0   0   8   0   0   0 212  30   4   1   0]
 [  0   0  26   0   0   0  38 175  11   0   0]
 [  0   0   6   0   0   1  12   5 256  25  10]
 [  2   0   5   0   0   0   7   3  32 290   6]
 [  0   0   1   2   0   2   1   1  45  20 283]]
------ TEST ACCURACY:  weights.42-0.52.hdf5  ------
0.581818181818
[[ 3  1  0  0  0  1  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  2  0  8  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  2  0  8  0  0  0]
 [ 0  0  1  1  0  2  1  0]
 [ 0  0  0  2  0  0 11  2]
 [ 0  0  0  1  0  1  2  6]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/15/weights.44-0.52.hdf5
------ TRAIN ACCURACY:  weights.44-0.52.hdf5  ------
0.885882352941
[[243   0   0   0   0   0   2   0   0   0   0]
 [  0 130   0   5   0   0   0   0   0   0   0]
 [  0   0 230   0   1   0   2  16   8   1   2]
 [  0   0   0 222   0   2   0   0   0   1   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   2   0   0   0 209  30   2   2   0]
 [  0   0  25   0   0   0  31 193   1   0   0]
 [  0   2   6   1   1   1  12   8 229  26  14]
 [  1   0   4   0   1   0   8   7  10 308   6]
 [  0   0   2   2   3   5   2   1  22  13 295]]
------ TEST ACCURACY:  weights.44-0.52.hdf5  ------
0.457142857143
[[4 0 0 0 6 0 0 0 0 0]
 [0 0 0 5 0 0 0 0 0 0]
 [0 0 0 0 0 4 1 0 0 0]
 [0 2 0 8 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 9 1 0 0 0]
 [0 0 0 0 0 4 6 0 0 0]
 [0 0 0 0 0 1 2 7 6 4]
 [2 0 0 0 0 0 0 2 7 4]
 [0 1 0 0 0 0 0 3 9 7]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/16/weights.49-0.50.hdf5
------ TRAIN ACCURACY:  weights.49-0.50.hdf5  ------
0.894890510949
[[211   0   0   2   0   0   1   1   0   0   0]
 [  0  89   0  10   0   0   0   0   0   0   1]
 [  0   0 197   0   0   0   1  10   6   3   3]
 [  0   0   0 195   0   0   0   0   0   0   0]
 [  0   0   0   0  64   0   0   0   0   0   1]
 [  0   0   0   0   0  68   0   0   1   0   1]
 [  0   0   9   0   0   0 170  24   4   3   0]
 [  0   0  24   0   0   0  10 179   0   0   2]
 [  0   0   3   0   0   0   2   4 199  11  16]
 [  2   0   4   0   0   0   3   3  12 221  15]
 [  0   0   2   4   0   1   0   1   9   7 246]]
------ TEST ACCURACY:  weights.49-0.50.hdf5  ------
0.523333333333
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 4 17  0  1  0  0  0  0  3  7  8]
 [ 0  0 37  0  0  0  1  4  1  1  1]
 [ 4  0  0 36  0  0  0  0  0  0  0]
 [ 0  0  1  0 19  0  1  0  5  6  3]
 [ 0  0  0  2  2  0  0  0 11  7  8]
 [ 0  0  0  0  0  0 20 23  0  2  0]
 [ 0  0  7  0  0  0  5 32  0  1  0]
 [ 0  1  3  0  0  0  9  2 35 19 16]
 [ 1  1  2  0  0  0 10  5 12 44 25]
 [ 1  1  0  3  0  0  3  0 16 37 34]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_kpca_conv1d_10/17/weights.44-0.42.hdf5
------ TRAIN ACCURACY:  weights.44-0.42.hdf5  ------
0.877171717172
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  0   0 232   0   0   0   2  11   4   0   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   9   0   0   0 199  28   3   1   0]
 [  0   0  31   0   0   0   7 205   2   0   0]
 [  1   0   8   0   1   1   8   4 228  19  20]
 [  1   0  10   0   0   1  10   3  18 248  39]
 [  1   0   5   3   0   1   3   0  26  20 276]]
------ TEST ACCURACY:  weights.44-0.42.hdf5  ------
0.477777777778
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  2  0 13  0  0  0  0  0  0  0]
 [ 1  0  6  0  1  0  0  1  4  1  1]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 10  4  0  1  0]
 [ 0  0  4  0  0  0  3  8  0  0  0]
 [ 1  0  3  0  3  0  0  3 14  5  1]
 [ 0  0  4  0  1  0  3  3  8  4  7]
 [ 0  0  2  2  1  1  0  0  9  3 12]]
[0.5578947368421052, 0.5466666666666666, 0.4540540540540541, 0.5526315789473685, 0.4857142857142857, 0.32, 0.7727272727272727, 0.5052631578947369, 0.5555555555555556, 0.49523809523809526, 0.59, 0.45625, 0.47368421052631576, 0.5818181818181818, 0.45714285714285713, 0.5233333333333333, 0.4777777777777778]
0.517985397896
0.0892150316863

Process finished with exit code 0

'''