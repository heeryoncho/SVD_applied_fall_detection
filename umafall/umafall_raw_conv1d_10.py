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

X = pickle.load(open("data/X_umafall_raw.p", "rb"))
y = pickle.load(open("data/y_umafall_raw.p", "rb"))

n_classes = 11
signal_rows = 450 * 3
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
    print X_train.shape # (2465, 1350)

    # input layer
    input_signal = Input(shape=(signal_rows, 1))
    print K.int_shape(input_signal) # (None, 1350, 1)

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

    new_dir = 'model/umafall_raw_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_raw_conv1d_10/' + str(i+1) + '/'
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
[0.4842105263157895, 0.5066666666666667, 0.6972972972972973, 0.5947368421052631, 0.5714285714285714, 0.64, 0.7545454545454545, 0.7157894736842105, 0.6833333333333333, 0.6190476190476191, 0.55, 0.55, 0.5578947368421052, 0.5454545454545454, 0.34285714285714286, 0.7433333333333333, 0.24444444444444444]
0.576531763962
0.131081402478
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_raw_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/1/weights.09-0.51.hdf5
------ TRAIN ACCURACY:  weights.09-0.51.hdf5  ------
0.733062880325
[[201   0   4   0   0   0  23   7   0   0   5]
 [  3 114   1   6   0   0   1   0   0   0   0]
 [  0   0 182   0   0   0  25  43   0   0   0]
 [  3   4   0 205   3   4   1   0   0   0   0]
 [ 22   0  12   0  47  12   0   7   0   0   0]
 [ 14   0   7   1   5  73   0   0   0   0   0]
 [  4   0   1   0   0   0 135  47  11   4  38]
 [  3   0  41   0   0   0  19 163   0   4  15]
 [  0   0   7   0   0   0   6   0 194  34  49]
 [  0   0   0   0   0   0  11   4  14 216  75]
 [  0   0   0   0   0   0   2   3  23  30 277]]
------ TEST ACCURACY:  weights.09-0.51.hdf5  ------
0.484210526316
[[10  0  0  0  0  5  0  0  0  0]
 [ 0 12  0  2  0  0  0  0  0  1]
 [ 0  0 13  0  0  0  2  0  0  0]
 [ 0  0  0 12  3  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  7  0  7  1]
 [ 0  0  0  0  0  0 11  0  4  0]
 [ 0  0  0  0  0  9  0 16  0  5]
 [ 0  0  0  0  0  0  0  1 18 21]
 [ 0  0  0  0  0  0  0  0 30  0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/2/weights.43-0.59.hdf5
------ TRAIN ACCURACY:  weights.43-0.59.hdf5  ------
0.853493013972
[[236   0   2   2   0   0   0   0   0   0   0]
 [  9   1   0 115   0   0   0   0   0   0   0]
 [  0   0 236   0   0   0   0  12   2   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   7   0  93   0   0   0   0   0   0]
 [  0   0   5   0   1  94   0   0   0   0   0]
 [  0   0   3   0   0   0 222   8   1   0   6]
 [  1   0  29   0   0   0   3 209   0   1   2]
 [  1   0   3   0   0   0   2   2 246  24  27]
 [  1   0   0   0   0   0   6   1  11 284  27]
 [  0   0   0   0   0   0   1   2   9  41 297]]
------ TEST ACCURACY:  weights.43-0.59.hdf5  ------
0.506666666667
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  8  0  0  7  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  4  0 11  0  0  0  0]
 [ 0  0 12  0  3  0  0  0  0]
 [ 0  0  0  0  0  0  5  4  6]
 [ 0  0  0  0  0  0  0 15 15]
 [ 0  0  0  0  0  0  4  4  7]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/3/weights.48-0.64.hdf5
------ TRAIN ACCURACY:  weights.48-0.64.hdf5  ------
0.895141700405
[[238   0   2   0   0   0   0   0   0   0   0]
 [  0 116   0   9   0   0   0   0   0   0   0]
 [  0   0 219   0   0   0   0  21   5   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   7   0  93   0   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   3   0   0   0 221   5   7   1   3]
 [  0   0  31   0   0   0   4 205   2   2   1]
 [  1   0   4   0   0   0   2   1 236  33  13]
 [  0   0   0   0   0   0   2   0  13 306   9]
 [  0   0   0   0   0   0   1   0  13  59 262]]
------ TEST ACCURACY:  weights.48-0.64.hdf5  ------
0.697297297297
[[15  0  0  0  0  0  0  0  0]
 [ 5  7  0  2  0  1  0  0  0]
 [ 0  0 14  0  0  6  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  9  1  3  0  2]
 [ 0  0  0  0 14  1  0  0  0]
 [ 0  0  0  0  2  0 16  0 12]
 [ 0  0  0  0  0  0  1 28  1]
 [ 0  0  0  0  0  0  4  2 24]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/4/weights.50-0.70.hdf5
------ TRAIN ACCURACY:  weights.50-0.70.hdf5  ------
0.907505070994
[[233   0   2   0   0   0   0   0   0   0   0]
 [  0 124   0   0   0   0   0   1   0   0   0]
 [  0   0 242   0   0   0   0   8   0   0   0]
 [  0   5   0 215   0   0   0   0   0   0   0]
 [  0   0   7   0  92   1   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   2   0   0   0 195  20   6   1   6]
 [  0   0  31   0   0   0   0 211   0   2   1]
 [  2   0   0   0   0   0   1   0 253  27   7]
 [  0   0   0   0   0   0   7   0   8 306  14]
 [  0   0   0   0   0   0   0   3   9  52 271]]
------ TEST ACCURACY:  weights.50-0.70.hdf5  ------
0.594736842105
[[20  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0]
 [ 0 12  0  3  0  0  0  0  0]
 [ 3  0  0  0 11  0  0  6  5]
 [ 0  0  2  0  4  8  0  0  1]
 [ 0  0  5  0  0  0  7 15  3]
 [ 0  0  0  0  0  0  2 23  0]
 [ 0  0  0  0  2  0 11  5 12]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/5/weights.39-0.66.hdf5
------ TRAIN ACCURACY:  weights.39-0.66.hdf5  ------
0.843921568627
[[238   0   2   0   0   0   0   0   0   0   0]
 [  0 139   0   0   0   0   0   1   0   0   0]
 [  0   0 236   0   0   0   0  11   3   0   0]
 [  0   0   1 217   0   2   0   0   0   0   0]
 [  1   0   7   0  71  21   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   6   0   0   0 174  29  11  16   4]
 [  1   0  57   0   0   0   1 179   1   4   2]
 [  1   0   4   0   0   0   1   1 252  45  16]
 [  0   0   0   0   0   0  10   2  21 325   2]
 [  0   0   0   0   0   0   0   0  15  94 226]]
------ TEST ACCURACY:  weights.39-0.66.hdf5  ------
0.571428571429
[[15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  2  0  0]
 [ 0  1  0 14  0  0  0  0]
 [ 0  0  0  0 10  0  0  5]
 [ 0  0  5  0  1  8  1  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  1  0  1  3 25  0]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/6/weights.41-0.65.hdf5
------ TRAIN ACCURACY:  weights.41-0.65.hdf5  ------
0.881765834933
[[245   0   2   2   0   0   0   0   0   0   1]
 [  1 139   0   0   0   0   0   0   0   0   0]
 [  3   0 253   0   0   0   0   2   2   0   0]
 [  0   1   0 234   0   0   0   0   0   0   0]
 [  0   0   7   0  93   0   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   5   0   0   0 223  11   8   4   4]
 [  0   0  59   0   1   0   2 184   1   2   1]
 [  2   0   5   0   0   0   2   0 247  35  19]
 [  0   0   0   0   0   0   8   0  17 299  26]
 [  0   0   0   0   0   0   2   0  16  52 285]]
------ TEST ACCURACY:  weights.41-0.65.hdf5  ------
0.64
[[5 0 0 0 0 0]
 [0 5 0 0 0 0]
 [0 5 5 0 0 0]
 [0 0 0 9 0 1]
 [0 0 0 5 5 0]
 [0 0 0 0 7 3]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/7/weights.49-0.58.hdf5
------ TRAIN ACCURACY:  weights.49-0.58.hdf5  ------
0.866404715128
[[233   4   2   1   0   0   0   0   0   0   0]
 [  0 139   0   0   0   0   0   1   0   0   0]
 [  0   0 246   0   0   0   1   3   0   0   0]
 [  0  10   0 210   0   0   0   0   0   0   0]
 [  0   0   7   0  75   2   0   1   0   0   0]
 [  0   0   5   0   0  80   0   0   0   0   0]
 [  0   0   6   0   0   0 188  25   1   8   7]
 [  0   0  52   0   0   0   0 188   1   2   2]
 [  0   0   4   0   0   0   3   1 231  43  38]
 [  0   0   0   0   0   0   9   1  10 321  19]
 [  0   0   0   0   0   0   1   0  10  60 294]]
------ TEST ACCURACY:  weights.49-0.58.hdf5  ------
0.754545454545
[[14  1  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 0  0  1  9  4  1  0  0  0  0]
 [ 3  0  0  0 11  1  0  0  0  0]
 [ 1  0  0  0  8  6  0  0  0  0]
 [ 0  0  0  0  0  0 18  0  0  2]
 [ 0  0  3  0  0  0  0 10  2  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/8/weights.45-0.66.hdf5
------ TRAIN ACCURACY:  weights.45-0.66.hdf5  ------
0.894140625
[[238   0   2   0   0   0   0   0   0   0   0]
 [  0 139   0   0   0   0   1   0   0   0   0]
 [  0   0 247   0   0   0   0   3   0   0   0]
 [  0   2   0 218   0   0   0   0   0   0   0]
 [  0   0   5   0  85   0   0   0   0   0   0]
 [  0   0   5   0   0  85   0   0   0   0   0]
 [  0   0   3   0   0   0 222   5   1   1   8]
 [  0   0  50   0   0   0   1 190   0   2   2]
 [  1   0   5   0   0   0   1   0 243  44  26]
 [  0   0   0   0   0   0   9   0   3 324  24]
 [  0   0   0   0   0   0   1   0   8  58 298]]
------ TEST ACCURACY:  weights.45-0.66.hdf5  ------
0.715789473684
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  1  0]
 [ 0  5  0 10  0  0  0  0  0]
 [ 0  0  2  1  2  4  0  0  1]
 [ 0  0  0  0  2  8  0  0  0]
 [ 0  0  0  0  0  0  9  6  0]
 [ 0  0  5  0  0  0  0 10  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/9/weights.44-0.67.hdf5
------ TRAIN ACCURACY:  weights.44-0.67.hdf5  ------
0.866262626263
[[240   0   0   0   0   0   0   0   0   0   0]
 [  2 120   0   3   0   0   0   0   0   0   0]
 [  5   0 232   0   0   0   0  11   2   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  5   0   2   0  93   0   0   0   0   0   0]
 [  0   0   5   5   0  90   0   0   0   0   0]
 [  1   0   4   0   0   0 194  18   9   4  10]
 [  8   0  48   1   2   0   3 177   1   3   2]
 [  2   0   4   0   0   0   1   0 233  30  20]
 [  0   0   0   0   0   0   2   4  28 275  21]
 [  0   0   0   0   0   0   3   0  19  43 270]]
------ TEST ACCURACY:  weights.44-0.67.hdf5  ------
0.683333333333
[[15  0  0  0  0  0  0  0  0]
 [ 0 14  0  1  0  0  0  0  0]
 [ 0  0 12  0  1  1  1  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  7  0  1  2  5]
 [ 0  0  4  0  9  0  0  0  2]
 [ 0  0  0  0  0  0 23  1  6]
 [ 0  0  0  0  0  0  0 28  2]
 [ 0  0  0  0  0  0  2 19  9]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/10/weights.22-0.53.hdf5
------ TRAIN ACCURACY:  weights.22-0.53.hdf5  ------
0.78862745098
[[199  20   1   0   0   0   0  20   0   0   0]
 [  1 139   0   0   0   0   0   0   0   0   0]
 [  2   0 200   0   0   0   0  48   0   0   0]
 [  2   2   0 214   2   0   0   0   0   0   0]
 [  2   0   8   0  73   2   0   0   0   0   0]
 [  0   0   5   0   0  80   0   0   0   0   0]
 [  4   0   3   0   0   0 136  49  29  11   8]
 [  7   0  36   0   0   1   7 187   1   4   2]
 [  0   0   5   0   0   0   2   0 254  45  14]
 [  0   0   0   0   0   0   2   2  23 292  41]
 [  0   0   0   0   0   0   3   5  43  77 237]]
------ TEST ACCURACY:  weights.22-0.53.hdf5  ------
0.619047619048
[[15  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0]
 [ 8  3  0  0  4  0  0  0  0]
 [ 4  0  0  1 10  0  0  0  0]
 [ 0  0  0  0  0 10  2  1  2]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/11/weights.46-0.68.hdf5
------ TRAIN ACCURACY:  weights.46-0.68.hdf5  ------
0.881409001957
[[236   0   2   2   0   0   0   0   0   0   0]
 [  0 139   1   0   0   0   0   0   0   0   0]
 [  0   0 238   0   0   0   0   7   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   7   0  83   0   0   0   0   0   0]
 [  0   0   5   0   0  85   0   0   0   0   0]
 [  0   0   2   0   0   0 207  12   6   0  13]
 [  0   0  42   0   0   0   3 194   0   2   4]
 [  1   0   5   0   0   0   0   0 253  21  40]
 [  0   0   0   0   0   0   6   0  20 284  50]
 [  0   0   0   0   0   0   0   0  12  40 313]]
------ TEST ACCURACY:  weights.46-0.68.hdf5  ------
0.55
[[15  0  0  0  0  0  0]
 [ 0 20  0  0  0  0  0]
 [ 0  0 15  0  0  0  0]
 [ 0  2  0  4  4  0  0]
 [ 0  0  0  9  1  0  0]
 [ 0  2  0  0  0  0 13]
 [ 0 15  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/12/weights.26-0.62.hdf5
------ TRAIN ACCURACY:  weights.26-0.62.hdf5  ------
0.840881763527
[[236   0   1   2   0   0   1   0   0   0   0]
 [  0 130   1   9   0   0   0   0   0   0   0]
 [  0   0 237   0   0   0   0   8   0   0   0]
 [  0   4   0 215   1   0   0   0   0   0   0]
 [  0   0   7   0  77   1   0   0   0   0   0]
 [  1   0   0   0   0  79   0   0   0   0   0]
 [  1   0   4   0   0   0 183  23   9   3  17]
 [  0   0  71   0   1   0  16 150   0   4   3]
 [  1   0   5   0   0   0   0   0 244  33  22]
 [  0   0   0   0   0   0   9   0  27 289  20]
 [  0   0   0   0   0   0   3   0  15  74 258]]
------ TEST ACCURACY:  weights.26-0.62.hdf5  ------
0.55
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 20  0  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [11  0  0  0  1  3  0  0  0  0  0]
 [ 4  3  5  6  0  2  0  0  0  0  0]
 [ 0  0  0  0  0  0 15  0  0  0  0]
 [ 0  0 10  0  0  0  4  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  0 15  0]
 [ 0  0  0  0  0  0  0  0 11  4  0]
 [ 0  0  0  0  0  0  0  0  0  0 15]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/13/weights.17-0.64.hdf5
------ TRAIN ACCURACY:  weights.17-0.64.hdf5  ------
0.81953125
[[243   0   0   0   0   2   0   0   0   0   0]
 [  0 131   1   3   0   0   0   0   0   0   0]
 [  0   0 241   0   0   0   0  19   0   0   0]
 [  0   4   0 222   1   3   0   0   0   0   0]
 [  2   0   7   0  59  32   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   4   0   0   0 177  37   7   3  22]
 [  0   0  69   0   0   0   5 173   1   5   2]
 [  2   0   5   0   0   0   0   0 224  41  28]
 [  0   0   0   0   0   0   3   2  15 268  52]
 [  0   0   0   0   0   0   0   1  23  56 265]]
------ TEST ACCURACY:  weights.17-0.64.hdf5  ------
0.557894736842
[[ 7  0  3  0  0  0  0  0  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  1  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  0  0  1  0  4  0  0]
 [ 0  0  2  0  0  3  0  0  0]
 [ 0  0  0  0  0  0 14  5  1]
 [ 0  0  0  0  0  0  0  1 19]
 [ 0  0  0  0  0  0  7  0 13]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/14/weights.44-0.63.hdf5
------ TRAIN ACCURACY:  weights.44-0.63.hdf5  ------
0.880769230769
[[247   0   2   1   0   0   0   0   0   0   0]
 [  0 133   0   6   0   0   0   1   0   0   0]
 [  0   0 243   0   0   0   0  12   0   0   0]
 [  0   0   1 234   0   0   0   0   0   0   0]
 [  0   0  10   0  88   2   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   2   0   0   0 233  14   0   0   6]
 [  0   0  37   0   0   0   3 204   0   3   3]
 [  2   0   2   0   0   0   2   3 238  29  39]
 [  0   0   0   0   0   0   3   0  13 276  53]
 [  0   0   0   0   0   0   0   0  11  45 299]]
------ TEST ACCURACY:  weights.44-0.63.hdf5  ------
0.545454545455
[[5 0 0 0 0 0 0]
 [0 2 0 8 0 0 0]
 [0 0 0 0 0 0 0]
 [0 4 0 6 0 0 0]
 [0 0 1 0 3 0 1]
 [0 0 0 3 0 5 7]
 [0 0 0 0 0 1 9]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/15/weights.14-0.56.hdf5
------ TRAIN ACCURACY:  weights.14-0.56.hdf5  ------
0.746666666667
[[202   3   3   0   0   0   0  37   0   0   0]
 [  1 129   1   4   0   0   0   0   0   0   0]
 [  0   0 196   0   0   0   2  62   0   0   0]
 [  0   0   0 224   1   0   0   0   0   0   0]
 [  8   0  14   0  67  10   0   1   0   0   0]
 [  3   0   6   0   1  90   0   0   0   0   0]
 [  5   0   1   0   0   0 152  44  11   9  23]
 [  0   0  60   0   0   0  17 159   0   5   9]
 [  0   0   5   0   0   0   9   1 210  27  48]
 [  0   0   0   0   0   0  13   2  30 227  73]
 [  0   0   0   0   0   0   4   9  22  62 248]]
------ TEST ACCURACY:  weights.14-0.56.hdf5  ------
0.342857142857
[[10  0  0  0  0  0  0  0  0  0]
 [ 2  0  0  3  0  0  0  0  0  0]
 [ 4  0  1  0  0  0  0  0  0  0]
 [ 2  1  0  5  2  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  1  2  0  6  1]
 [ 0  0  0  0  0  4  6  0  0  0]
 [ 0  0  0  0  0  0  0  9  4  7]
 [ 0  0  0  0  0  0  0  0  0 15]
 [ 0  0  0  0  0  0  0  0 16  4]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/16/weights.44-0.57.hdf5
------ TRAIN ACCURACY:  weights.44-0.57.hdf5  ------
0.892457420925
[[210   2   2   0   0   0   1   0   0   0   0]
 [  1  97   0   2   0   0   0   0   0   0   0]
 [  0   0 215   0   0   0   0   0   5   0   0]
 [  0   0   0 195   0   0   0   0   0   0   0]
 [  0   0   2   0  63   0   0   0   0   0   0]
 [  0   0   5   0   3  62   0   0   0   0   0]
 [  0   0   1   0   0   0 197   4   4   0   4]
 [  0   0  33   0   0   0   4 174   0   2   2]
 [  0   0   5   0   0   0   3   2 180  33  12]
 [  0   0   0   0   0   0  10   0   3 234  13]
 [  0   0   0   0   0   0   4   0   9  50 207]]
------ TEST ACCURACY:  weights.44-0.57.hdf5  ------
0.743333333333
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 0 31  0  9  0  0  0  0  0  0  0]
 [ 0  0 34  0  0  0  0 11  0  0  0]
 [ 0  3  0 35  0  1  0  1  0  0  0]
 [ 2  0  5  0 23  4  0  0  1  0  0]
 [ 0  0  0  0 11 18  0  0  1  0  0]
 [ 0  0  2  0  0  0 38  5  0  0  0]
 [ 0  0 12  0  0  0  4 27  2  0  0]
 [ 2  0  0  0  0  0  0  0 73  0 10]
 [ 3  1  0  0  0  0  2  1 12 73  8]
 [ 0  0  0  0  0  0 15  0 17  9 54]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_conv1d_10/17/weights.02-0.85.hdf5
------ TRAIN ACCURACY:  weights.02-0.85.hdf5  ------
0.586262626263
[[206  12  13   7   0   0   0   2   0   0   0]
 [  5  12   0 108   0   0   0   0   0   0   0]
 [ 33   0 143   0   0   0   0  74   0   0   0]
 [  6   6   1 204   0   3   0   0   0   0   0]
 [ 42   0  11   4  16  11   0  16   0   0   0]
 [ 41   0   8  14   9  26   0   2   0   0   0]
 [  5   0  10   0   0   0  65  43  56  26  35]
 [ 12   0  54   0   0   0  14 141   0  17   7]
 [  7   0   0   0   0   0   0   0 217  43  23]
 [  2   0   0   0   0   0   0   4  44 258  22]
 [  0   0   0   0   0   0   0   4  42 126 163]]
------ TEST ACCURACY:  weights.02-0.85.hdf5  ------
0.244444444444
[[ 0  3  0  0  0 12  0  0  0]
 [ 0  5  0 10  0  0  0  0  0]
 [ 0  0  8  0  0  7  0  0  0]
 [ 0  1  0 14  0  0  0  0  0]
 [ 0  0  0  0  0  8  0  7  0]
 [ 0  0  2  0  0  9  0  4  0]
 [ 0  0  0  0  0  0  7 23  0]
 [ 0  0  0  0  0  0 19  1 10]
 [ 0  0  0  0  0  0  0 30  0]]
[0.4842105263157895, 0.5066666666666667, 0.6972972972972973, 0.5947368421052631, 0.5714285714285714, 0.64, 0.7545454545454545, 0.7157894736842105, 0.6833333333333333, 0.6190476190476191, 0.55, 0.55, 0.5578947368421052, 0.5454545454545454, 0.34285714285714286, 0.7433333333333333, 0.24444444444444444]
0.576531763962
0.131081402478

Process finished with exit code 0

'''