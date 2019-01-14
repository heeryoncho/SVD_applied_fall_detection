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
X_raw = pickle.load(open("data/X_umafall_raw.p", "rb"))
X = np.concatenate((X_svd, X_raw), axis=1)

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

    new_dir = 'model/umafall_raw_svd_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_raw_svd_conv1d_10/' + str(i+1) + '/'
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
[0.6157894736842106, 0.4533333333333333, 0.6378378378378379, 0.6473684210526316, 0.6571428571428571, 0.72, 0.8, 0.6421052631578947, 0.5944444444444444, 0.7333333333333333, 0.63, 0.5875, 0.7473684210526316, 0.6181818181818182, 0.7047619047619048, 0.6966666666666667, 0.5111111111111111]
0.646879110927
0.0828349305384
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_raw_svd_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/1/weights.40-0.62.hdf5
------ TRAIN ACCURACY:  weights.40-0.62.hdf5  ------
0.9261663286
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  0   0 243   0   0   0   2   5   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   1   0   0   0 228  10   0   1   0]
 [  0   0  29   0   0   0   7 208   0   0   1]
 [  1   0   2   0   3   0   8   0 224  36  16]
 [  0   0   0   0   0   0  12   1   6 297   4]
 [  0   0   0   0   0   0   3   0   3  30 299]]
------ TEST ACCURACY:  weights.40-0.62.hdf5  ------
0.615789473684
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  3  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  2  0  0  0]
 [ 0  0  0 14  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  2  6  0  7  0]
 [ 0  0  0  0  0  0 12  0  3  0]
 [ 0  0  0  0  0  8  0 12  4  6]
 [ 0  0  0  0  0  0  0  0 37  3]
 [ 0  0  0  0  0  0  0  0 30  0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/2/weights.47-0.73.hdf5
------ TRAIN ACCURACY:  weights.47-0.73.hdf5  ------
0.938123752495
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 123   0   2   0   0   0   0   0   0   0]
 [  1   0 241   0   0   0   0   8   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  1   0   1   0  97   0   0   1   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 230   9   1   0   0]
 [  0   0  17   0   0   0   7 220   1   0   0]
 [  0   0   1   0   1   0   5   1 270  12  15]
 [  0   0   0   0   0   0   4   0  27 275  24]
 [  0   0   0   0   0   0   1   0   7   8 334]]
------ TEST ACCURACY:  weights.47-0.73.hdf5  ------
0.453333333333
[[15  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  2  0  0  0]
 [ 0 10  0  5  0  0  0  0  0]
 [ 0  0  3  0  2  6  4  0  0]
 [ 0  0 11  0  2  2  0  0  0]
 [ 0  0  0  0  0  0  2  1 12]
 [ 0  0  0  0  0  0  0 11 19]
 [ 0  0  0  0  0  0 11  1  3]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/3/weights.29-0.71.hdf5
------ TRAIN ACCURACY:  weights.29-0.71.hdf5  ------
0.884615384615
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  1   0 238   0   0   0   0   6   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  2   0   1   0  97   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   3   0   0   0 226   6   4   0   1]
 [  2   0  57   0   0   1  19 164   0   1   1]
 [  2   0   0   0   3   0   1   0 249   9  26]
 [  0   0   0   0   0   0   9   2  40 199  80]
 [  0   0   1   0   0   0   1   0   5   0 328]]
------ TEST ACCURACY:  weights.29-0.71.hdf5  ------
0.637837837838
[[15  0  0  0  0  0  0  0  0]
 [ 8  7  0  0  0  0  0  0  0]
 [ 0  0 19  0  0  1  0  0  0]
 [ 0  1  0 14  0  0  0  0  0]
 [ 0  0  0  0 11  0  3  0  1]
 [ 0  0  0  0 10  5  0  0  0]
 [ 0  0  0  0  0  0 15  0 15]
 [ 0  0  0  0  0  0  6  3 21]
 [ 0  0  0  0  0  0  1  0 29]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/4/weights.41-0.70.hdf5
------ TRAIN ACCURACY:  weights.41-0.70.hdf5  ------
0.939553752535
[[232   0   0   0   0   0   0   0   0   0   3]
 [  0 124   0   0   0   0   0   1   0   0   0]
 [  1   0 242   0   0   0   2   4   1   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 226   3   1   0   0]
 [  0   0  21   0   0   0  25 198   0   0   1]
 [  0   0   1   0   1   0   5   0 257  18   8]
 [  0   0   0   0   0   0   8   1   5 308  13]
 [  0   0   0   0   0   0   1   0   3  22 309]]
------ TEST ACCURACY:  weights.41-0.70.hdf5  ------
0.647368421053
[[20  0  0  0  0  0  0  0  0  0]
 [ 0  8  0  7  0  0  0  0  0  0]
 [ 1  0 14  0  0  0  0  0  0  0]
 [ 0  8  0  7  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0 21  0  3  0  0]
 [ 0  0  1  0  0  9  5  0  0  0]
 [ 0  0  2  0  2  0  1 11 10  4]
 [ 0  0  0  0  0  2  0  0 20  3]
 [ 0  0  0  0  0  0  0 10  3 17]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/5/weights.46-0.65.hdf5
------ TRAIN ACCURACY:  weights.46-0.65.hdf5  ------
0.934901960784
[[238   0   0   0   0   0   0   0   0   0   2]
 [  0 135   0   5   0   0   0   0   0   0   0]
 [  2   0 242   0   0   0   0   3   3   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   3   0   0   0 218  13   4   1   1]
 [  0   0  30   0   0   0   2 212   0   1   0]
 [  1   0   5   0   1   0   4   0 276  29   4]
 [  0   0   0   0   0   0   4   0  10 345   1]
 [  1   0   1   0   0   0   0   0   4  31 298]]
------ TEST ACCURACY:  weights.46-0.65.hdf5  ------
0.657142857143
[[15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0]
 [ 0  3  0 12  0  0  0  0]
 [ 0  0  0  0  9  0  2  4]
 [ 0  0  3  0  0 12  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  5 19  6]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/6/weights.48-0.64.hdf5
------ TRAIN ACCURACY:  weights.48-0.64.hdf5  ------
0.938579654511
[[247   0   0   0   0   0   0   0   0   0   3]
 [  0 138   0   2   0   0   0   0   0   0   0]
 [  1   0 255   0   0   0   0   4   0   0   0]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   1   0  98   1   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 242  11   1   1   0]
 [  0   0  27   0   0   0   6 214   1   2   0]
 [  2   0   2   0   2   0   3   0 272  16  13]
 [  0   0   0   0   0   0   2   0  20 311  17]
 [  0   0   0   0   0   0   0   0   3  19 333]]
------ TEST ACCURACY:  weights.48-0.64.hdf5  ------
0.72
[[ 5  0  0  0  0  0  0]
 [ 0  4  1  0  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 3  1  0  6  0  0  0]
 [ 0  0  0  0 10  0  0]
 [ 0  0  0  0  4  6  0]
 [ 0  0  0  0  0  5  5]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/7/weights.45-0.63.hdf5
------ TRAIN ACCURACY:  weights.45-0.63.hdf5  ------
0.935166994106
[[237   0   0   0   0   0   2   0   0   0   1]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  1   0 243   0   0   0   1   5   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  85   0   0   0   0   0   0]
 [  0   0   0   0   0  85   0   0   0   0   0]
 [  0   0   2   0   0   0 226   5   1   1   0]
 [  0   0  14   0   0   0  17 211   1   2   0]
 [  0   0   1   0   0   0   6   2 272  26  13]
 [  0   0   0   0   0   0   4   0  19 325  12]
 [  0   0   0   0   0   0   0   0   5  24 336]]
------ TEST ACCURACY:  weights.45-0.63.hdf5  ------
0.8
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  0  7  5  3  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 0  0  0  1 13  0  0  0  0  1]
 [ 0  0  0  0  0 19  0  1  0  0]
 [ 0  0  0  0  0 10  4  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/8/weights.36-0.66.hdf5
------ TRAIN ACCURACY:  weights.36-0.66.hdf5  ------
0.91796875
[[239   0   0   0   0   0   0   0   0   0   1]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  3   0 239   0   0   0   1   4   0   0   3]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  1   0   0   0  89   0   0   0   0   0   0]
 [  0   0   0   0   0  90   0   0   0   0   0]
 [  0   0   0   0   0   0 222   9   4   0   5]
 [  0   0  34   0   0   0   7 201   0   2   1]
 [  1   0   4   0   3   0   3   0 254  16  39]
 [  0   0   0   0   0   0   4   1  11 305  39]
 [  0   0   0   0   0   0   0   0   2  12 351]]
------ TEST ACCURACY:  weights.36-0.66.hdf5  ------
0.642105263158
[[15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 1  0 14  0  0  0  0  0]
 [ 0  6  0  9  0  0  0  0]
 [ 0  0  2  0  4  4  0  0]
 [ 0  0  1  0  1  8  0  0]
 [ 0  0  0  0  0  0  4 11]
 [ 0  0  8  0  0  0  0  7]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/9/weights.35-0.69.hdf5
------ TRAIN ACCURACY:  weights.35-0.69.hdf5  ------
0.922828282828
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  1   0 244   0   0   0   1   3   1   0   0]
 [  0   1   0 219   0   0   0   0   0   0   0]
 [  0   0   1   0  99   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   1   0   0   0 230   7   2   0   0]
 [  1   0  50   0   0   0   9 182   0   2   1]
 [  1   0   3   0   3   0   4   0 242  29   8]
 [  0   0   0   0   0   0   5   1  12 303   9]
 [  0   0   0   0   0   0   1   0  10  24 300]]
------ TEST ACCURACY:  weights.35-0.69.hdf5  ------
0.594444444444
[[14  1  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0]
 [ 0 11  0  4  0  0  0  0  0]
 [ 0  0  0  0  6  0  1  8  0]
 [ 0  0  6  0  9  0  0  0  0]
 [ 0  0  0  0  0  0 26  1  3]
 [ 0  0  0  0  0  0  0 22  8]
 [ 0  0  0  0  0  0  4 21  5]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/10/weights.47-0.66.hdf5
------ TRAIN ACCURACY:  weights.47-0.66.hdf5  ------
0.942745098039
[[239   0   0   0   0   0   0   0   0   0   1]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  1   0 239   0   0   0   0  10   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  85   0   0   0   0   0   0]
 [  0   0   0   0   0  85   0   0   0   0   0]
 [  0   0   1   0   0   0 222  14   1   1   1]
 [  0   0  11   0   0   0   1 230   0   2   1]
 [  0   0   2   0   3   0   5   1 273  25  11]
 [  0   0   0   0   0   0   4   0  10 336  10]
 [  0   0   0   0   0   0   0   0   4  26 335]]
------ TEST ACCURACY:  weights.47-0.66.hdf5  ------
0.733333333333
[[14  0  0  1  0  0  0]
 [ 1 11  0  2  0  0  1]
 [ 0  0 15  0  0  0  0]
 [ 3  0  0 12  0  0  0]
 [ 1  2  0  1 11  0  0]
 [ 0  0  0  0  0 11  4]
 [ 0 12  0  0  0  0  3]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/11/weights.39-0.66.hdf5
------ TRAIN ACCURACY:  weights.39-0.66.hdf5  ------
0.923679060665
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 139   0   1   0   0   0   0   0   0   0]
 [  1   0 232   0   0   0   0  12   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   1   0  89   0   0   0   0   0   0]
 [  0   0   0   0   0  90   0   0   0   0   0]
 [  0   0   0   0   0   0 232   7   1   0   0]
 [  0   0  15   0   0   0  14 214   1   1   0]
 [  1   0   2   0   3   0   7   0 265  21  21]
 [  0   0   0   0   0   0  12   1  17 302  28]
 [  0   0   0   0   0   0   2   1   6  19 337]]
------ TEST ACCURACY:  weights.39-0.66.hdf5  ------
0.63
[[15  0  0  0  0  0  0]
 [ 1 19  0  0  0  0  0]
 [ 0  0 15  0  0  0  0]
 [ 1  0  0  9  0  0  0]
 [ 2  0  0  3  5  0  0]
 [ 0  0  0  0  0  0 15]
 [ 0 13  0  1  1  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/12/weights.29-0.65.hdf5
------ TRAIN ACCURACY:  weights.29-0.65.hdf5  ------
0.902605210421
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  1   0 238   0   0   0   0   5   1   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  1   0   3   0  81   0   0   0   0   0   0]
 [  0   0   3   0   1  76   0   0   0   0   0]
 [  0   0   0   0   0   0 223  14   3   0   0]
 [  0   0  34   0   0   0  16 192   0   2   1]
 [  0   0   4   0   1   0   5   1 246  29  19]
 [  0   0   0   0   0   0  10   1  19 303  12]
 [  0   0   1   0   0   0   2   0  16  38 293]]
------ TEST ACCURACY:  weights.29-0.65.hdf5  ------
0.5875
[[15  0  0  0  0  0  0  0  0  0]
 [ 2 18  0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 9  0  3  2  1  0  0  0  0  0]
 [ 8  0 10  0  2  0  0  0  0  0]
 [ 0  0  0  0  0 15  0  0  0  0]
 [ 0  3  0  0  0  4  8  0  0  0]
 [ 0  0  0  0  0  1  0  0 13  1]
 [ 0  0  0  0  0  6  0  5  4  0]
 [ 0  0  0  0  0  0  0  0  0 15]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/13/weights.26-0.66.hdf5
------ TRAIN ACCURACY:  weights.26-0.66.hdf5  ------
0.886328125
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 134   0   1   0   0   0   0   0   0   0]
 [  0   0 255   0   0   0   1   3   1   0   0]
 [  0   0   0 230   0   0   0   0   0   0   0]
 [  1   0   3   0  96   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   2   0   0   0 236   5   4   2   1]
 [  0   0  49   0   0   0  27 176   1   2   0]
 [  0   0   5   0   0   0   5   2 248  31   9]
 [  0   0   0   0   0   0  12   0  17 307   4]
 [  0   0   0   0   0   0   1   0  29  73 242]]
------ TEST ACCURACY:  weights.26-0.66.hdf5  ------
0.747368421053
[[ 6  0  1  0  0  3  0  0  0]
 [ 0  2  0  3  0  0  0  0  0]
 [ 0  0  5  0  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  0  0  2  0  3  0  0]
 [ 0  0  2  0  2  1  0  0  0]
 [ 0  0  0  0  0  0 17  3  0]
 [ 0  0  0  0  0  0  0 15  5]
 [ 0  0  0  0  0  0  2  0 18]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/14/weights.48-0.67.hdf5
------ TRAIN ACCURACY:  weights.48-0.67.hdf5  ------
0.944230769231
[[248   0   0   0   0   0   0   0   0   0   2]
 [  0 139   0   1   0   0   0   0   0   0   0]
 [  1   0 248   0   0   0   1   4   1   0   0]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 246   8   1   0   0]
 [  0   0  10   0   0   0   4 234   0   2   0]
 [  1   0   1   0   1   0   6   0 265  28  13]
 [  0   0   0   0   0   0  11   0   8 311  15]
 [  0   0   0   0   0   0   0   1   3  22 329]]
------ TEST ACCURACY:  weights.48-0.67.hdf5  ------
0.618181818182
[[5 0 0 0 0 0 0]
 [0 1 0 9 0 0 0]
 [0 0 0 0 0 0 0]
 [0 2 0 8 0 0 0]
 [0 0 0 0 5 0 0]
 [0 0 2 2 2 6 3]
 [0 0 1 0 0 0 9]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/15/weights.23-0.67.hdf5
------ TRAIN ACCURACY:  weights.23-0.67.hdf5  ------
0.890980392157
[[243   0   0   0   0   0   0   2   0   0   0]
 [  0 132   0   3   0   0   0   0   0   0   0]
 [  2   0 233   0   0   0   0  25   0   0   0]
 [  0   0   0 225   0   0   0   0   0   0   0]
 [  5   0   4   1  88   1   0   1   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 204  34   4   3   0]
 [  0   0  22   0   0   0   4 222   0   2   0]
 [  0   0   5   0   1   0   5   1 226  35  27]
 [  0   0   0   0   0   0   7   1   8 280  49]
 [  0   0   0   1   0   0   1   3   6  15 319]]
------ TEST ACCURACY:  weights.23-0.67.hdf5  ------
0.704761904762
[[10  0  0  0  0  0  0  0  0]
 [ 0  1  0  4  0  0  0  0  0]
 [ 0  0  2  0  0  3  0  0  0]
 [ 1  0  0  9  0  0  0  0  0]
 [ 0  0  0  0  8  2  0  0  0]
 [ 0  0  0  0  1  9  0  0  0]
 [ 0  0  0  0  0  1  9  5  5]
 [ 0  0  0  0  0  0  1  6  8]
 [ 0  0  0  0  0  0  0  0 20]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/16/weights.49-0.56.hdf5
------ TRAIN ACCURACY:  weights.49-0.56.hdf5  ------
0.920681265207
[[215   0   0   0   0   0   0   0   0   0   0]
 [  0 100   0   0   0   0   0   0   0   0   0]
 [  1   0 202   0   0   0   0  17   0   0   0]
 [  0   0   0 195   0   0   0   0   0   0   0]
 [  0   0   0   0  65   0   0   0   0   0   0]
 [  0   0   0   0   0  70   0   0   0   0   0]
 [  0   0   0   0   0   0 190  16   3   1   0]
 [  0   0   5   0   0   0   1 208   0   1   0]
 [  1   0   1   0   1   0   2   2 193  35   0]
 [  0   0   0   0   0   0   3   0  20 234   3]
 [  0   0   0   0   0   0   0   1  16  33 220]]
------ TEST ACCURACY:  weights.49-0.56.hdf5  ------
0.696666666667
[[38  0  0  0  0  2  0  0  0  0  0]
 [ 0 30  0 10  0  0  0  0  0  0  0]
 [ 0  0 31  0  0  0  0 14  0  0  0]
 [ 4  0  0 35  1  0  0  0  0  0  0]
 [ 2  0  5  0 26  0  0  2  0  0  0]
 [ 0  0  0  0 18  8  0  1  3  0  0]
 [ 0  0  0  0  0  0 27 18  0  0  0]
 [ 0  0  6  0  0  0  0 39  0  0  0]
 [ 0  0  1  0  0  0  4  0 80  0  0]
 [ 0  0  0  0  0  0  5  2 11 81  1]
 [ 1  0  0  0  0  0  3  0 48 20 23]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_svd_conv1d_10/17/weights.49-0.80.hdf5
------ TRAIN ACCURACY:  weights.49-0.80.hdf5  ------
0.942222222222
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  0   0 248   0   0   0   0   2   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 238   0   1   1   0]
 [  0   0  36   0   0   0  17 190   0   2   0]
 [  0   0   2   0   1   0   5   1 265   7   9]
 [  0   0   0   0   0   0   4   1  12 310   3]
 [  0   0   0   0   0   0   2   0  12  25 296]]
------ TEST ACCURACY:  weights.49-0.80.hdf5  ------
0.511111111111
[[14  0  0  0  0  0  0  0  0  1]
 [ 0 14  0  1  0  0  0  0  0  0]
 [ 1  0 13  0  0  1  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  8  5  0  2  0]
 [ 0  0  3  0  0  2 10  0  0  0]
 [ 0  0  0  0  1  0  0  8 21  0]
 [ 0  0  0  0  0  5  0 15  2  8]
 [ 0  0  0  0  0  0  0  1 21  8]]
[0.6157894736842106, 0.4533333333333333, 0.6378378378378379, 0.6473684210526316, 0.6571428571428571, 0.72, 0.8, 0.6421052631578947, 0.5944444444444444, 0.7333333333333333, 0.63, 0.5875, 0.7473684210526316, 0.6181818181818182, 0.7047619047619048, 0.6966666666666667, 0.5111111111111111]
0.646879110927
0.0828349305384

Process finished with exit code 0

'''