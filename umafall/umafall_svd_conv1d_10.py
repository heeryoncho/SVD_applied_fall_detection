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

X = pickle.load(open("data/X_umafall_svd.p", "rb"))
y = pickle.load(open("data/y_umafall_svd.p", "rb"))

n_classes = 11
signal_rows = 450
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
    print X_train.shape # (2465, 450)

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

    new_dir = 'model/umafall_svd_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_svd_conv1d_10/' + str(i+1) + '/'
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
[0.5473684210526316, 0.52, 0.44324324324324327, 0.5105263157894737, 0.6, 0.44, 0.7454545454545455, 0.5894736842105263, 0.5611111111111111, 0.5238095238095238, 0.58, 0.48125, 0.5263157894736842, 0.6, 0.49523809523809526, 0.535, 0.49444444444444446]
0.540778539637
0.0696986541548
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_svd_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/1/weights.50-0.51.hdf5
------ TRAIN ACCURACY:  weights.50-0.51.hdf5  ------
0.843813387424
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 115   0   9   0   0   1   0   0   0   0]
 [  1   0 221   0   0   0   4   7   8   7   2]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  96   0   0   0   2   0   2]
 [  0   0   0   0   0  98   0   0   0   0   2]
 [  0   0  10   0   0   0 196  30   3   0   1]
 [  0   0  41   0   0   0  30 165   4   4   1]
 [  1   0   7   0   2   1   7   8 210  20  34]
 [  0   0  12   0   0   0  13   5  16 247  27]
 [  0   0  15   1   0   2   0   0  25  17 275]]
------ TEST ACCURACY:  weights.50-0.51.hdf5  ------
0.547368421053
[[14  0  0  0  0  0  0  0  1  0  0]
 [ 0  8  0  7  0  0  0  0  0  0  0]
 [ 0  0 10  0  0  0  0  3  0  1  1]
 [ 0  0  0 13  0  2  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0 13  0  1  0  0]
 [ 0  0  0  0  0  0 12  2  1  0  0]
 [ 1  0  4  0  2  1  1  0  9  5  7]
 [ 0  0  5  0  1  1  1  2  6 17  7]
 [ 0  0  2  0  0  0  0  0  4  6 18]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/2/weights.38-0.53.hdf5
------ TRAIN ACCURACY:  weights.38-0.53.hdf5  ------
0.787225548902
[[233   0   0   0   1   0   4   0   0   1   1]
 [  0 121   0   2   0   0   2   0   0   0   0]
 [  1   0 225   0   0   1   5   2   9   2   5]
 [  1   0   0 218   0   0   0   0   0   0   1]
 [  0   0   1   0  93   1   0   0   1   1   3]
 [  0   0   1   0   1  96   0   0   0   0   2]
 [  0   0  18   0   0   0 207  11   4   0   0]
 [  0   0  66   0   0   0  67 104   6   2   0]
 [  0   0  22   0   1   2  16   1 202  15  46]
 [  0   0  26   0   0   0  15   0  33 197  59]
 [  0   0  26   2   2   3   2   1  29   9 276]]
------ TEST ACCURACY:  weights.38-0.53.hdf5  ------
0.52
[[15  0  0  0  0  0  0  0  0  0]
 [ 5 10  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0  0]
 [ 0  0  0  7  0  0  0  0  0  8]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  6  3  2  0  0]
 [ 0  0 13  0  0  0  2  0  0  0]
 [ 0  0  0  0  0  0  0  5  0 10]
 [ 0  0  2  0  1  1  1  1  8 16]
 [ 0  0  0  0  1  0  0  2  1 11]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/3/weights.47-0.58.hdf5
------ TRAIN ACCURACY:  weights.47-0.58.hdf5  ------
0.845344129555
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  1   0 201   0   1   0   3  18  13   2   6]
 [  0   6   0 213   0   0   0   0   0   0   1]
 [  0   0   0   0  94   0   0   0   3   0   3]
 [  0   0   0   0   0  98   0   0   0   0   2]
 [  0   0   4   0   0   0 164  66   6   0   0]
 [  0   0  16   0   0   0  24 194   9   1   1]
 [  1   0   4   0   3   0   6  11 222  14  29]
 [  0   0   7   0   0   0   8   7  29 238  41]
 [  0   0   1   2   2   1   1   3  18   6 301]]
------ TEST ACCURACY:  weights.47-0.58.hdf5  ------
0.443243243243
[[15  0  0  0  0  0  0  0  0]
 [ 3 11  0  0  0  1  0  0  0]
 [ 0  0  9  0  0  9  1  0  1]
 [ 0 14  0  0  0  0  0  0  1]
 [ 0  0  0  0 12  0  2  1  0]
 [ 0  0  0  0  4 11  0  0  0]
 [ 1  0  1  1  0  1  3  3 20]
 [ 2  0  0  0  1  0  4  5 18]
 [ 0  4  0  0  0  0  2  8 16]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/4/weights.45-0.51.hdf5
------ TRAIN ACCURACY:  weights.45-0.51.hdf5  ------
0.829614604462
[[233   0   0   0   0   0   1   1   0   0   0]
 [  0 109   0  15   0   0   1   0   0   0   0]
 [  1   0 201   0   0   1   3  32   6   3   3]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  1   0   2   0  91   2   1   0   2   1   0]
 [  0   0   0   0   0  98   0   0   0   1   1]
 [  0   0   6   0   0   0 153  68   2   1   0]
 [  0   0  24   0   0   0   7 209   3   2   0]
 [  0   0   9   1   0   1   8  14 216  19  22]
 [  0   0  10   0   0   0  15   8  23 250  29]
 [  1   0  15   2   1   2   4   3  22  20 265]]
------ TEST ACCURACY:  weights.45-0.51.hdf5  ------
0.510526315789
[[18  1  0  1  0  0  0  0  0  0  0]
 [ 5  0  0  7  0  0  0  0  0  0  3]
 [ 0  0 10  0  0  0  0  2  0  1  2]
 [15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0 14  9  1  0  0]
 [ 0  0  1  0  0  0  3 11  0  0  0]
 [ 1  0  2  0  3  1  0  2 14  1  6]
 [ 1  0  0  0  0  0  2  0  2  9 11]
 [ 0  0  0  0  0  0  0  0  4  5 21]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/5/weights.46-0.52.hdf5
------ TRAIN ACCURACY:  weights.46-0.52.hdf5  ------
0.832549019608
[[236   0   0   0   0   0   3   0   0   1   0]
 [  0 131   0   8   0   0   1   0   0   0   0]
 [  1   0 196   0   0   1   2  37   6   5   2]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  95   1   0   0   2   1   1]
 [  0   0   0   0   0  96   0   0   0   2   2]
 [  0   0   3   0   0   0 137  97   2   1   0]
 [  0   0  17   0   0   0   9 214   2   2   1]
 [  2   0  11   0   2   0   5  17 217  36  30]
 [  0   0   9   0   0   0   6  16  15 298  16]
 [  2   0   2   1   3   2   1   2  14  24 284]]
------ TEST ACCURACY:  weights.46-0.52.hdf5  ------
0.6
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  2  0  0  1]
 [ 0  2  0 13  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  1  7  3  1  1  0]
 [ 0  0  0  0  0  1 14  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  1  2  5  8  2]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/6/weights.48-0.52.hdf5
------ TRAIN ACCURACY:  weights.48-0.52.hdf5  ------
0.846065259117
[[246   0   0   0   0   0   2   0   0   2   0]
 [  0 137   0   2   0   0   0   0   0   1   0]
 [  1   0 225   0   2   0   6   9   7   8   2]
 [  0   1   0 233   0   0   0   0   0   0   1]
 [  0   0   0   0  97   0   0   0   1   1   1]
 [  0   0   0   0   0  98   0   0   0   1   1]
 [  0   0   5   0   0   0 212  32   5   1   0]
 [  0   0  33   0   0   0  43 161   6   5   2]
 [  0   0   8   0   4   1  11   3 205  46  32]
 [  0   0   8   0   0   0   9   3  12 291  27]
 [  0   0   3   2   5   1   2   0  13  30 299]]
------ TEST ACCURACY:  weights.48-0.52.hdf5  ------
0.44
[[5 0 0 0 0 0 0 0]
 [0 4 0 0 0 0 1 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [1 0 0 3 3 2 1 0]
 [2 0 1 0 0 1 4 2]
 [0 0 0 0 0 3 5 2]
 [0 0 0 0 0 2 4 4]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/7/weights.43-0.55.hdf5
------ TRAIN ACCURACY:  weights.43-0.55.hdf5  ------
0.798428290766
[[238   0   0   0   0   0   2   0   0   0   0]
 [  1 137   0   2   0   0   0   0   0   0   0]
 [  1   0 219   0   0   0   6   8  12   2   2]
 [  1   0   0 217   0   2   0   0   0   0   0]
 [  1   0   1   0  79   0   0   0   2   0   2]
 [  1   0   0   0   0  81   0   0   1   0   2]
 [  0   0   9   0   0   0 206  15   5   0   0]
 [  1   0  51   0   0   0  71 109   8   3   2]
 [  4   0  13   0   0   1  11   3 234  10  44]
 [  2   0  14   0   0   0  17   3  45 195  84]
 [  2   0   9   1   2   2   6   1  22   3 317]]
------ TEST ACCURACY:  weights.43-0.55.hdf5  ------
0.745454545455
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 13  0  0  0  0  0  1  0  1]
 [ 0  0 11  0  0  0  0  0  1  3]
 [ 2  0  0 11  0  0  0  1  0  1]
 [ 0  0  0  1  9  0  0  1  0  4]
 [ 0  3  0  0  0 16  0  1  0  0]
 [ 0  0  0  0  0  8  7  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/8/weights.26-0.53.hdf5
------ TRAIN ACCURACY:  weights.26-0.53.hdf5  ------
0.78046875
[[234   0   0   0   0   0   4   0   0   1   1]
 [  0 131   0   6   0   0   0   0   0   3   0]
 [  0   0 183   0   3   0   3  19  21  11  10]
 [  0   0   0 217   0   0   0   0   0   1   2]
 [  1   0   0   0  83   0   0   0   0   4   2]
 [  0   0   0   0   0  85   0   0   0   2   3]
 [  0   0   6   0   0   0 163  59   7   4   1]
 [  0   0  20   0   0   0  29 176  10   8   2]
 [  0   1   5   0   1   1   8  12 200  33  59]
 [  0   0   6   0   0   0   9   8  36 225  76]
 [  0   0   5   2   2   2   3   1  29  20 301]]
------ TEST ACCURACY:  weights.26-0.53.hdf5  ------
0.589473684211
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  0  0  2  0  1]
 [ 0  2  0 10  0  0  0  0  0  0  3]
 [ 0  0  0  0  4  2  0  0  3  1  0]
 [ 0  0  0  0  0  6  0  0  0  2  2]
 [ 0  0  0  0  0  0  3 12  0  0  0]
 [ 0  0  7  0  0  0  0  6  1  1  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/9/weights.45-0.50.hdf5
------ TRAIN ACCURACY:  weights.45-0.50.hdf5  ------
0.819393939394
[[238   0   0   0   0   0   2   0   0   0   0]
 [  1 119   0   5   0   0   0   0   0   0   0]
 [  1   0 222   0   1   0   3   4  10   5   4]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  0   0   1   0  93   1   0   0   2   1   2]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0  30   0   0   0 194  12   3   1   0]
 [  2   0  66   0   0   0  43 122   6   2   4]
 [  4   0  10   0   4   1   9   5 191  27  39]
 [  1   0  12   0   0   0   9   4  22 246  36]
 [  0   0  12   2   2   1   0   0  16  17 285]]
------ TEST ACCURACY:  weights.45-0.50.hdf5  ------
0.561111111111
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  0  1  0  0]
 [ 0  9  0  6  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  1  0 10  1  1  0  0]
 [ 0  0  4  0  0  0  4  6  1  0  0]
 [ 0  0  1  0  1  1  1  0 10  6 10]
 [ 2  1  0  0  0  0  2  0  4  9 12]
 [ 1  2  1  1  1  2  0  0  4  2 16]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/10/weights.34-0.54.hdf5
------ TRAIN ACCURACY:  weights.34-0.54.hdf5  ------
0.828235294118
[[235   0   0   0   0   0   4   0   0   1   0]
 [  1 130   0   8   0   0   1   0   0   0   0]
 [  1   0 207   0   0   1  10  14  11   5   1]
 [  0   0   0 218   0   0   0   0   0   1   1]
 [  0   0   0   0  79   1   0   0   3   1   1]
 [  0   0   0   0   0  84   0   0   1   0   0]
 [  0   0   4   0   1   0 210  18   7   0   0]
 [  0   0  28   0   0   0  47 159   9   1   1]
 [  1   0   7   0   1   1  10   6 239  22  33]
 [  1   0   8   0   0   0  18   2  30 265  36]
 [  2   0   8   1   2   2   5   0  41  18 286]]
------ TEST ACCURACY:  weights.34-0.54.hdf5  ------
0.52380952381
[[11  0  0  0  4  0  0  0  0  0]
 [ 0  1  0  3  1  0  0  6  2  2]
 [ 0  0 14  0  1  0  0  0  0  0]
 [ 1  0  0  9  4  0  0  1  0  0]
 [ 1  0  0  0 11  0  0  1  0  2]
 [ 0  0  0  0  0  9  6  0  0  0]
 [ 0  4  0  0  0  1  0  6  2  2]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/11/weights.42-0.51.hdf5
------ TRAIN ACCURACY:  weights.42-0.51.hdf5  ------
0.815655577299
[[238   0   0   0   0   0   2   0   0   0   0]
 [  2 129   0   9   0   0   0   0   0   0   0]
 [  1   0 196   0   5   0   4  21   6   6   6]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  1   0   0   0  84   0   0   0   2   1   2]
 [  0   0   0   0   1  87   0   0   0   0   2]
 [  0   0   6   0   1   0 181  45   5   1   1]
 [  2   0  26   0   0   0  17 190   6   1   3]
 [  4   0   9   2   5   0   9   9 205  16  61]
 [  1   0   9   0   0   0  10   2  30 231  77]
 [  4   0   4   1   4   1   5   0  15   7 324]]
------ TEST ACCURACY:  weights.42-0.51.hdf5  ------
0.58
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  2  0  2  4  1]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 2  0  0  8  0  0  0  0  0  0]
 [ 2  0  0  0  8  0  0  0  0  0]
 [ 0  1  0  0  0  1 12  1  0  0]
 [ 0  9  0  0  0  0  0  5  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/12/weights.23-0.54.hdf5
------ TRAIN ACCURACY:  weights.23-0.54.hdf5  ------
0.747094188377
[[235   0   0   0   0   0   4   0   0   1   0]
 [  0 139   0   0   0   0   0   0   0   1   0]
 [  1   0 191   0   3   0   5  12  15   7  11]
 [  1  20   0 195   0   0   0   0   0   0   4]
 [  1   1   1   0  74   2   0   0   2   1   3]
 [  0   0   0   0   1  72   0   0   0   2   5]
 [  0   0  24   0   2   0 146  60   7   1   0]
 [  1   0  40   0   0   0  26 163  10   2   3]
 [  3   2  12   0   4   1  12   7 178  26  60]
 [  0   3  14   0   0   0  13  17  27 182  89]
 [  4   3  19   2   2   1   1   1  15  13 289]]
------ TEST ACCURACY:  weights.23-0.54.hdf5  ------
0.48125
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  0  0  3  1  3]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [13  0  0  0  0  2  0  0  0  0  0]
 [ 8  1  0  7  0  3  0  0  0  0  1]
 [ 0  0  4  0  0  0 10  1  0  0  0]
 [ 0  0  2  0  0  0  4  9  0  0  0]
 [ 1  0  1  0  0  0  0  0  2  1 10]
 [ 1  1  3  0  0  0  1  0  3  3  3]
 [ 0  1  0  0  1  0  1  0  5  0  7]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/13/weights.47-0.52.hdf5
------ TRAIN ACCURACY:  weights.47-0.52.hdf5  ------
0.837109375
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 132   0   2   0   0   0   1   0   0   0]
 [  1   0 206   0   0   1   5  25  14   7   1]
 [  3   1   0 225   0   0   0   0   0   1   0]
 [  1   0   0   0  95   0   0   0   2   1   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   2   0   0   0 198  48   1   1   0]
 [  0   0  20   0   0   0  32 193   7   2   1]
 [  0   0   7   0   0   1   7  10 231  28  16]
 [  0   0   7   0   0   1  12   9  32 262  17]
 [  0   0   5   1   3   1   1   0  47  30 257]]
------ TEST ACCURACY:  weights.47-0.52.hdf5  ------
0.526315789474
[[ 5  0  0  0  3  1  0  1  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  2  0  1  0]
 [ 0  2  0  3  0  0  0  0  0]
 [ 0  0  0  0  3  0  2  0  0]
 [ 0  0  0  0  1  4  0  0  0]
 [ 0  0  0  0  0  0  7 10  3]
 [ 0  1  0  0  0  0  0 12  7]
 [ 0  2  0  0  0  0  6  3  9]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/14/weights.28-0.54.hdf5
------ TRAIN ACCURACY:  weights.28-0.54.hdf5  ------
0.762307692308
[[243   0   0   0   1   1   4   0   0   1   0]
 [  1 127   0  11   0   0   1   0   0   0   0]
 [  0   0 216   0   1   1   5   5  13  11   3]
 [  0   0   0 233   0   0   0   0   1   1   0]
 [  0   1   1   0  89   2   0   0   2   4   1]
 [  0   0   0   0   0  98   0   0   1   1   0]
 [  0   0  26   0   0   0 166  52  10   1   0]
 [  0   0  62   0   0   0  30 143  10   5   0]
 [  0   0  12   0   1   3   7  11 210  49  22]
 [  0   0  16   0   0   0   7   8  48 241  25]
 [  0   0  18   4   2   2   1   1  56  55 216]]
------ TEST ACCURACY:  weights.28-0.54.hdf5  ------
0.6
[[5 0 0 0 0 0 0]
 [0 8 0 2 0 0 0]
 [0 0 0 0 0 0 0]
 [0 5 0 5 0 0 0]
 [0 1 0 1 2 0 1]
 [0 0 1 2 1 8 3]
 [0 0 1 0 0 4 5]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/15/weights.43-0.50.hdf5
------ TRAIN ACCURACY:  weights.43-0.50.hdf5  ------
0.837254901961
[[241   0   0   0   0   0   3   0   0   1   0]
 [  0 131   0   3   0   0   1   0   0   0   0]
 [  1   0 226   0   0   1   5  15   5   5   2]
 [  0   0   0 224   0   0   0   0   0   1   0]
 [  0   0   0   0  94   0   2   0   2   1   1]
 [  0   0   0   0   1  96   0   0   0   2   1]
 [  0   0   7   0   0   0 203  33   1   1   0]
 [  0   0  32   0   0   0  44 169   2   3   0]
 [  1   0  17   0   4   1   9  10 187  35  36]
 [  0   0  10   0   0   0  17   4   9 288  17]
 [  0   0  18   1   3   1   2   2  14  28 276]]
------ TEST ACCURACY:  weights.43-0.50.hdf5  ------
0.495238095238
[[ 7  0  0  0  3  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  4  1  0  0  0]
 [ 0  2  0  8  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  6  2  0  0  0]
 [ 0  0  0  0  0  2  8  0  0  0]
 [ 0  0  2  0  0  1  3  4  4  6]
 [ 2  0  0  0  0  0  0  2  7  4]
 [ 0  0  0  0  0  0  1  1  6 12]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/16/weights.40-0.52.hdf5
------ TRAIN ACCURACY:  weights.40-0.52.hdf5  ------
0.843309002433
[[212   0   0   0   1   0   2   0   0   0   0]
 [  0  96   0   3   0   0   1   0   0   0   0]
 [  1   0 171   0   1   0   4  30   7   3   3]
 [  0   0   0 194   0   0   0   0   0   1   0]
 [  0   0   0   0  63   0   0   0   2   0   0]
 [  0   0   0   0   0  68   0   0   0   1   1]
 [  0   0   5   0   0   0 161  41   2   1   0]
 [  0   0  14   0   0   0  25 171   2   1   2]
 [  1   0   5   0   1   0   4  11 159  18  36]
 [  2   0   5   0   0   0   7  10  10 207  19]
 [  0   0   5   2   4   0   5   2  10  11 231]]
------ TEST ACCURACY:  weights.40-0.52.hdf5  ------
0.535
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 3 35  0  0  0  0  0  0  0  0  2]
 [ 0  0 33  0  0  0  0 10  2  0  0]
 [ 5  0  0 35  0  0  0  0  0  0  0]
 [ 1  3  2  0 23  0  1  0  1  3  1]
 [ 0  0  1  2  2  3  0  0  5  6 11]
 [ 0  0  0  0  0  0 24 21  0  0  0]
 [ 0  0  4  0  0  0  6 34  0  1  0]
 [ 1  0  2  3  0  0 13  3 32 21 10]
 [ 0  3  9  0  0  0  9 10 12 34 23]
 [ 1  3  0  0  0  0  4  0  9 50 28]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_svd_conv1d_10/17/weights.26-0.38.hdf5
------ TRAIN ACCURACY:  weights.26-0.38.hdf5  ------
0.765252525253
[[235   0   0   0   0   0   4   0   0   1   0]
 [  0 119   0   4   0   0   2   0   0   0   0]
 [  0   0 185   0   3   1   5  33  13   2   8]
 [  6   0   0 211   0   0   0   0   0   2   1]
 [  1   0   0   0  88   2   1   0   3   2   3]
 [  0   0   0   0   0  93   0   0   0   4   3]
 [  0   0   6   0   1   0 166  60   7   0   0]
 [  0   0  23   0   0   0  29 177  12   3   1]
 [  4   0  10   1   3   1  17   5 198  19  32]
 [  0   0  11   0   1   1  12  12  40 190  63]
 [  1   0  10   0   1   0  10   2  51  28 232]]
------ TEST ACCURACY:  weights.26-0.38.hdf5  ------
0.494444444444
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  3  0 12  0  0  0  0  0  0  0]
 [ 0  0  7  0  0  1  1  0  2  3  1]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 12  3  0  0  0]
 [ 0  0  2  0  0  0  8  5  0  0  0]
 [ 0  0  3  0  0  0  1  2 19  3  2]
 [ 0  0  4  0  0  0  6  3 11  3  3]
 [ 0  0  0  2  2  2  0  0 13  1 10]]
[0.5473684210526316, 0.52, 0.44324324324324327, 0.5105263157894737, 0.6, 0.44, 0.7454545454545455, 0.5894736842105263, 0.5611111111111111, 0.5238095238095238, 0.58, 0.48125, 0.5263157894736842, 0.6, 0.49523809523809526, 0.535, 0.49444444444444446]
0.540778539637
0.0696986541548

Process finished with exit code 0

'''