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

X_kpca = pickle.load(open("data/X_umafall_kpca.p", "rb"))
X_smv = pickle.load(open("data/X_umafall_smv.p", "rb"))
X = np.concatenate((X_kpca, X_smv), axis=1)

y = pickle.load(open("data/y_umafall_kpca.p", "rb"))

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

    new_dir = 'model/umafall_smv_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_smv_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.5263157894736842, 0.6266666666666667, 0.4864864864864865, 0.5894736842105263, 0.5238095238095238, 0.42, 0.5545454545454546, 0.4842105263157895, 0.5166666666666667, 0.47619047619047616, 0.58, 0.5125, 0.49473684210526314, 0.5454545454545454, 0.5333333333333333, 0.515, 0.45555555555555555]
0.520055620636
0.0491604879674
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_smv_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/1/weights.29-0.48.hdf5
------ TRAIN ACCURACY:  weights.29-0.48.hdf5  ------
0.821095334686
[[236   1   0   1   0   0   2   0   0   0   0]
 [  0 121   0   4   0   0   0   0   0   0   0]
 [  0   0 233   0   0   0   1  14   1   1   0]
 [  0   0   0 217   0   2   0   0   0   1   0]
 [  0   0   1   0  95   0   0   0   2   1   1]
 [  1   0   1   0   1  89   0   0   0   3   5]
 [  0   0  27   0   0   0 184  26   1   1   1]
 [  0   0  69   0   0   0  22 153   0   0   1]
 [  1   0  28   0   0   1   6  12 187  19  36]
 [  0   0  22   0   1   0  16   1  27 221  32]
 [  1   0  12   0   1   1   2   2  16  12 288]]
------ TEST ACCURACY:  weights.29-0.48.hdf5  ------
0.526315789474
[[15  0  0  0  0  0  0  0  0]
 [ 0 10  0  5  0  0  0  0  0]
 [ 0  0 11  0  0  4  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  3  0 12  0  0  0  0]
 [ 0  0  0  0 10  5  0  0  0]
 [ 0  0 11  0  1  2  5  4  7]
 [ 0  0  9  0  0  2  9 10 10]
 [ 0  0  0  0  0  0  4  9 17]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/2/weights.25-0.53.hdf5
------ TRAIN ACCURACY:  weights.25-0.53.hdf5  ------
0.803992015968
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  1   0 201   0   0   0  22   8  14   3   1]
 [  0   2   0 213   0   2   0   0   1   1   1]
 [  1   0   0   0  92   0   1   1   4   1   0]
 [  1   0   0   0   1  91   0   0   3   2   2]
 [  1   0   7   0   0   0 226   3   2   1   0]
 [  0   0  42   0   0   0  95 101   6   0   1]
 [  1   0  13   0   0   0  20   1 221  21  28]
 [  1   0   5   0   1   0  31   3  33 240  16]
 [  4   0   2   0   0   2   7   1  48  20 266]]
------ TEST ACCURACY:  weights.25-0.53.hdf5  ------
0.626666666667
[[15  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  1  1  0  0]
 [ 0  7  0  8  0  0  0  0  0]
 [ 0  0  0  0 14  1  0  0  0]
 [ 0  0  8  0  6  0  1  0  0]
 [ 0  0  0  0  0  0  6  3  6]
 [ 3  0  0  0  0  0  4 14  9]
 [ 1  0  0  0  0  0  1  4  9]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/3/weights.35-0.48.hdf5
------ TRAIN ACCURACY:  weights.35-0.48.hdf5  ------
0.864777327935
[[236   0   0   1   1   0   2   0   0   0   0]
 [  0 121   0   4   0   0   0   0   0   0   0]
 [  0   0 230   0   0   0   1   6   6   1   1]
 [  0   0   0 217   0   2   0   0   0   1   0]
 [  1   0   0   0  96   0   0   0   2   0   1]
 [  0   0   0   0   0  95   0   0   2   2   1]
 [  0   0   8   0   0   0 198  28   4   2   0]
 [  0   0  47   0   0   0  25 168   4   0   1]
 [  0   0   9   0   0   0   4   6 215  34  22]
 [  0   0   6   0   1   0  10   5   9 286  13]
 [  0   0   4   0   0   2   0   1  16  38 274]]
------ TEST ACCURACY:  weights.35-0.48.hdf5  ------
0.486486486486
[[15  0  0  0  0  0  0  0  0]
 [ 4 10  0  0  0  0  0  1  0]
 [ 0  0 12  0  0  7  0  0  1]
 [ 0  4  0  9  0  0  1  1  0]
 [ 0  0  0  0 14  0  0  1  0]
 [ 0  0  0  0  5 10  0  0  0]
 [ 1  0  1  0  2  0  0 11 15]
 [ 2  0  4  0  0  0  5  4 15]
 [ 0  0  1  0  0  0  1 12 16]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/4/weights.31-0.47.hdf5
------ TRAIN ACCURACY:  weights.31-0.47.hdf5  ------
0.829208924949
[[231   0   0   2   0   0   2   0   0   0   0]
 [  0 121   0   4   0   0   0   0   0   0   0]
 [  1   0 165   0   1   0  16  33  17  13   4]
 [  0   0   0 217   0   2   0   0   0   1   0]
 [  0   0   0   0  93   0   0   0   3   2   2]
 [  2   0   0   0   1  87   0   0   2   4   4]
 [  0   0   0   0   0   0 202  23   2   3   0]
 [  0   0  14   0   0   0  45 176  10   0   0]
 [  1   0   1   0   0   0  16   6 209  43  14]
 [  0   0   2   0   0   0  18   3  15 292   5]
 [  1   0   0   0   0   1   1   2  28  51 251]]
------ TEST ACCURACY:  weights.31-0.47.hdf5  ------
0.589473684211
[[16  0  0  0  1  0  0  1  0  2]
 [ 1  5  0  9  0  0  0  0  0  0]
 [ 0  0 10  0  0  0  2  2  1  0]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  0 20  3  0  0  0]
 [ 0  0  1  0  0  7  7  0  0  0]
 [ 1  0  0  0  0  1  2 10  7  9]
 [ 0  0  0  0  0  0  1  7 16  1]
 [ 0  0  0  0  0  0  0  4 13 13]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/5/weights.20-0.53.hdf5
------ TRAIN ACCURACY:  weights.20-0.53.hdf5  ------
0.765490196078
[[235   0   0   1   0   1   2   1   0   0   0]
 [  0 135   0   4   0   0   0   1   0   0   0]
 [  0   0 211   0   0   0   0  19  19   1   0]
 [  0   2   0 216   0   1   0   0   1   0   0]
 [  0   0   2   0  87   0   0   0   8   2   1]
 [  0   0   0   0   0  87   0   0   9   3   1]
 [  0   0  25   0   0   0 147  61   5   2   0]
 [  0   0  57   0   0   0  20 160   8   0   0]
 [  0   0  16   0   0   0   4  15 256  16  13]
 [  0   0  24   0   0   0  10   8  83 223  12]
 [  1   0   1   0   0   0   0   0 119  19 195]]
------ TEST ACCURACY:  weights.20-0.53.hdf5  ------
0.52380952381
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0]
 [ 0  8  0  7  0  0  0  0  0]
 [ 0  0  4  0  7  1  3  0  0]
 [ 0  0  1  0  2 12  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  5  0  2  1 18  4  0]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/6/weights.16-0.49.hdf5
------ TRAIN ACCURACY:  weights.16-0.49.hdf5  ------
0.778119001919
[[246   1   0   0   0   0   2   0   0   1   0]
 [  1 137   0   2   0   0   0   0   0   0   0]
 [  0   0 223   0   0   0   3  17  10   6   1]
 [  0   5   0 225   0   2   0   0   1   2   0]
 [  1   0   0   0  88   0   0   0   5   4   2]
 [  2   0   0   0   5  80   0   0   4   3   6]
 [  0   0  30   0   1   0 156  55   5   5   3]
 [  0   0  62   0   0   0  22 158   7   0   1]
 [  1   0  24   0   1   0   5   5 192  40  42]
 [  1   0  14   0   1   0  10   6  41 248  29]
 [  3   0   4   0   1   2   0   1  39  31 274]]
------ TEST ACCURACY:  weights.16-0.49.hdf5  ------
0.42
[[5 0 0 0 0 0 0]
 [0 4 0 0 0 0 1]
 [0 0 0 0 0 0 0]
 [1 0 1 3 2 3 0]
 [2 1 0 0 4 1 2]
 [0 2 0 0 3 4 1]
 [0 6 0 0 2 1 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/7/weights.43-0.49.hdf5
------ TRAIN ACCURACY:  weights.43-0.49.hdf5  ------
0.821611001965
[[236   2   0   0   0   0   2   0   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  0   1 140   0   0   0  35  39  18  12   5]
 [  0  25   0 193   0   2   0   0   0   0   0]
 [  0   1   0   0  74   0   1   2   5   0   2]
 [  0   2   0   0   0  76   0   0   3   2   2]
 [  0   0   2   0   0   0 229   1   0   3   0]
 [  0   0  10   0   0   0  72 154   8   0   1]
 [  1   3   3   0   0   0  17   4 236  41  15]
 [  0   5   1   0   0   0  20   4   8 314   8]
 [  2   2   1   0   0   1   4   0  23  33 299]]
------ TEST ACCURACY:  weights.43-0.49.hdf5  ------
0.554545454545
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  7  0  0  0  0  5  2  1  0]
 [ 0  1  0  7  0  2  0  0  1  3  1]
 [ 3  0  0  0  2  1  0  0  6  2  1]
 [ 1  0  0  0  0  2  0  0  2  1  9]
 [ 0  0  0  0  0  0 19  0  0  1  0]
 [ 0  0  0  0  0  0  6  9  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/8/weights.41-0.44.hdf5
------ TRAIN ACCURACY:  weights.41-0.44.hdf5  ------
0.86484375
[[234   0   0   1   0   1   1   1   1   1   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  0   0 231   0   0   0   1  12   1   4   1]
 [  0   1   0 216   0   2   0   0   0   1   0]
 [  0   0   0   0  87   0   0   1   1   1   0]
 [  0   0   0   0   0  85   0   0   0   2   3]
 [  0   0  11   0   0   0 183  41   0   4   1]
 [  0   0  31   0   0   0   9 204   0   0   1]
 [  0   0  12   0   0   0   4  14 192  68  30]
 [  0   0   5   0   1   0   4  11   3 330   6]
 [  0   0   5   0   0   0   0   1   6  41 312]]
------ TEST ACCURACY:  weights.41-0.44.hdf5  ------
0.484210526316
[[14  0  0  0  1  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  0  2  1  1  0]
 [ 0  8  0  7  0  0  0  0  0  0  0]
 [ 0  0  0  0  3  0  0  0  2  3  2]
 [ 0  0  1  2  0  3  0  0  0  2  2]
 [ 0  0  1  0  0  0  4 10  0  0  0]
 [ 0  0 10  0  0  0  0  4  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/9/weights.36-0.45.hdf5
------ TRAIN ACCURACY:  weights.36-0.45.hdf5  ------
0.825858585859
[[237   0   0   0   0   1   2   0   0   0   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  0   0 205   0   0   0  22  16   3   1   3]
 [  1   1   0 215   0   2   0   0   0   1   0]
 [  0   0   0   0  95   0   2   0   1   0   2]
 [  0   0   0   0   0  98   0   0   0   0   2]
 [  0   0   0   0   0   0 236   3   0   0   1]
 [  0   0  24   0   0   0  92 127   1   0   1]
 [  1   0   9   0   0   0  27   6 170  12  65]
 [  0   0   3   0   1   0  28   2   8 224  64]
 [  4   0   3   0   1   2   7   2   3   1 312]]
------ TEST ACCURACY:  weights.36-0.45.hdf5  ------
0.516666666667
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  1  0  0]
 [ 0 11  0  4  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0 14  0  0  0  0]
 [ 0  0  1  0  0 12  2  0  0  0]
 [ 0  0  5  0  0  3  0  4  4 14]
 [ 0  0  2  0  0  5  0  4  5 14]
 [ 1  0  0  0  1  0  0  6  2 20]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/10/weights.20-0.51.hdf5
------ TRAIN ACCURACY:  weights.20-0.51.hdf5  ------
0.799215686275
[[235   0   0   2   0   0   1   2   0   0   0]
 [  0 128   0  11   0   0   0   1   0   0   0]
 [  0   0 201   0   0   0   2  24  15   4   4]
 [  0   0   0 218   0   1   0   0   0   1   0]
 [  0   0   0   0  80   0   0   0   3   1   1]
 [  0   0   0   0   0  80   0   0   2   1   2]
 [  0   0  19   0   0   0 156  56   7   2   0]
 [  0   0  46   0   0   0  11 183   4   0   1]
 [  1   0  11   0   1   0  10  10 219  28  40]
 [  0   0  15   0   1   0  10   9  28 252  45]
 [  1   0   7   0   0   3   8   1  40  19 286]]
------ TEST ACCURACY:  weights.20-0.51.hdf5  ------
0.47619047619
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  2  0  0  0  0  0  9  1  3]
 [ 3  0 12  0  0  0  0  0  0  0]
 [ 2  0  0  5  7  0  0  0  0  1]
 [ 1  0  0  2 12  0  0  0  0  0]
 [ 0  1  0  0  0  4 10  0  0  0]
 [ 0 10  0  0  0  0  0  4  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/11/weights.36-0.51.hdf5
------ TRAIN ACCURACY:  weights.36-0.51.hdf5  ------
0.85831702544
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  0   0 195   0   1   0   0  30   8   7   4]
 [  0   0   0 217   0   2   0   0   0   1   0]
 [  0   0   0   0  87   0   0   0   2   0   1]
 [  0   0   0   0   0  89   0   0   0   0   1]
 [  0   0   3   0   2   0 167  57   4   6   1]
 [  0   0  29   0   0   0   8 203   3   1   1]
 [  1   0   8   0   0   0   3  10 226  42  30]
 [  0   0   3   0   1   0   0  12   8 323  13]
 [  1   0   1   0   1   2   0   0  19  33 308]]
------ TEST ACCURACY:  weights.36-0.51.hdf5  ------
0.58
[[15  0  0  0  0  0  0  0  0]
 [ 0 10  0  0  0  0  2  1  7]
 [ 0  0 15  0  0  0  0  0  0]
 [ 0  0  0 10  0  0  0  0  0]
 [ 2  0  0  2  6  0  0  0  0]
 [ 0  3  0  0  0  0 12  0  0]
 [ 0 11  0  0  0  0  2  2  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/12/weights.22-0.48.hdf5
------ TRAIN ACCURACY:  weights.22-0.48.hdf5  ------
0.781563126253
[[238   0   0   0   0   0   1   1   0   0   0]
 [  0 130   0  10   0   0   0   0   0   0   0]
 [  2   0 213   0   0   0   3  19   5   0   3]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  2   0   0   0  80   0   0   0   2   0   1]
 [  0   0   0   0   0  77   0   0   1   0   2]
 [  1   0  17   0   1   0 183  35   2   0   1]
 [  2   0  58   0   0   0  24 159   1   0   1]
 [  7   0  26   0   2   1  12  13 176   5  63]
 [ 15   0  27   0   3   0  21   5  23 170  81]
 [  8   0  12   0   1   4   6   3  11   0 305]]
------ TEST ACCURACY:  weights.22-0.48.hdf5  ------
0.5125
[[15  0  0  0  0  0  0  0  0  0]
 [ 3 16  0  0  0  0  0  0  1  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [13  0  0  0  2  0  0  0  0  0]
 [10  0  7  0  2  0  0  0  0  1]
 [ 0  1  0  0  0 13  1  0  0  0]
 [ 0  3  0  0  0  3  9  0  0  0]
 [ 4  0  0  0  0  0  0  2  3  6]
 [ 2  3  0  0  0  1  0  4  1  4]
 [ 1  0  0  1  0  0  1  2  1  9]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/13/weights.48-0.45.hdf5
------ TRAIN ACCURACY:  weights.48-0.45.hdf5  ------
0.86484375
[[244   0   0   0   0   1   0   0   0   0   0]
 [  0 135   0   0   0   0   0   0   0   0   0]
 [  0   0 244   0   0   0   7   4   2   2   1]
 [  0   0   0 227   0   2   0   0   0   1   0]
 [  0   0   0   0  98   0   0   0   1   0   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0   4   0   0   0 242   2   0   2   0]
 [  0   0  46   0   0   0  76 131   1   0   1]
 [  1   0  13   0   0   0  11   3 188  34  50]
 [  0   0   7   0   1   0  15   1   3 283  30]
 [  1   0   3   0   0   2   2   0   7   7 323]]
------ TEST ACCURACY:  weights.48-0.45.hdf5  ------
0.494736842105
[[ 5  0  0  0  4  1  0  0  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  0  1  0]
 [ 0  3  0  2  0  0  0  0  0]
 [ 0  0  0  0  3  0  0  1  1]
 [ 0  0  0  0  3  2  0  0  0]
 [ 0  0  1  0  0  0  1 11  7]
 [ 0  0  1  0  0  0  1 12  6]
 [ 0  0  0  0  0  0  4  3 13]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/14/weights.22-0.48.hdf5
------ TRAIN ACCURACY:  weights.22-0.48.hdf5  ------
0.786538461538
[[242   0   0   1   0   1   3   0   0   3   0]
 [  0 123   0  15   0   0   0   1   0   1   0]
 [  0   0 194   0   0   0   2  25  20  10   4]
 [  0   0   0 231   0   2   0   0   0   2   0]
 [  0   0   0   0  80   0   0   0  12   5   3]
 [  0   0   0   0   0  77   0   0   9   4  10]
 [  0   0  15   0   0   0 161  55  13  10   1]
 [  0   0  40   0   0   0  15 178  12   3   2]
 [  0   0   4   0   0   0   3   7 215  47  39]
 [  0   0   7   0   0   0   6   4  38 273  17]
 [  0   0   1   0   0   0   0   0  43  40 271]]
------ TEST ACCURACY:  weights.22-0.48.hdf5  ------
0.545454545455
[[4 0 0 0 0 1 0]
 [0 7 0 3 0 0 0]
 [0 0 0 0 0 0 0]
 [0 1 0 7 1 1 0]
 [0 1 0 0 2 1 1]
 [0 0 4 0 0 7 4]
 [0 0 0 0 4 3 3]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/15/weights.41-0.47.hdf5
------ TRAIN ACCURACY:  weights.41-0.47.hdf5  ------
0.856470588235
[[242   0   0   0   1   0   1   0   0   1   0]
 [  0 127   0   8   0   0   0   0   0   0   0]
 [  0   0 248   0   1   0   0   6   3   1   1]
 [  0   0   0 224   0   1   0   0   0   0   0]
 [  0   0   0   0  97   0   0   0   2   0   1]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  0   0  30   0   2   0 163  42   3   4   1]
 [  0   0  60   0   0   0   6 180   2   1   1]
 [  0   0  18   0   1   1   2   2 217  21  38]
 [  0   0  11   0   1   0   0   8  17 280  28]
 [  1   0   7   0   1   4   0   0  12  13 307]]
------ TEST ACCURACY:  weights.41-0.47.hdf5  ------
0.533333333333
[[ 9  0  0  0  1  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  5  0  0  0]
 [ 0  0  0 10  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  3  0  0  5  2  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  2  0  0  1  1  3  5  8]
 [ 1  0  1  0  0  0  0  2  7  4]
 [ 0  0  1  0  0  0  1  3  3 12]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/16/weights.34-0.47.hdf5
------ TRAIN ACCURACY:  weights.34-0.47.hdf5  ------
0.799513381995
[[212   0   0   0   1   0   2   0   0   0   0]
 [  0  98   0   2   0   0   0   0   0   0   0]
 [  3   0 119   0   0   0  65   7   8   6  12]
 [  0   2   0 189   0   2   0   0   0   0   2]
 [  0   0   0   0  63   0   0   0   1   0   1]
 [  0   0   0   0   0  68   0   0   1   0   1]
 [  1   0   1   0   0   0 207   0   0   0   1]
 [  1   0  15   0   0   0 115  77   4   0   3]
 [  1   0   2   0   0   0  17   2 163  16  34]
 [  1   0   3   0   0   0  21   0   9 197  29]
 [  5   0   1   0   0   2   1   0   9   2 250]]
------ TEST ACCURACY:  weights.34-0.47.hdf5  ------
0.515
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 0 40  0  0  0  0  0  0  0  0  0]
 [ 0  0 20  0  0  0 22  1  0  1  1]
 [ 5  3  0 32  0  0  0  0  0  0  0]
 [ 3  1  0  0 19  0  4  0  3  3  2]
 [ 0  0  0  0  1  6  0  0  4  3 16]
 [ 0  0  0  0  0  0 44  1  0  0  0]
 [ 0  0  2  0  0  0 31 11  0  1  0]
 [ 0  0  0  0  0  0 17  0 29 15 24]
 [ 3  1  2  0  0  0 12  0 20 31 31]
 [ 3  0  0  0  0  0  2  0 15 38 37]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_kpca_conv1d_10/17/weights.23-0.42.hdf5
------ TRAIN ACCURACY:  weights.23-0.42.hdf5  ------
0.793939393939
[[237   0   0   0   0   0   2   0   0   1   0]
 [  0 119   0   6   0   0   0   0   0   0   0]
 [  0   0 198   0   0   0   2  25   7  14   4]
 [  0   0   0 218   0   0   0   0   0   1   1]
 [  1   1   0   0  82   2   0   0   6   5   3]
 [  0   0   0   0   0  86   0   0   2   6   6]
 [  0   0  12   0   0   0 177  42   3   5   1]
 [  0   0  39   0   0   0  32 163   8   1   2]
 [  0   0   4   0   0   1   8   2 205  40  30]
 [  0   0   3   0   0   0  17   2  33 260  15]
 [  1   0   1   0   0   0   5   0  32  76 220]]
------ TEST ACCURACY:  weights.23-0.42.hdf5  ------
0.455555555556
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  4  0 11  0  0  0  0  0  0]
 [ 0  0  3  0  0  1  2  5  4  0]
 [ 0  0  0 13  2  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 10  3  0  2  0]
 [ 0  0  4  0  0  4  5  2  0  0]
 [ 0  0  1  0  0  1  4 10 11  3]
 [ 0  0  4  0  0  8  2  3 10  3]
 [ 1  0  0  0  2  0  0  8  7 12]]
[0.5263157894736842, 0.6266666666666667, 0.4864864864864865, 0.5894736842105263, 0.5238095238095238, 0.42, 0.5545454545454546, 0.4842105263157895, 0.5166666666666667, 0.47619047619047616, 0.58, 0.5125, 0.49473684210526314, 0.5454545454545454, 0.5333333333333333, 0.515, 0.45555555555555555]
0.520055620636
0.0491604879674

Process finished with exit code 0

'''