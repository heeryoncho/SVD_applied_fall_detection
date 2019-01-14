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
X_raw = pickle.load(open("data/X_umafall_raw.p", "rb"))
X = np.concatenate((X_kpca, X_raw), axis=1)

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

    new_dir = 'model/umafall_raw_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_raw_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.6105263157894737, 0.56, 0.6594594594594595, 0.6578947368421053, 0.5904761904761905, 0.68, 0.8545454545454545, 0.6736842105263158, 0.6555555555555556, 0.638095238095238, 0.59, 0.53125, 0.6947368421052632, 0.6181818181818182, 0.6666666666666666, 0.7466666666666667, 0.46111111111111114]
0.640520603884
0.0844756884228
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_raw_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/1/weights.46-0.61.hdf5
------ TRAIN ACCURACY:  weights.46-0.61.hdf5  ------
0.941987829615
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 122   0   3   0   0   0   0   0   0   0]
 [  3   0 243   0   0   0   0   4   0   0   0]
 [  0   0   0 219   0   1   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 220  17   1   2   0]
 [  0   0  17   0   0   1   4 222   0   0   1]
 [  3   0   1   0   1   0   1   1 254  21   8]
 [  0   0   0   0   0   0  10   0   8 289  13]
 [  0   0   0   0   0   0   0   0   3  19 313]]
------ TEST ACCURACY:  weights.46-0.61.hdf5  ------
0.610526315789
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 10  0  5  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  3  0  0  0]
 [ 0  0  0 14  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  9  0  6  0]
 [ 0  0  0  0  0  0 12  0  3  0]
 [ 0  0  0  0  0  3  0 24  2  1]
 [ 0  0  0  0  0  0  0  9 29  2]
 [ 0  0  0  0  0  0  0  0 30  0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/2/weights.38-0.64.hdf5
------ TRAIN ACCURACY:  weights.38-0.64.hdf5  ------
0.926946107784
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 121   0   4   0   0   0   0   0   0   0]
 [  1   0 244   0   0   0   0   5   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   2   0  96   0   0   2   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   1   0   0   0 227   5   6   1   0]
 [  0   0  21   0   0   0  16 206   0   1   1]
 [  2   0   0   0   0   0   6   2 260  26   9]
 [  0   0   0   0   0   0   6   0  21 293  10]
 [  0   0   0   0   0   0   1   0   5  29 315]]
------ TEST ACCURACY:  weights.38-0.64.hdf5  ------
0.56
[[15  0  0  0  0  0  0  0  0]
 [ 2 13  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0]
 [ 0  6  0  9  0  0  0  0  0]
 [ 0  0  5  0  4  5  1  0  0]
 [ 0  0 13  0  1  1  0  0  0]
 [ 0  0  0  0  0  0  2  3 10]
 [ 0  0  0  0  0  0  0 25  5]
 [ 0  0  0  0  0  0 13  1  1]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/3/weights.49-0.65.hdf5
------ TRAIN ACCURACY:  weights.49-0.65.hdf5  ------
0.939271255061
[[239   0   0   0   0   0   0   0   0   0   1]
 [  0 121   0   4   0   0   0   0   0   0   0]
 [  2   0 236   0   0   0   0   7   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   0   0   0   0 224  10   2   2   2]
 [  0   0  27   0   0   0   3 215   0   0   0]
 [  2   0   2   0   0   0   3   1 263  12   7]
 [  0   0   0   0   0   0   5   0  19 287  19]
 [  0   0   0   0   0   0   0   0   3  17 315]]
------ TEST ACCURACY:  weights.49-0.65.hdf5  ------
0.659459459459
[[15  0  0  0  0  0  0  0  0]
 [ 4 10  0  0  0  1  0  0  0]
 [ 0  0 17  0  0  3  0  0  0]
 [ 0  1  0 14  0  0  0  0  0]
 [ 0  0  0  0  8  1  4  0  2]
 [ 0  0  0  0  5 10  0  0  0]
 [ 0  0  0  0  0  0 14  0 16]
 [ 0  0  0  0  0  0  7  5 18]
 [ 0  0  0  0  0  0  1  0 29]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/4/weights.28-0.69.hdf5
------ TRAIN ACCURACY:  weights.28-0.69.hdf5  ------
0.898174442191
[[233   0   0   0   0   0   0   2   0   0   0]
 [  0 123   0   1   0   0   0   1   0   0   0]
 [  5   0 175   0   1   3   0  64   2   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0 100   0   0   0   0   0   0]
 [  1   0   0   0   0  99   0   0   0   0   0]
 [  0   0   0   0   0   0 193  26   5   5   1]
 [  0   0   4   0   0   1   8 230   0   1   1]
 [  1   0   0   0   0   0   2   0 241  30  16]
 [  0   0   0   0   0   0  10   1   0 314  10]
 [  0   0   0   0   0   0   0   1   4  44 286]]
------ TEST ACCURACY:  weights.28-0.69.hdf5  ------
0.657894736842
[[20  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  4  0  0  0  0  0  0]
 [ 1  0 12  0  0  0  2  0  0  0]
 [ 0  6  0  9  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 18  3  2  1  1]
 [ 0  0  0  0  0  4 11  0  0  0]
 [ 2  0  0  0  1  0  2 11 13  1]
 [ 0  0  0  0  0  1  0  1 19  4]
 [ 0  0  0  0  0  0  0  7  9 14]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/5/weights.32-0.62.hdf5
------ TRAIN ACCURACY:  weights.32-0.62.hdf5  ------
0.906666666667
[[239   0   1   0   0   0   0   0   0   0   0]
 [  0 138   0   2   0   0   0   0   0   0   0]
 [  3   0 235   0   0   0   3   4   4   0   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   1   0  99   0   0   0   0   0   0]
 [  1   0   0   0   0  99   0   0   0   0   0]
 [  0   0   1   0   0   0 229   5   3   0   2]
 [  0   0  33   0   0   0  30 181   1   0   0]
 [  2   0   1   0   2   0   7   1 274  12  21]
 [  0   0   0   0   0   0  14   0  21 285  40]
 [  0   0   0   0   0   0   0   0   3  19 313]]
------ TEST ACCURACY:  weights.32-0.62.hdf5  ------
0.590476190476
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0  0]
 [ 0  6  0  9  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  8  0  0  1  6]
 [ 0  0  2  0  0  4  9  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  1  2  2  4 14  7]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/6/weights.29-0.65.hdf5
------ TRAIN ACCURACY:  weights.29-0.65.hdf5  ------
0.908253358925
[[246   0   2   0   0   0   0   1   0   0   1]
 [  0 134   0   6   0   0   0   0   0   0   0]
 [  0   0 255   0   0   0   2   3   0   0   0]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   5   0  95   0   0   0   0   0   0]
 [  0   0   1   0   1  98   0   0   0   0   0]
 [  0   0   3   0   0   0 214  26   7   4   1]
 [  0   0  42   0   0   0   4 202   0   1   1]
 [  0   0   6   0   0   0   4   1 261  23  15]
 [  0   0   0   0   0   0   7   1  13 316  13]
 [  0   0   1   0   0   0   0   0  11  33 310]]
------ TEST ACCURACY:  weights.29-0.65.hdf5  ------
0.68
[[ 4  1  0  0  0  0]
 [ 0  5  0  0  0  0]
 [ 1  3  6  0  0  0]
 [ 0  0  0 10  0  0]
 [ 0  0  0  5  5  0]
 [ 0  0  0  0  6  4]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/7/weights.48-0.66.hdf5
------ TRAIN ACCURACY:  weights.48-0.66.hdf5  ------
0.937131630648
[[239   0   0   0   0   0   0   0   0   0   1]
 [  0 137   0   2   0   0   0   0   0   0   1]
 [  2   0 244   0   0   0   0   3   1   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  85   0   0   0   0   0   0]
 [  0   0   0   0   0  85   0   0   0   0   0]
 [  0   0   2   0   0   0 214  15   2   2   0]
 [  0   0  42   0   0   0   1 201   1   0   0]
 [  2   0   1   0   1   0   2   0 284  18  12]
 [  0   0   0   0   0   0   6   0  11 334   9]
 [  0   0   0   0   0   0   0   0   5  18 342]]
------ TEST ACCURACY:  weights.48-0.66.hdf5  ------
0.854545454545
[[15  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 11  3  0  0  0  0  1]
 [ 2  0  0 12  1  0  0  0  0]
 [ 0  0  0  2 12  0  0  0  1]
 [ 0  0  0  0  0 19  0  1  0]
 [ 0  0  0  0  0  5 10  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/8/weights.40-0.65.hdf5
------ TRAIN ACCURACY:  weights.40-0.65.hdf5  ------
0.9234375
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 136   0   3   0   0   0   1   0   0   0]
 [  5   0 237   0   0   0   0   8   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  90   0   0   0   0   0   0]
 [  0   0   0   0   0  90   0   0   0   0   0]
 [  0   0   0   0   0   0 220  10   6   3   1]
 [  0   0  17   0   0   0   4 223   0   1   0]
 [  2   0   0   0   2   0   1   1 281  23  10]
 [  0   0   0   0   0   0   6   0  42 299  13]
 [  0   0   1   0   0   0   0   0   9  27 328]]
------ TEST ACCURACY:  weights.40-0.65.hdf5  ------
0.673684210526
[[15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 1  0 13  0  0  0  0  1]
 [ 1  4  0 10  0  0  0  0]
 [ 0  0  2  0  7  1  0  0]
 [ 0  0  2  0  1  7  0  0]
 [ 0  0  0  0  0  0  7  8]
 [ 0  0 10  0  0  0  0  5]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/9/weights.45-0.65.hdf5
------ TRAIN ACCURACY:  weights.45-0.65.hdf5  ------
0.924848484848
[[240   0   0   0   0   0   0   0   0   0   0]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  4   0 239   0   0   0   0   4   3   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   1   0  99   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   1   0   0   0 213  19   0   7   0]
 [  0   0  19   0   0   0   3 222   1   0   0]
 [  1   0   1   0   1   0   2   2 241  33   9]
 [  0   0   0   0   0   0   4   0   5 315   6]
 [  0   0   1   0   0   0   0   0   4  54 276]]
------ TEST ACCURACY:  weights.45-0.65.hdf5  ------
0.655555555556
[[15  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  1  0  0]
 [ 0  9  0  6  0  0  0  0  0]
 [ 0  0  0  0 10  0  1  4  0]
 [ 0  0  2  0 10  2  1  0  0]
 [ 0  0  0  0  0  0 23  2  5]
 [ 0  0  0  0  0  0  0 30  0]
 [ 0  0  0  0  0  0  1 26  3]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/10/weights.25-0.65.hdf5
------ TRAIN ACCURACY:  weights.25-0.65.hdf5  ------
0.883137254902
[[237   0   0   0   0   0   0   3   0   0   0]
 [  0 135   0   4   0   0   0   1   0   0   0]
 [  0   0 222   0   0   1   0  26   0   0   1]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   1   0  78   6   0   0   0   0   0]
 [  0   0   0   0   0  85   0   0   0   0   0]
 [  0   0   0   0   0   0 209  22   6   1   2]
 [  0   0  19   0   0   0   8 217   0   0   1]
 [  1   0   4   0   0   0   5   2 264  14  30]
 [  0   0   0   0   0   0  15   2  32 247  64]
 [  0   0   0   0   0   0   0   1  10  16 338]]
------ TEST ACCURACY:  weights.25-0.65.hdf5  ------
0.638095238095
[[14  0  0  0  1  0  0]
 [ 1  8  0  0  6  0  0]
 [ 0  0 15  0  0  0  0]
 [ 1  0  0  3 11  0  0]
 [ 0  0  0  1 14  0  0]
 [ 0  0  0  0  0 13  2]
 [ 0 14  0  0  1  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/11/weights.13-0.66.hdf5
------ TRAIN ACCURACY:  weights.13-0.66.hdf5  ------
0.827397260274
[[238   0   1   0   0   0   0   1   0   0   0]
 [  1 135   0   4   0   0   0   0   0   0   0]
 [  3   0 231   0   0   0   2   9   0   0   0]
 [  1   1   0 218   0   0   0   0   0   0   0]
 [  7   0  10   0  64   9   0   0   0   0   0]
 [  3   0   2   0   0  85   0   0   0   0   0]
 [  0   0   1   0   0   0 153  55  18   2  11]
 [  2   0  52   0   0   0   3 186   0   1   1]
 [  2   0   5   0   0   0   4   1 249  30  29]
 [  0   0   0   0   0   0   2   4  29 225 100]
 [  0   0   2   0   0   0   1   1  19  12 330]]
------ TEST ACCURACY:  weights.13-0.66.hdf5  ------
0.59
[[15  0  0  0  0  0  0]
 [ 0 19  0  0  0  0  1]
 [ 0  0 15  0  0  0  0]
 [ 2  0  0  2  6  0  0]
 [ 2  0  0  0  8  0  0]
 [ 0  2  0  0  0  0 13]
 [ 0 15  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/12/weights.17-0.62.hdf5
------ TRAIN ACCURACY:  weights.17-0.62.hdf5  ------
0.842084168337
[[238   0   1   0   0   0   0   1   0   0   0]
 [  0 138   0   1   0   0   0   1   0   0   0]
 [  5   0 209   0   0   0   5  22   4   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  5   0   4   0  75   1   0   0   0   0   0]
 [  1   0   3   0   0  76   0   0   0   0   0]
 [  0   0   0   0   0   0 218  13   9   0   0]
 [  2   0  38   0   0   0  24 181   0   0   0]
 [  2   0   1   0   0   0  12   4 248  32   6]
 [  0   0   0   0   0   0  18   2  21 293  11]
 [  0   0   0   0   0   0   9   2  78  56 205]]
------ TEST ACCURACY:  weights.17-0.62.hdf5  ------
0.53125
[[15  0  0  0  0  0  0  0  0  0]
 [ 3 16  0  0  1  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [12  0  0  0  3  0  0  0  0  0]
 [ 8  0  9  0  3  0  0  0  0  0]
 [ 0  0  0  0  0 15  0  0  0  0]
 [ 0  2  0  0  0  2 11  0  0  0]
 [ 0  0  0  0  0  0  0  0 15  0]
 [ 0  0  0  0  0  6  0  5  4  0]
 [ 0  0  0  0  0  1  0  8  0  6]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/13/weights.43-0.66.hdf5
------ TRAIN ACCURACY:  weights.43-0.66.hdf5  ------
0.927734375
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 135   0   0   0   0   0   0   0   0   0]
 [  0   0 251   0   0   0   1   8   0   0   0]
 [  0   0   0 230   0   0   0   0   0   0   0]
 [  0   0   3   0  97   0   0   0   0   0   0]
 [  0   0   1   0   0  99   0   0   0   0   0]
 [  0   0   1   0   0   0 236  11   2   0   0]
 [  0   0  24   0   0   0   8 223   0   0   0]
 [  0   0   2   0   0   0   7   0 261  16  14]
 [  0   0   0   0   0   0  20   1  20 284  15]
 [  0   0   0   0   0   0   0   0   9  22 314]]
------ TEST ACCURACY:  weights.43-0.66.hdf5  ------
0.694736842105
[[ 5  0  3  0  0  2  0  0  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  1  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  0  0  2  0  3  0  0]
 [ 0  0  2  0  0  3  0  0  0]
 [ 0  0  0  0  0  0 16  3  1]
 [ 0  0  0  0  0  0  0  7 13]
 [ 0  0  0  0  0  0  1  0 19]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/14/weights.23-0.66.hdf5
------ TRAIN ACCURACY:  weights.23-0.66.hdf5  ------
0.898461538462
[[249   0   1   0   0   0   0   0   0   0   0]
 [  0 140   0   0   0   0   0   0   0   0   0]
 [  0   0 250   0   1   0   1   2   0   0   1]
 [  0   6   0 229   0   0   0   0   0   0   0]
 [  3   0   1   0  95   1   0   0   0   0   0]
 [  3   0   1   0   0  96   0   0   0   0   0]
 [  0   0   2   0   0   0 226  11   8   0   8]
 [  2   0  53   0   0   0  10 184   0   1   0]
 [  2   0   1   0   2   0   4   0 263  25  18]
 [  0   0   0   0   0   0   8   0  20 285  32]
 [  0   0   1   0   0   0   0   0   7  28 319]]
------ TEST ACCURACY:  weights.23-0.66.hdf5  ------
0.618181818182
[[5 0 0 0 0 0 0]
 [0 5 0 5 0 0 0]
 [0 0 0 0 0 0 0]
 [0 3 0 7 0 0 0]
 [0 0 0 0 3 0 2]
 [0 0 1 2 1 7 4]
 [0 0 1 0 0 2 7]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/15/weights.31-0.61.hdf5
------ TRAIN ACCURACY:  weights.31-0.61.hdf5  ------
0.896078431373
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 131   0   4   0   0   0   0   0   0   0]
 [  4   0 254   0   0   0   0   2   0   0   0]
 [  0   0   0 225   0   0   0   0   0   0   0]
 [  1   0   2   0  96   1   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   2   0   0   0 213  22   7   1   0]
 [  0   0  48   0   0   0   7 194   0   0   1]
 [  2   0   5   0   0   0   5   1 265  13   9]
 [  0   0   0   0   0   0   7   0  50 253  35]
 [  0   0   1   0   0   0   0   0  19  16 309]]
------ TEST ACCURACY:  weights.31-0.61.hdf5  ------
0.666666666667
[[10  0  0  0  0  0  0  0  0  0]
 [ 0  1  0  4  0  0  0  0  0  0]
 [ 0  0  5  0  0  0  0  0  0  0]
 [ 2  4  0  3  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  9  1  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  2  1 13  4  0]
 [ 0  0  0  0  0  0  0 11  1  3]
 [ 0  0  0  0  0  1  1  0  0 18]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/16/weights.42-0.51.hdf5
------ TRAIN ACCURACY:  weights.42-0.51.hdf5  ------
0.915328467153
[[215   0   0   0   0   0   0   0   0   0   0]
 [  0  99   0   1   0   0   0   0   0   0   0]
 [  5   0 201   0   1   0   0  13   0   0   0]
 [  0   0   0 195   0   0   0   0   0   0   0]
 [  0   0   0   0  65   0   0   0   0   0   0]
 [  0   0   0   0   0  70   0   0   0   0   0]
 [  0   0   0   0   0   0 198   7   2   2   1]
 [  0   0  11   0   1   0  10 191   0   1   1]
 [  2   0   0   0   1   0   4   3 181  21  23]
 [  0   0   0   0   0   0  12   0   8 222  18]
 [  0   0   0   0   0   0   0   0   1  25 244]]
------ TEST ACCURACY:  weights.42-0.51.hdf5  ------
0.746666666667
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 0 35  0  5  0  0  0  0  0  0  0]
 [ 0  0 21  0  0  0  0 24  0  0  0]
 [ 5  0  0 35  0  0  0  0  0  0  0]
 [ 5  0  0  0 26  3  0  1  0  0  0]
 [ 0  0  0  0 18 12  0  0  0  0  0]
 [ 0  0  0  0  0  0 39  6  0  0  0]
 [ 0  0  3  0  0  0  5 37  0  0  0]
 [ 0  0  2  0  0  0  6  0 72  1  4]
 [ 1  0  0  0  0  0  4  1 11 77  6]
 [ 0  0  0  0  0  0  7  0 18 16 54]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_kpca_conv1d_10/17/weights.20-0.81.hdf5
------ TRAIN ACCURACY:  weights.20-0.81.hdf5  ------
0.903434343434
[[237   0   1   0   0   0   0   2   0   0   0]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  0   0 241   0   1   0   0   8   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   2   0  98   0   0   0   0   0   0]
 [  1   0   1   0   1  97   0   0   0   0   0]
 [  0   0   0   0   0   0 218   6  10   3   3]
 [  0   0  48   0   0   0  22 173   0   2   0]
 [  1   0   1   0   1   0   1   3 265   8  10]
 [  0   0   0   0   0   0   3   3  26 287  11]
 [  0   0   1   0   0   0   1   1  39  17 276]]
------ TEST ACCURACY:  weights.20-0.81.hdf5  ------
0.461111111111
[[15  0  0  0  0  0  0  0  0]
 [ 0  7  0  8  0  0  0  0  0]
 [ 0  0 13  0  0  2  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  6  6  1  2  0]
 [ 0  0  3  0  0 11  0  0  1]
 [ 0  0  0  0  2  0  6 18  4]
 [ 0  0  0  0  6  0 23  0  1]
 [ 0  0  0  0  0  0  0 20 10]]
[0.6105263157894737, 0.56, 0.6594594594594595, 0.6578947368421053, 0.5904761904761905, 0.68, 0.8545454545454545, 0.6736842105263158, 0.6555555555555556, 0.638095238095238, 0.59, 0.53125, 0.6947368421052632, 0.6181818181818182, 0.6666666666666666, 0.7466666666666667, 0.46111111111111114]
0.640520603884
0.0844756884228

Process finished with exit code 0

'''