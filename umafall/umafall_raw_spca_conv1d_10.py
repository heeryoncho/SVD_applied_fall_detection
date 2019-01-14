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
X_raw = pickle.load(open("data/X_umafall_raw.p", "rb"))
X = np.concatenate((X_spca, X_raw), axis=1)

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

    new_dir = 'umafall/model/umafall_raw_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_raw_spca_conv1d_10/' + str(i+1) + '/'
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
[0.45263157894736844, 0.44666666666666666, 0.572972972972973, 0.5684210526315789, 0.6761904761904762, 0.7, 0.8090909090909091, 0.8, 0.5777777777777777, 0.6, 0.57, 0.61875, 0.6210526315789474, 0.5272727272727272, 0.4, 0.6283333333333333, 0.45]
0.58936236038
0.112547291288
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_raw_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/1/weights.32-0.45.hdf5
------ TRAIN ACCURACY:  weights.32-0.45.hdf5  ------
0.810953346856
[[202   0   3  18   0   0  17   0   0   0   0]
 [  0  99   0  25   0   0   0   1   0   0   0]
 [  0   0 193   0   0   0  16  41   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  1   0   8   0  90   1   0   0   0   0   0]
 [  0   0   5   0   1  94   0   0   0   0   0]
 [  4   0   0   0   0   0 175  29  13   8  11]
 [  2   0  31   0   0   0  14 187   0   6   5]
 [  0   0   4   0   1   0   3   0 228  29  25]
 [  0   0   0   0   0   0  13   1  22 238  46]
 [  0   0   0   0   0   0   5   2  23  32 273]]
------ TEST ACCURACY:  weights.32-0.45.hdf5  ------
0.452631578947
[[10  0  0  0  3  2  0  0  0]
 [ 0  8  0  7  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  0  7  0  8  0]
 [ 0  0  0  0  0 10  0  5  0]
 [ 0  0  0  0  3  0 26  0  1]
 [ 0  0  0  0  0  0 16  3 21]
 [ 0  0  0  0  0  0  0 30  0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/2/weights.47-0.52.hdf5
------ TRAIN ACCURACY:  weights.47-0.52.hdf5  ------
0.821956087824
[[209   0   1  14   0   0   0  16   0   0   0]
 [  0 104   0  20   0   0   0   1   0   0   0]
 [  0   0 174   0   0   0   0  76   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  93   1   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   0   0   0   0 169  39  19  10   3]
 [  1   0   8   0   0   0   2 227   3   4   0]
 [  0   0   3   0   0   0   0   2 261  29  10]
 [  0   0   0   0   0   0   6   6  43 272   3]
 [  0   0   0   0   0   0   2   2  25  86 235]]
------ TEST ACCURACY:  weights.47-0.52.hdf5  ------
0.446666666667
[[ 5  0  0  3  0  7  0  0  0]
 [ 0  5  0 10  0  0  0  0  0]
 [ 0  0  0  0  0 15  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  0 12  3  0  0]
 [ 0  0  0  0  3 11  0  0  1]
 [ 0  0  0  0  0  0  5  2  8]
 [ 0  0  0  0  0  0  4 25  1]
 [ 0  0  0  0  0  0 14  0  1]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/3/weights.35-0.60.hdf5
------ TRAIN ACCURACY:  weights.35-0.60.hdf5  ------
0.827530364372
[[238   0   2   0   0   0   0   0   0   0   0]
 [  2   0   0 123   0   0   0   0   0   0   0]
 [  0   0 226   0   0   0   0  19   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   7   0  93   0   0   0   0   0   0]
 [  1   0   5   0   3  91   0   0   0   0   0]
 [  0   0   2   0   0   0 225   6   2   2   3]
 [  1   0  36   0   0   0  13 194   0   0   1]
 [  1   0   4   0   0   0   9   1 240  10  25]
 [  0   0   0   0   0   0   7   0  14 228  81]
 [  0   0   0   0   0   0   2   3  14  27 289]]
------ TEST ACCURACY:  weights.35-0.60.hdf5  ------
0.572972972973
[[15  0  0  0  0  0  0  0  0]
 [ 9  0  0  5  0  1  0  0  0]
 [ 0  0 14  0  0  6  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0 10  0  3  0  2]
 [ 0  0  0  0  9  4  0  2  0]
 [ 0  0  0  0  1  0 15  0 14]
 [ 0  0  0  0  0  0  2  8 20]
 [ 0  0  0  0  0  0  3  2 25]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/4/weights.32-0.61.hdf5
------ TRAIN ACCURACY:  weights.32-0.61.hdf5  ------
0.878701825558
[[234   0   1   0   0   0   0   0   0   0   0]
 [  1 122   0   2   0   0   0   0   0   0   0]
 [ 26   0 205   0   0   0   0  18   0   0   1]
 [  0   1   0 219   0   0   0   0   0   0   0]
 [  1   0   7   0  89   3   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  1   0   0   0   0   0 199  13   6   5   6]
 [  3   0  37   0   2   0   2 196   0   2   3]
 [  2   0   0   0   0   0   0   0 242  39   7]
 [  0   0   0   0   0   0   6   1   9 294  25]
 [  0   0   0   0   0   0   1   0   9  54 271]]
------ TEST ACCURACY:  weights.32-0.61.hdf5  ------
0.568421052632
[[20  0  0  0  0  0  0  0  0]
 [ 0 10  0  5  0  0  0  0  0]
 [ 0  0 14  0  0  1  0  0  0]
 [ 0 10  0  5  0  0  0  0  0]
 [ 5  0  0  0 10  0  6  0  4]
 [ 0  0  0  0  4 10  0  0  1]
 [ 0  0  5  0  0  0  9  8  8]
 [ 0  0  0  0  0  0  2 17  6]
 [ 0  0  0  0  0  0  8  9 13]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/5/weights.41-0.57.hdf5
------ TRAIN ACCURACY:  weights.41-0.57.hdf5  ------
0.834509803922
[[240   0   0   0   0   0   0   0   0   0   0]
 [  6   3   0 131   0   0   0   0   0   0   0]
 [  0   0 227   0   0   0   0  23   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  91   2   0   1   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   2   0   0   0 231   4   1   2   0]
 [  0   0  27   0   0   0  19 196   1   2   0]
 [  2   0   0   0   0   0   2   5 236  51  24]
 [  0   0   0   0   0   0  10   0   6 333  11]
 [  0   0   0   0   0   0   0   0  11  68 256]]
------ TEST ACCURACY:  weights.41-0.57.hdf5  ------
0.67619047619
[[15  0  0  0  0  0  0]
 [ 0 10  0  0  5  0  0]
 [ 0  0 15  0  0  0  0]
 [ 0  0  0 11  1  0  3]
 [ 0  3  0  2 10  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0  0  0  6  1 13 10]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/6/weights.50-0.56.hdf5
------ TRAIN ACCURACY:  weights.50-0.56.hdf5  ------
0.863339731286
[[249   0   1   0   0   0   0   0   0   0   0]
 [  4   4   0 132   0   0   0   0   0   0   0]
 [  0   0 255   0   0   0   0   4   1   0   0]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   5   0  95   0   0   0   0   0   0]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0   2   0   0   0 246   5   0   1   1]
 [  0   0  32   0   0   0   7 210   1   0   0]
 [  1   0   5   0   0   0   2   0 248  17  37]
 [  0   0   0   0   0   0   7   0  15 286  42]
 [  0   0   1   0   0   0   1   3   9  20 321]]
------ TEST ACCURACY:  weights.50-0.56.hdf5  ------
0.7
[[ 5  0  0  0  0  0]
 [ 0  5  0  0  0  0]
 [ 4  1  5  0  0  0]
 [ 0  0  0 10  0  0]
 [ 0  0  0  4  6  0]
 [ 0  0  0  0  6  4]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/7/weights.47-0.62.hdf5
------ TRAIN ACCURACY:  weights.47-0.62.hdf5  ------
0.862868369352
[[238   0   2   0   0   0   0   0   0   0   0]
 [  5   1   0 134   0   0   0   0   0   0   0]
 [  0   0 243   0   0   0   0   7   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   6   0  79   0   0   0   0   0   0]
 [  0   0   5   0   0  80   0   0   0   0   0]
 [  0   0   0   0   0   0 219  11   0   1   4]
 [  0   0  21   0   0   0   1 220   0   2   1]
 [  1   0   3   0   0   0   1   2 256  34  23]
 [  0   0   0   0   0   0   3   0   4 336  17]
 [  0   0   0   0   0   0   0   0  10  51 304]]
------ TEST ACCURACY:  weights.47-0.62.hdf5  ------
0.809090909091
[[13  0  0  0  2  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 11  3  1  0  0  0  0]
 [ 1  0  0 11  3  0  0  0  0]
 [ 0  0  0  4 11  0  0  0  0]
 [ 0  0  0  0  0 19  0  1  0]
 [ 0  3  0  0  0  2  9  0  1]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/8/weights.50-0.59.hdf5
------ TRAIN ACCURACY:  weights.50-0.59.hdf5  ------
0.8828125
[[238   0   2   0   0   0   0   0   0   0   0]
 [  1 139   0   0   0   0   0   0   0   0   0]
 [  8   0 202   0   0   0   1  39   0   0   0]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   5   0  85   0   0   0   0   0   0]
 [  0   0   5   0   0  85   0   0   0   0   0]
 [  0   0   0   0   0   0 221  10   5   1   3]
 [  0   0  27   0   2   0   2 211   1   2   0]
 [  2   0   5   0   0   0   2   0 253  32  26]
 [  0   0   0   0   0   0  10   0  17 296  37]
 [  0   0   0   0   0   0   1   0  11  43 310]]
------ TEST ACCURACY:  weights.50-0.59.hdf5  ------
0.8
[[15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0]
 [ 0  1  0 14  0  0  0  0]
 [ 0  0  2  0  4  4  0  0]
 [ 0  0  0  1  1  8  0  0]
 [ 0  0  0  0  0  0  9  6]
 [ 0  0  4  0  0  0  0 11]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/9/weights.11-0.59.hdf5
------ TRAIN ACCURACY:  weights.11-0.59.hdf5  ------
0.724848484848
[[227   0   3   0   8   2   0   0   0   0   0]
 [  4   0   1 120   0   0   0   0   0   0   0]
 [  0   0 246   0   0   0   0   4   0   0   0]
 [  0   0   0 218   1   1   0   0   0   0   0]
 [  2   0  14   0  76   8   0   0   0   0   0]
 [  1   0   5   1   6  87   0   0   0   0   0]
 [  3   0   8   0   0   0 180  15   7   7  20]
 [  0   0  91   0   0   0  52  85   2   8   7]
 [  1   0   5   0   0   0  17   0 193  46  28]
 [  1   0   0   0   0   0   8   0  53 235  33]
 [  0   0   0   0   0   0   6   0  27  55 247]]
------ TEST ACCURACY:  weights.11-0.59.hdf5  ------
0.577777777778
[[15  0  0  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0 14  0  1  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0  4  0  2  4  5]
 [ 0  0  1  0 12  0  1  0  1]
 [ 0  0  0  0  1  0 23  3  3]
 [ 0  0  0  0  0  0  0 27  3]
 [ 0  0  0  0  0  0  0 24  6]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/10/weights.28-0.59.hdf5
------ TRAIN ACCURACY:  weights.28-0.59.hdf5  ------
0.836078431373
[[238   0   2   0   0   0   0   0   0   0   0]
 [  4  93   0  43   0   0   0   0   0   0   0]
 [  0   0 249   0   0   0   0   1   0   0   0]
 [  0  49   0 171   0   0   0   0   0   0   0]
 [  1   0   9   0  75   0   0   0   0   0   0]
 [  0   0   5   0   0  80   0   0   0   0   0]
 [  0   0   6   0   0   0 216  10   2   2   4]
 [  0   0  72   0   0   0   8 162   0   2   1]
 [  0   1   5   0   0   0  18   0 230  40  26]
 [  0   2   0   0   0   0   8   0   5 310  35]
 [  0   0   0   0   0   0   3   3  12  39 308]]
------ TEST ACCURACY:  weights.28-0.59.hdf5  ------
0.6
[[15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0]
 [ 0  4  0 11  0  0  0  0]
 [ 0  0  9  0  3  3  0  0]
 [ 0  0  1  0  7  7  0  0]
 [ 0  0  0  0  0  0 12  3]
 [ 0  0 15  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/11/weights.43-0.63.hdf5
------ TRAIN ACCURACY:  weights.43-0.63.hdf5  ------
0.894716242661
[[237   0   1   1   0   0   1   0   0   0   0]
 [  0 135   0   4   0   0   0   1   0   0   0]
 [  0   0 239   0   0   0   1   5   0   0   0]
 [  0   6   0 214   0   0   0   0   0   0   0]
 [  1   0   6   0  83   0   0   0   0   0   0]
 [  0   0   2   0   0  88   0   0   0   0   0]
 [  0   0   2   0   0   0 216  16   2   0   4]
 [  0   0  32   0   0   0   1 210   0   1   1]
 [  1   0   3   0   1   0   3   2 244  41  25]
 [  0   0   0   0   0   0   3   1   5 313  38]
 [  0   0   0   0   0   0   0   2   7  49 307]]
------ TEST ACCURACY:  weights.43-0.63.hdf5  ------
0.57
[[15  0  0  0  0  0  0]
 [ 0 20  0  0  0  0  0]
 [ 0  0 15  0  0  0  0]
 [ 0  4  0  0  6  0  0]
 [ 0  0  0  3  7  0  0]
 [ 0  3  0  0  0  0 12]
 [ 0 15  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/12/weights.44-0.60.hdf5
------ TRAIN ACCURACY:  weights.44-0.60.hdf5  ------
0.882164328657
[[238   0   1   1   0   0   0   0   0   0   0]
 [  1 136   0   3   0   0   0   0   0   0   0]
 [  0   0 245   0   0   0   0   0   0   0   0]
 [  0   0   1 219   0   0   0   0   0   0   0]
 [  0   0   7   0  78   0   0   0   0   0   0]
 [  0   0   0   0   0  80   0   0   0   0   0]
 [  0   0   4   0   0   0 225   3   3   2   3]
 [  1   0  75   0   0   0   4 161   1   2   1]
 [  1   0   5   0   0   0   1   0 243  25  30]
 [  0   0   0   0   0   0   9   0  13 275  48]
 [  0   0   0   0   0   0   1   3  11  34 301]]
------ TEST ACCURACY:  weights.44-0.60.hdf5  ------
0.61875
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 20  0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [13  0  0  0  2  0  0  0  0  0]
 [ 6  5  5  0  4  0  0  0  0  0]
 [ 0  0  0  0  0 15  0  0  0  0]
 [ 0 10  0  0  0  1  4  0  0  0]
 [ 0  0  0  0  0  0  0  0  6  9]
 [ 0  0  0  0  0  1  0  1 13  0]
 [ 0  0  0  0  0  1  0  1  0 13]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/13/weights.12-0.59.hdf5
------ TRAIN ACCURACY:  weights.12-0.59.hdf5  ------
0.766796875
[[245   0   0   0   0   0   0   0   0   0   0]
 [ 11   0   1 123   0   0   0   0   0   0   0]
 [  0   0 247   0   0   0   0  13   0   0   0]
 [  0   0   0 228   1   1   0   0   0   0   0]
 [  4   0  13   0  75   8   0   0   0   0   0]
 [  2   0   6   0   4  88   0   0   0   0   0]
 [  1   0  11   0   0   0 182  31   8   8   9]
 [  0   0  75   0   0   0  13 157   1   5   4]
 [  1   0   5   0   0   0   4   0 232  39  19]
 [  2   0   0   0   0   0   1   2  51 260  24]
 [  0   0   0   0   0   0   0   0  33  63 249]]
------ TEST ACCURACY:  weights.12-0.59.hdf5  ------
0.621052631579
[[ 7  0  3  0  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  4  0  0  1  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  0  0  1  0  4  0  0]
 [ 0  0  2  0  0  3  0  0  0]
 [ 0  0  0  0  0  0 15  4  1]
 [ 0  0  0  0  0  0  0  8 12]
 [ 0  0  0  0  0  0  4  0 16]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/14/weights.36-0.57.hdf5
------ TRAIN ACCURACY:  weights.36-0.57.hdf5  ------
0.830384615385
[[249   0   1   0   0   0   0   0   0   0   0]
 [ 11   0   0 129   0   0   0   0   0   0   0]
 [  0   0 250   0   0   0   0   5   0   0   0]
 [  0   0   0 234   1   0   0   0   0   0   0]
 [  1   0   7   0  92   0   0   0   0   0   0]
 [  0   0   5   0   0  95   0   0   0   0   0]
 [  0   0   0   0   0   0 228   5   9  11   2]
 [  0   0  43   0   0   0   3 199   1   3   1]
 [  1   0   4   0   0   0   1   1 242  50  16]
 [  0   0   0   0   0   0   5   0  18 294  28]
 [  0   0   0   0   0   0   4   0  18  57 276]]
------ TEST ACCURACY:  weights.36-0.57.hdf5  ------
0.527272727273
[[5 0 0 0 0 0 0]
 [0 1 0 9 0 0 0]
 [0 0 0 0 0 0 0]
 [0 3 1 6 0 0 0]
 [0 0 1 0 4 0 0]
 [0 0 3 1 1 6 4]
 [0 0 0 0 0 3 7]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/15/weights.42-0.52.hdf5
------ TRAIN ACCURACY:  weights.42-0.52.hdf5  ------
0.843529411765
[[205  30   0   0   0   0   8   2   0   0   0]
 [  1 132   0   2   0   0   0   0   0   0   0]
 [  0   0 191   0   0   0   3  66   0   0   0]
 [  0   2   0 223   0   0   0   0   0   0   0]
 [  1   0   5   0  94   0   0   0   0   0   0]
 [  0   0   5   3   1  91   0   0   0   0   0]
 [  0   0   0   0   0   0 223  10   1   6   5]
 [  1   0  13   0   0   0  16 213   0   3   4]
 [  2   0   1   0   0   0  10   4 215  21  47]
 [  0   0   0   0   0   0  11   1   5 272  56]
 [  0   0   0   0   0   0  10   0   6  37 292]]
------ TEST ACCURACY:  weights.42-0.52.hdf5  ------
0.4
[[10  0  0  0  0  0  0  0  0]
 [ 1  1  0  3  0  0  0  0  0]
 [ 0  0  0  0  0  5  0  0  0]
 [ 0  2  0  8  0  0  0  0  0]
 [ 0  0  0  0  8  0  0  2  0]
 [ 0  0  0  0  5  5  0  0  0]
 [ 0  0  0  0  4  0  6  5  5]
 [ 0  0  0  0  0  0  2  3 10]
 [ 0  0  0  0 19  0  0  0  1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/16/weights.19-0.56.hdf5
------ TRAIN ACCURACY:  weights.19-0.56.hdf5  ------
0.792214111922
[[214   0   1   0   0   0   0   0   0   0   0]
 [  6   0   0  93   0   0   0   1   0   0   0]
 [  0   0 161   0   0   0   0  59   0   0   0]
 [  0   0   0 193   2   0   0   0   0   0   0]
 [  0   0   1   0  63   0   0   1   0   0   0]
 [  0   0   4   0   3  62   0   1   0   0   0]
 [  2   0   0   0   0   0 151  34   4   4  15]
 [  0   0  17   0   0   0   1 193   0   2   2]
 [  0   0   3   0   0   0   4   2 160  58   8]
 [  0   0   0   0   0   0   2   0   8 243   7]
 [  0   0   0   0   0   0   5   0   7  70 188]]
------ TEST ACCURACY:  weights.19-0.56.hdf5  ------
0.628333333333
[[38  0  0  0  2  0  0  0  0  0  0]
 [ 0  0  0 40  0  0  0  0  0  0  0]
 [ 0  0 21  0  1  0  0 23  0  0  0]
 [ 0  0  0 40  0  0  0  0  0  0  0]
 [ 0  0  5  0 23  4  0  3  0  0  0]
 [ 0  0  0  1 18 11  0  0  0  0  0]
 [ 0  0  0  0  0  0 36  9  0  0  0]
 [ 1  0  9  0  0  0  2 33  0  0  0]
 [ 2  0  0  0  0  0  2  0 74  4  3]
 [ 2  0  0  1  0  0  2  1 11 68 15]
 [ 0  0  0  0  0  0 14  0 14 34 33]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_raw_spca_conv1d_10/17/weights.22-0.82.hdf5
------ TRAIN ACCURACY:  weights.22-0.82.hdf5  ------
0.831919191919
[[212  25   3   0   0   0   0   0   0   0   0]
 [  0 124   1   0   0   0   0   0   0   0   0]
 [  0   0 233   0   0   0   0  17   0   0   0]
 [  0  21   0 198   1   0   0   0   0   0   0]
 [  2   0  11   0  86   1   0   0   0   0   0]
 [  0   0   5   0   7  88   0   0   0   0   0]
 [  1   0   1   0   0   0 151  44  23  13   7]
 [  0   0  48   0   0   0   0 191   1   4   1]
 [  1   0   5   0   0   0   0   0 257  17  10]
 [  2   1   0   0   0   0   1   3  25 288  10]
 [  0   0   0   0   0   0   1   1  30  72 231]]
------ TEST ACCURACY:  weights.22-0.82.hdf5  ------
0.45
[[ 0 15  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0]
 [ 0  8  0  7  0  0  0  0  0]
 [ 0  0  1  0  1  9  1  2  1]
 [ 0  0  3  0  0 11  0  0  1]
 [ 0  0  0  0  0  0 10 20  0]
 [ 0  0  0  0  4  0 10 12  4]
 [ 0  0  0  0  0  0  1 19 10]]
[0.45263157894736844, 0.44666666666666666, 0.572972972972973, 0.5684210526315789, 0.6761904761904762, 0.7, 0.8090909090909091, 0.8, 0.5777777777777777, 0.6, 0.57, 0.61875, 0.6210526315789474, 0.5272727272727272, 0.4, 0.6283333333333333, 0.45]
0.58936236038
0.112547291288

Process finished with exit code 0

'''