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

X = pickle.load(open("data/X_umafall_kpca.p", "rb"))
y = pickle.load(open("data/y_umafall_kpca.p", "rb"))

n_classes = 11
signal_rows = 450
signal_columns = 1
n_subject = 17

'''
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

    new_dir = 'model/umafall_kpca_conv1d_10/' + str(i+1) + '/'
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
'''

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
    path_str = 'model/umafall_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.5421052631578948, 0.4866666666666667, 0.4810810810810811, 0.4842105263157895, 0.45714285714285713, 0.38, 0.7, 0.47368421052631576, 0.55, 0.5047619047619047, 0.58, 0.53125, 0.4842105263157895, 0.6363636363636364, 0.5238095238095238, 0.5, 0.45]
0.515605070361
0.0716025755826
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/1/weights.37-0.46.hdf5
------ TRAIN ACCURACY:  weights.37-0.46.hdf5  ------
0.810547667343
[[235   0   0   2   0   0   2   0   0   1   0]
 [  0 110   0  14   0   0   1   0   0   0   0]
 [  1   0 196   0   1   0   4  24  15   3   6]
 [  0   0   0 218   0   0   0   0   0   2   0]
 [  1   0   0   0  95   0   0   0   2   0   2]
 [  0   0   0   2   0  93   0   0   0   1   4]
 [  1   0  12   0   1   0 171  50   4   1   0]
 [  1   0  27   0   0   0  31 174  10   1   1]
 [  2   0   7   3   3   0  10  10 190  29  36]
 [  4   0   2   0   0   0  15  10  14 248  27]
 [  1   1   8   4   0   4   1   0  21  27 268]]
------ TEST ACCURACY:  weights.37-0.46.hdf5  ------
0.542105263158
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  7  0  8  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  5  0  0  2]
 [ 0  0  0 13  0  2  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  0 12  0  1  0  0]
 [ 0  0  0  0  0  0 10  5  0  0  0]
 [ 1  0  2  0  4  0  2  0 11  5  5]
 [ 2  0  3  0  1  1  1  2  5 19  6]
 [ 3  0  1  0  0  1  0  0  1 11 13]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/2/weights.22-0.49.hdf5
------ TRAIN ACCURACY:  weights.22-0.49.hdf5  ------
0.753293413174
[[234   0   0   2   0   0   1   1   0   2   0]
 [  0 116   0   8   0   0   0   0   0   0   1]
 [  0   0 185   0   2   0   2  33  21   3   4]
 [  1   4   0 212   0   1   0   0   0   0   2]
 [  4   2   0   0  82   3   0   0   3   1   5]
 [  0   0   0   0   0  94   0   0   3   1   2]
 [  0   0  18   0   0   0 135  73  10   4   0]
 [  0   0  24   0   0   0  19 190  10   2   0]
 [  1   2   8   2   2   1   9  11 210  19  40]
 [  0   1   7   0   1   1  12  19  40 190  59]
 [  2   1  12   2   2   7   5   3  48  29 239]]
------ TEST ACCURACY:  weights.22-0.49.hdf5  ------
0.486666666667
[[14  1  0  0  0  0  0  0  0  0]
 [11  4  0  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  3  1  0  0]
 [ 1  7  0  6  0  0  0  0  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  4 11  0  0  0]
 [ 0  0  7  0  0  0  7  1  0  0]
 [ 0  0  0  0  0  0  0  5  1  9]
 [ 0  2  0  0  0  1  0  5 11 11]
 [ 1  0  0  0  1  0  0  1  1 11]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/3/weights.41-0.49.hdf5
------ TRAIN ACCURACY:  weights.41-0.49.hdf5  ------
0.815789473684
[[234   0   0   2   0   0   2   0   0   2   0]
 [  0 123   0   2   0   0   0   0   0   0   0]
 [  0   0 215   0   1   0   9   6   7   4   3]
 [  0   4   0 213   0   1   0   0   0   0   2]
 [  0   0   0   0  95   0   0   0   1   2   2]
 [  0   0   0   0   0  95   0   0   0   1   4]
 [  0   0  17   0   0   0 175  41   4   2   1]
 [  1   0  45   0   0   0  37 150   9   1   2]
 [  0   0  15   1   3   0   7   4 176  17  67]
 [  0   0   7   0   1   0  12   4  17 237  52]
 [  1   0   8   1   2   1   0   0   9  11 302]]
------ TEST ACCURACY:  weights.41-0.49.hdf5  ------
0.481081081081
[[14  1  0  0  0  0  0  0  0  0]
 [ 3  9  0  0  0  1  0  0  2  0]
 [ 0  0 13  0  0  0  6  0  0  1]
 [ 0  6  0  7  1  0  0  0  0  1]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 14  0  0  0  1]
 [ 0  0  0  0  0  6  9  0  0  0]
 [ 2  0  2  0  0  0  0  2  5 19]
 [ 1  0  1  0  0  0  0  5  4 19]
 [ 0  0  0  0  0  0  0  1 12 17]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/4/weights.18-0.44.hdf5
------ TRAIN ACCURACY:  weights.18-0.44.hdf5  ------
0.742799188641
[[227   0   0   3   0   0   2   0   1   2   0]
 [  1 104   0  18   0   0   0   0   0   2   0]
 [  0   0 185   0   1   0   1  25  20   8  10]
 [  1   0   0 216   0   1   0   0   0   1   1]
 [  0   0   0   2  90   0   0   0   4   2   2]
 [  1   0   0   5   4  78   0   0   3   2   7]
 [  0   0  19   0   2   0 130  66   8   4   1]
 [  1   0  43   0   1   0  14 173   9   2   2]
 [  3   1   9   0   5   0   5  14 158  31  64]
 [  1   0   8   0   1   0  12  12  24 217  60]
 [  2   0   8   5   0   2   1   3  27  34 253]]
------ TEST ACCURACY:  weights.18-0.44.hdf5  ------
0.484210526316
[[17  0  0  3  0  0  0  0  0  0  0]
 [ 2  5  0  8  0  0  0  0  0  0  0]
 [ 0  0 11  0  0  0  0  1  2  0  1]
 [13  0  0  2  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0 13 10  1  0  0]
 [ 0  0  2  0  0  0  5  8  0  0  0]
 [ 2  0  2  1  2  1  0  1  7  4 10]
 [ 1  0  0  0  0  0  0  0  4 10 10]
 [ 1  1  0  1  0  0  0  0  1  7 19]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/5/weights.11-0.53.hdf5
------ TRAIN ACCURACY:  weights.11-0.53.hdf5  ------
0.659215686275
[[232   0   0   2   0   0   2   0   1   2   1]
 [ 15 102   0  19   0   0   0   0   3   1   0]
 [  2   0 165   0   1   1  15  14  30   5  17]
 [ 18   1   0 195   0   0   0   0   1   0   5]
 [  8   0   0   3  72   1   0   0   9   4   3]
 [  6   0   0   6   2  64   0   0   7   0  15]
 [  0   0  21   0   0   0 148  54  16   1   0]
 [  0   0  45   0   0   0  49 125  21   3   2]
 [ 11   0  11   1   3   1   5   8 204   8  68]
 [ 11   1  17   0   3   0  13   5 102 103 105]
 [  6   0   5   4   2   3   1   0  38   5 271]]
------ TEST ACCURACY:  weights.11-0.53.hdf5  ------
0.457142857143
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  1  0  1  1]
 [ 0  9  0  6  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  3  0  0  0  6  0  5  1]
 [ 0  0  2  0  0  0  5  8  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  1  1  2  0 12  1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/6/weights.38-0.46.hdf5
------ TRAIN ACCURACY:  weights.38-0.46.hdf5  ------
0.812284069098
[[246   0   0   2   0   0   2   0   0   0   0]
 [  1 129   0  10   0   0   0   0   0   0   0]
 [  1   0 213   0   2   0   8  14  15   5   2]
 [  0   0   0 234   0   0   0   0   0   0   1]
 [  0   0   0   0  94   0   0   0   4   1   1]
 [  3   0   0   0   1  90   0   0   1   1   4]
 [  0   0  11   0   1   0 220  16   5   1   1]
 [  0   0  43   0   0   0  61 135   8   1   2]
 [  4   0  13   1   3   0   9   2 204  41  33]
 [  2   0   3   0   0   0  20   2  19 283  21]
 [  4   0   6   1   2   2   1   0  29  42 268]]
------ TEST ACCURACY:  weights.38-0.46.hdf5  ------
0.38
[[5 0 0 0 0 0 0 0]
 [0 4 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [3 0 0 4 2 1 0 0]
 [3 0 0 0 0 2 1 4]
 [0 0 1 0 0 4 2 3]
 [0 0 0 0 0 6 0 4]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/7/weights.40-0.49.hdf5
------ TRAIN ACCURACY:  weights.40-0.49.hdf5  ------
0.803536345776
[[237   0   0   0   0   0   2   0   0   1   0]
 [  0 128   0  11   0   0   1   0   0   0   0]
 [  1   0 165   0   2   0   5  44  20   6   7]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  80   0   0   0   3   1   1]
 [  1   0   0   0   0  79   0   0   1   0   4]
 [  0   0   2   0   1   0 167  60   4   0   1]
 [  1   0  14   0   0   0  32 184  10   1   3]
 [  2   1   2   2   4   0   9   8 228  14  50]
 [  3   0   1   0   0   0  16  12  34 253  41]
 [  4   0   2   1   5   1   2   4  33   9 304]]
------ TEST ACCURACY:  weights.40-0.49.hdf5  ------
0.7
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 10  0  0  0  0  2  2  1  0]
 [ 0  0 11  0  0  0  0  0  3  1]
 [ 4  0  0 11  0  0  0  0  0  0]
 [ 2  0  0  2  5  0  0  2  0  4]
 [ 0  4  0  0  0 13  2  1  0  0]
 [ 0  0  0  0  0  3 12  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/8/weights.32-0.48.hdf5
------ TRAIN ACCURACY:  weights.32-0.48.hdf5  ------
0.78671875
[[237   0   0   0   0   0   1   1   0   1   0]
 [  0 134   0   5   0   0   1   0   0   0   0]
 [  0   0 195   0   2   0   3  17  23   5   5]
 [  4   1   0 211   0   1   0   0   0   2   1]
 [  0   0   0   0  87   0   0   0   2   0   1]
 [  4   0   0   0   0  83   0   0   1   0   2]
 [  0   0  16   0   1   0 167  46   5   3   2]
 [  1   0  31   0   0   0  31 169  12   0   1]
 [  3   0   5   2   3   0  12   6 211  12  66]
 [  6   0   2   0   1   0  16   8  39 218  70]
 [  3   0   7   2   3   2   1   3  35   7 302]]
------ TEST ACCURACY:  weights.32-0.48.hdf5  ------
0.473684210526
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  9  0  0  0  0  2  2  2  0]
 [ 3  6  0  4  0  0  0  0  0  0  2]
 [ 0  0  0  0  5  1  0  0  1  0  3]
 [ 0  0  0  0  0  3  0  0  2  1  4]
 [ 0  0  1  0  0  0  4 10  0  0  0]
 [ 0  0  8  0  1  0  0  5  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/9/weights.34-0.51.hdf5
------ TRAIN ACCURACY:  weights.34-0.51.hdf5  ------
0.79797979798
[[237   0   0   1   0   0   2   0   0   0   0]
 [  0 115   0   9   0   0   1   0   0   0   0]
 [  1   0 206   0   0   0   6  19  13   1   4]
 [  4   0   0 214   0   0   0   0   0   0   2]
 [  2   1   0   0  86   3   0   0   4   1   3]
 [  1   0   0   0   0  94   0   0   2   0   3]
 [  1   0   7   0   1   0 187  40   4   0   0]
 [  1   0  35   0   0   0  36 162   9   1   1]
 [  4   1   8   2   1   0  14   6 198  10  46]
 [  2   0   7   0   0   0  17   6  38 197  63]
 [  4   0  12   2   0   4   1   2  23   8 279]]
------ TEST ACCURACY:  weights.34-0.51.hdf5  ------
0.55
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  1  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  0  1  0  0]
 [ 0 11  0  4  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  1  0 11  2  0  0  0]
 [ 0  0  0  0  0  0  8  7  0  0  0]
 [ 0  0  3  0  3  2  2  0  9  1 10]
 [ 2  1  0  0  0  0  4  0  0  6 17]
 [ 1  0  0  0  1  3  0  0  3  3 19]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/10/weights.46-0.50.hdf5
------ TRAIN ACCURACY:  weights.46-0.50.hdf5  ------
0.847058823529
[[235   0   0   2   0   0   2   0   0   1   0]
 [  0 135   0   5   0   0   0   0   0   0   0]
 [  1   0 205   0   0   0   8  14  16   5   1]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  80   0   0   0   3   1   1]
 [  0   0   0   0   0  82   0   0   1   0   2]
 [  1   0   7   0   0   0 194  33   4   1   0]
 [  0   0  23   0   0   0  36 178   6   2   0]
 [  0   0   4   2   1   0  11   1 260  28  13]
 [  0   0   1   0   1   0  16   5  29 300   8]
 [  0   0   4   3   0   1   0   0  49  36 272]]
------ TEST ACCURACY:  weights.46-0.50.hdf5  ------
0.504761904762
[[ 8  0  6  0  1  0  0  0  0  0]
 [ 0  0  0  2  0  1  0  8  1  3]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0 11  3  0  0  1  0  0]
 [ 0  0  1  3 10  0  0  0  0  1]
 [ 0  0  0  0  0  8  7  0  0  0]
 [ 0  4  0  0  0  1  1  8  1  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/11/weights.33-0.50.hdf5
------ TRAIN ACCURACY:  weights.33-0.50.hdf5  ------
0.786301369863
[[236   0   0   2   0   0   2   0   0   0   0]
 [  2 130   0   6   0   0   1   0   0   0   1]
 [  1   0 200   0   1   0   7  20  11   1   4]
 [  3   0   0 215   0   0   0   0   0   0   2]
 [  2   0   0   0  80   1   0   0   4   0   3]
 [  1   0   0   0   0  85   0   0   0   0   4]
 [  1   0  19   0   0   0 186  27   5   1   1]
 [  1   0  37   0   0   0  45 151   9   1   1]
 [  2   0  16   0   4   0   8   6 217  16  51]
 [  3   0  17   0   0   0  11   4  40 221  64]
 [  4   0  15   1   2   2   1   1  39  12 288]]
------ TEST ACCURACY:  weights.33-0.50.hdf5  ------
0.58
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  1  1  1  1  1]
 [ 1  1  0 13  0  0  0  0  0  0  0]
 [ 0  0  0  0 10  0  0  0  0  0  0]
 [ 2  0  0  0  4  4  0  0  0  0  0]
 [ 0  0  5  0  0  0  1  9  0  0  0]
 [ 0  0 12  0  0  0  0  0  3  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/12/weights.48-0.52.hdf5
------ TRAIN ACCURACY:  weights.48-0.52.hdf5  ------
0.820841683367
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 138   0   2   0   0   0   0   0   0   0]
 [  1   0 186   0   2   0  12  17  11   7   9]
 [  4   3   0 211   0   0   0   0   0   1   1]
 [  0   0   0   0  81   0   0   0   2   1   1]
 [  0   0   0   0   0  77   0   0   0   0   3]
 [  0   0   3   0   1   0 204  27   3   1   1]
 [  1   0  25   0   0   0  47 157  13   1   1]
 [  1   1   5   0   3   1   9   3 190  23  69]
 [  0   0   3   0   1   1  14   6  13 249  58]
 [  4   0   2   0   3   2   0   0  13   9 317]]
------ TEST ACCURACY:  weights.48-0.52.hdf5  ------
0.53125
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 14  0  0  0  0  0  1  2  3]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [13  0  0  0  0  2  0  0  0  0  0]
 [11  1  0  3  0  2  0  0  1  1  1]
 [ 0  0  0  0  0  0 13  2  0  0  0]
 [ 0  0  1  0  0  0  5  9  0  0  0]
 [ 0  1  0  0  0  0  0  0  4  2  8]
 [ 0  0  0  0  0  0  0  0  3  5  7]
 [ 0  1  0  0  1  0  0  0  3  2  8]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/13/weights.47-0.51.hdf5
------ TRAIN ACCURACY:  weights.47-0.51.hdf5  ------
0.82265625
[[245   0   0   0   0   0   0   0   0   0   0]
 [  0 135   0   0   0   0   0   0   0   0   0]
 [  2   0 226   0   0   0   5  11  11   2   3]
 [  3  22   0 204   0   0   0   0   0   0   1]
 [  0   0   0   0  94   0   0   0   4   1   1]
 [  2   0   0   0   0  95   0   0   3   0   0]
 [  1   0  17   0   0   0 194  34   3   1   0]
 [  0   0  46   0   0   0  25 172  10   1   1]
 [  3   3  11   0   3   1   8   5 229  19  18]
 [  2   0   5   0   1   0  16   6  38 252  20]
 [  7   0   9   0   5   2   0   2  42  18 260]]
------ TEST ACCURACY:  weights.47-0.51.hdf5  ------
0.484210526316
[[ 5  0  0  0  3  0  0  2  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  1  0  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  1  0  2  0  2  0  0]
 [ 0  0  0  0  2  3  0  0  0]
 [ 0  2  1  0  0  0  9  4  4]
 [ 0  4  0  0  0  0  2 11  3]
 [ 0  5  0  0  0  0  7  1  7]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/14/weights.35-0.49.hdf5
------ TRAIN ACCURACY:  weights.35-0.49.hdf5  ------
0.793076923077
[[247   0   0   0   0   0   1   1   0   1   0]
 [  0 135   0   4   0   0   1   0   0   0   0]
 [  1   0 206   0   3   0   4  12  16   9   4]
 [  3   6   0 224   0   0   0   0   0   1   1]
 [  2   0   0   0  94   0   0   0   3   1   0]
 [  5   0   0   0   1  89   0   0   1   0   4]
 [  1   0  16   0   3   0 191  30   5   6   3]
 [  1   0  37   0   1   0  52 140  11   6   2]
 [  4   1  10   1  12   1   6   2 207  17  54]
 [  5   3   3   0   6   0  11   5  22 246  44]
 [  6   2   5   2   7   2   0   1  28  19 283]]
------ TEST ACCURACY:  weights.35-0.49.hdf5  ------
0.636363636364
[[5 0 0 0 0 0 0]
 [0 6 0 4 0 0 0]
 [0 0 0 0 0 0 0]
 [0 2 0 7 1 0 0]
 [0 1 1 0 1 2 0]
 [0 0 1 1 0 9 4]
 [0 0 0 0 2 1 7]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/15/weights.31-0.45.hdf5
------ TRAIN ACCURACY:  weights.31-0.45.hdf5  ------
0.790588235294
[[243   0   0   0   0   0   1   1   0   0   0]
 [  0 128   0   6   0   0   1   0   0   0   0]
 [  1   0 215   0   1   0   2  20   9   4   8]
 [  3   2   0 215   0   2   0   0   0   1   2]
 [  0   0   0   0  95   1   0   0   2   1   1]
 [  2   0   0   0   0  95   0   0   0   0   3]
 [  0   0  15   0   2   0 174  47   3   2   2]
 [  1   0  42   0   0   0  26 169   9   1   2]
 [  1   1  18   0   6   0   9   6 174  17  68]
 [  1   1   9   0   2   1  15   9  12 213  82]
 [  3   0  14   1   2   2   3   3  16   6 295]]
------ TEST ACCURACY:  weights.31-0.45.hdf5  ------
0.52380952381
[[ 9  0  0  0  1  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  4  1  0  0  0]
 [ 0  2  0  8  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  6  3  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  0  3  5  5  7]
 [ 2  0  0  0  0  0  0  2  6  5]
 [ 0  1  0  0  0  0  0  3  5 11]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/16/weights.26-0.48.hdf5
------ TRAIN ACCURACY:  weights.26-0.48.hdf5  ------
0.76496350365
[[210   0   0   2   0   0   1   1   0   1   0]
 [  0  93   0   7   0   0   0   0   0   0   0]
 [  1   0 150   0   3   0   3  35  14   3  11]
 [  0   2   0 191   0   0   0   0   0   1   1]
 [  1   0   0   0  61   0   0   0   2   0   1]
 [  2   0   0   4   0  58   0   0   2   0   4]
 [  0   0   6   0   0   0 106  90   5   3   0]
 [  1   0  22   0   0   0  10 169  11   1   1]
 [  3   1   4   0   1   0   5  13 160  11  37]
 [  2   3   4   0   1   0   7  20  31 164  28]
 [  2   2   9   4   1   1   0   8  23  10 210]]
------ TEST ACCURACY:  weights.26-0.48.hdf5  ------
0.5
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 7 33  0  0  0  0  0  0  0  0  0]
 [ 0  0 34  0  0  0  1  5  3  2  0]
 [ 5  0  0 35  0  0  0  0  0  0  0]
 [ 0  3  2  0 20  0  1  0  4  0  5]
 [ 0  0  0  1  2  4  0  0 10  4  9]
 [ 0  0  0  0  0  0 15 29  1  0  0]
 [ 0  0  6  0  0  0  3 35  0  1  0]
 [ 0  2  2  1  1  0  5 10 35 16 13]
 [ 1  4  4  0  0  0  4 15 21 21 30]
 [ 0  6  3  2  0  0  1  4 24 27 28]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_kpca_conv1d_10/17/weights.47-0.38.hdf5
------ TRAIN ACCURACY:  weights.47-0.38.hdf5  ------
0.816161616162
[[238   0   0   0   0   0   2   0   0   0   0]
 [  1 123   0   0   0   0   1   0   0   0   0]
 [  0   0 173   0   2   0  35  25   9   4   2]
 [ 19   0   0 200   0   0   0   0   0   0   1]
 [  0   0   0   0  96   0   0   0   2   1   1]
 [  3   0   0   0   0  95   0   0   2   0   0]
 [  0   0   4   0   1   0 212  19   3   1   0]
 [  0   0  14   0   0   0  56 163  10   2   0]
 [  2   0   2   1   3   0  13   2 200  29  38]
 [  2   1   2   0   0   0  11   3  20 262  29]
 [  3   0   1   1   5   2   3   1  18  43 258]]
------ TEST ACCURACY:  weights.47-0.38.hdf5  ------
0.45
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  7  0  8  0  0  0  0  0  0  0]
 [ 2  0  2  0  0  0  2  2  3  3  1]
 [ 0  0  0 14  0  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 1  0  0  0  0  0 13  1  0  0  0]
 [ 0  0  1  0  0  0 10  4  0  0  0]
 [ 1  0  2  0  3  0  3  1 10  5  5]
 [ 0  0  0  0  1  0 11  5  1  6  6]
 [ 1  0  0  1  3  1  0  0  9  5 10]]
[0.5421052631578948, 0.4866666666666667, 0.4810810810810811, 0.4842105263157895, 0.45714285714285713, 0.38, 0.7, 0.47368421052631576, 0.55, 0.5047619047619047, 0.58, 0.53125, 0.4842105263157895, 0.6363636363636364, 0.5238095238095238, 0.5, 0.45]
0.515605070361
0.0716025755826

Process finished with exit code 0

'''