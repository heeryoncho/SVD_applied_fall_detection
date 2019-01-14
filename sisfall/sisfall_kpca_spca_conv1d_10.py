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

X_spca = pickle.load(open("data/X_sisfall_spca.p", "rb"))
X_kpca = pickle.load(open("data/X_sisfall_kpca.p", "rb"))
X = np.concatenate((X_spca, X_kpca), axis=1)

y = pickle.load(open("data/y_sisfall_spca.p", "rb"))

n_classes = 34
signal_rows = 900
signal_columns = 1
n_subject = 23


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
    print X_train.shape #

    # input layer
    input_signal = Input(shape=(signal_rows, 1))
    print K.int_shape(input_signal)

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

    new_dir = 'model/sisfall_kpca_spca_conv1d_10/' + str(i+1) + '/'
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
    test = y.loc[y[1] == i + 1]
    test_index = test.index.values

    train = y[~y.index.isin(test_index)]
    train_index = train.index.values

    y_values = y.ix[:, 0].values

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_values[train_index] - 1, y_values[test_index] - 1

    print "\n>>>>>>>>>>>>>>", str(i + 1), "-fold <<<<<<<<<<<<<<<<"
    path_str = 'model/sisfall_kpca_spca_conv1d_10/' + str(i + 1) + '/'
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
[0.48823529411764705, 0.4470588235294118, 0.37058823529411766, 0.48823529411764705, 0.5117647058823529, 0.5294117647058824, 0.5176470588235295, 0.4823529411764706, 0.5470588235294118, 0.4764705882352941, 0.4294117647058823, 0.5176470588235295, 0.4764705882352941, 0.4294117647058823, 0.5818181818181818, 0.5294117647058824, 0.47337278106508873, 0.5235294117647059, 0.5117647058823529, 0.5240963855421686, 0.5411764705882353, 0.3764705882352941, 0.5470588235294118]
0.492194079088
0.0520705879358
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_kpca_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/1/weights.49-0.50.hdf5
------ TRAIN ACCURACY:  weights.49-0.50.hdf5  ------
0.919571045576
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  94   1   4]
 [  0   0   0 ...   5  85   9]
 [  0   0   0 ...   3   1  98]]
------ TEST ACCURACY:  weights.49-0.50.hdf5  ------
0.488235294118
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/2/weights.47-0.49.hdf5
------ TRAIN ACCURACY:  weights.47-0.49.hdf5  ------
0.916353887399
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  90   1   3]
 [  0   0   0 ...   6  94   1]
 [  0   0   0 ...   5   2  90]]
------ TEST ACCURACY:  weights.47-0.49.hdf5  ------
0.447058823529
[[2 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/3/weights.36-0.50.hdf5
------ TRAIN ACCURACY:  weights.36-0.50.hdf5  ------
0.9
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  88   1   3]
 [  0   0   0 ...   9  88   1]
 [  0   0   0 ...  11   2  85]]
------ TEST ACCURACY:  weights.36-0.50.hdf5  ------
0.370588235294
[[0 5 0 ... 0 0 0]
 [0 3 1 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/4/weights.45-0.50.hdf5
------ TRAIN ACCURACY:  weights.45-0.50.hdf5  ------
0.916353887399
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  90   1   4]
 [  0   0   0 ...   6  91   4]
 [  0   0   0 ...   3   2  96]]
------ TEST ACCURACY:  weights.45-0.50.hdf5  ------
0.488235294118
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/5/weights.22-0.49.hdf5
------ TRAIN ACCURACY:  weights.22-0.49.hdf5  ------
0.830563002681
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  86 ...   0   0   0]
 ...
 [  0   0   0 ...  73   1   4]
 [  0   0   0 ...   7  77   3]
 [  0   0   0 ...  11   3  53]]
------ TEST ACCURACY:  weights.22-0.49.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/6/weights.35-0.49.hdf5
------ TRAIN ACCURACY:  weights.35-0.49.hdf5  ------
0.899731903485
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  86   1   5]
 [  0   0   0 ...   7  83   3]
 [  0   0   0 ...   6   3  89]]
------ TEST ACCURACY:  weights.35-0.49.hdf5  ------
0.529411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 2 0 2]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/7/weights.43-0.50.hdf5
------ TRAIN ACCURACY:  weights.43-0.50.hdf5  ------
0.905898123324
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  97   3   1]
 [  0   0   0 ...   8  94   1]
 [  0   0   0 ...  13   9  78]]
------ TEST ACCURACY:  weights.43-0.50.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 3 1 0]
 [0 0 0 ... 1 3 0]
 [0 0 0 ... 1 1 1]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/8/weights.46-0.50.hdf5
------ TRAIN ACCURACY:  weights.46-0.50.hdf5  ------
0.914745308311
[[104   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  93   0   6]
 [  0   0   0 ...   7  88  10]
 [  0   0   0 ...   2   1 101]]
------ TEST ACCURACY:  weights.46-0.50.hdf5  ------
0.482352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/9/weights.32-0.49.hdf5
------ TRAIN ACCURACY:  weights.32-0.49.hdf5  ------
0.866219839142
[[107   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  99 ...   0   0   0]
 ...
 [  0   0   0 ...  78   0   1]
 [  0   0   0 ...   9  71   8]
 [  0   0   0 ...   9   0  73]]
------ TEST ACCURACY:  weights.32-0.49.hdf5  ------
0.547058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/10/weights.44-0.49.hdf5
------ TRAIN ACCURACY:  weights.44-0.49.hdf5  ------
0.904557640751
[[106   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  92   1   7]
 [  0   0   0 ...   6  84  14]
 [  0   0   0 ...   5   1  98]]
------ TEST ACCURACY:  weights.44-0.49.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/11/weights.48-0.50.hdf5
------ TRAIN ACCURACY:  weights.48-0.50.hdf5  ------
0.91528150134
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  88   3   1]
 [  0   0   0 ...   2  98   1]
 [  0   0   0 ...   6   4  88]]
------ TEST ACCURACY:  weights.48-0.50.hdf5  ------
0.429411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 1 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 3 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/12/weights.26-0.49.hdf5
------ TRAIN ACCURACY:  weights.26-0.49.hdf5  ------
0.830026809651
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  92 ...   0   0   0]
 ...
 [  0   0   0 ...  58   3  22]
 [  0   0   0 ...   6  82  15]
 [  0   0   0 ...   2   3  98]]
------ TEST ACCURACY:  weights.26-0.49.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 2 2]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/13/weights.41-0.49.hdf5
------ TRAIN ACCURACY:  weights.41-0.49.hdf5  ------
0.895710455764
[[106   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  73   0   2]
 [  0   0   0 ...   3  86   3]
 [  0   0   0 ...   5   1  77]]
------ TEST ACCURACY:  weights.41-0.49.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/14/weights.49-0.49.hdf5
------ TRAIN ACCURACY:  weights.49-0.49.hdf5  ------
0.909383378016
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  95   1   2]
 [  0   0   0 ...   9  87   5]
 [  0   0   0 ...  12   1  77]]
------ TEST ACCURACY:  weights.49-0.49.hdf5  ------
0.429411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/15/weights.46-0.50.hdf5
------ TRAIN ACCURACY:  weights.46-0.50.hdf5  ------
0.9046854083
[[104   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  82   6   3]
 [  0   0   0 ...   4  96   0]
 [  0   0   0 ...   5   8  84]]
------ TEST ACCURACY:  weights.46-0.50.hdf5  ------
0.581818181818
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 2 1 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/16/weights.42-0.48.hdf5
------ TRAIN ACCURACY:  weights.42-0.48.hdf5  ------
0.890616621984
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  97 ...   0   0   0]
 ...
 [  0   0   0 ...  72   3   2]
 [  0   0   0 ...   3  93   0]
 [  0   0   0 ...   3   1  68]]
------ TEST ACCURACY:  weights.42-0.48.hdf5  ------
0.529411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/17/weights.39-0.48.hdf5
------ TRAIN ACCURACY:  weights.39-0.48.hdf5  ------
0.897078531225
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  87   0  10]
 [  0   0   0 ...   5  88   6]
 [  0   0   0 ...   5   1  99]]
------ TEST ACCURACY:  weights.39-0.48.hdf5  ------
0.473372781065
[[0 0 0 ... 0 0 0]
 [3 2 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 2 1 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/18/weights.46-0.49.hdf5
------ TRAIN ACCURACY:  weights.46-0.49.hdf5  ------
0.895978552279
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  71   0   1]
 [  0   0   0 ...   6  83   0]
 [  0   0   0 ...   4   2  79]]
------ TEST ACCURACY:  weights.46-0.49.hdf5  ------
0.523529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/19/weights.43-0.49.hdf5
------ TRAIN ACCURACY:  weights.43-0.49.hdf5  ------
0.909651474531
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  73   4   6]
 [  0   0   0 ...   3  94   1]
 [  0   0   0 ...   1   2  97]]
------ TEST ACCURACY:  weights.43-0.49.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 3 1 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/20/weights.48-0.49.hdf5
------ TRAIN ACCURACY:  weights.48-0.49.hdf5  ------
0.926352437065
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  87   1   2]
 [  0   0   0 ...   4  90   0]
 [  0   0   0 ...   4   1  91]]
------ TEST ACCURACY:  weights.48-0.49.hdf5  ------
0.524096385542
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 2]
 [0 0 0 ... 1 2 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/21/weights.39-0.51.hdf5
------ TRAIN ACCURACY:  weights.39-0.51.hdf5  ------
0.902412868633
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  84   2   6]
 [  0   0   0 ...   3  94   5]
 [  0   0   0 ...   3   4  93]]
------ TEST ACCURACY:  weights.39-0.51.hdf5  ------
0.541176470588
[[1 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 1 3 0]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/22/weights.40-0.55.hdf5
------ TRAIN ACCURACY:  weights.40-0.55.hdf5  ------
0.90563002681
[[105   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  64   8  10]
 [  0   0   0 ...   2  98   2]
 [  0   0   0 ...   3   8  86]]
------ TEST ACCURACY:  weights.40-0.55.hdf5  ------
0.376470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_spca_conv1d_10/23/weights.43-0.49.hdf5
------ TRAIN ACCURACY:  weights.43-0.49.hdf5  ------
0.911528150134
[[106   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  84   2   5]
 [  0   0   0 ...   3  99   0]
 [  0   0   0 ...   5   7  87]]
------ TEST ACCURACY:  weights.43-0.49.hdf5  ------
0.547058823529
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 2]]
[0.48823529411764705, 0.4470588235294118, 0.37058823529411766, 0.48823529411764705, 0.5117647058823529, 0.5294117647058824, 0.5176470588235295, 0.4823529411764706, 0.5470588235294118, 0.4764705882352941, 0.4294117647058823, 0.5176470588235295, 0.4764705882352941, 0.4294117647058823, 0.5818181818181818, 0.5294117647058824, 0.47337278106508873, 0.5235294117647059, 0.5117647058823529, 0.5240963855421686, 0.5411764705882353, 0.3764705882352941, 0.5470588235294118]
0.492194079088
0.0520705879358

Process finished with exit code 0

'''
