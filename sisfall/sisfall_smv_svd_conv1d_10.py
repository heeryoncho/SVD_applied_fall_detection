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

X_svd = pickle.load(open("data/X_sisfall_svd.p", "rb"))
X_smv = pickle.load(open("data/X_sisfall_smv.p", "rb"))
X = np.concatenate((X_svd, X_smv), axis=1)

y = pickle.load(open("data/y_sisfall_svd.p", "rb"))

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

    new_dir = 'model/sisfall_smv_svd_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_smv_svd_conv1d_10/' + str(i+1) + '/'
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
[0.6058823529411764, 0.4176470588235294, 0.5117647058823529, 0.5764705882352941, 0.5705882352941176, 0.6, 0.6235294117647059, 0.5647058823529412, 0.5176470588235295, 0.5176470588235295, 0.5117647058823529, 0.5294117647058824, 0.5411764705882353, 0.5647058823529412, 0.5636363636363636, 0.6, 0.5680473372781065, 0.5823529411764706, 0.6058823529411764, 0.5602409638554217, 0.5823529411764706, 0.4470588235294118, 0.6058823529411764]
0.555147619696
0.0498109050091
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_smv_svd_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/1/weights.50-0.51.hdf5
------ TRAIN ACCURACY:  weights.50-0.51.hdf5  ------
0.912868632708
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   0]
 [  0   0   0 ...   4  95   0]
 [  0   0   0 ...   9   3  62]]
------ TEST ACCURACY:  weights.50-0.51.hdf5  ------
0.605882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/2/weights.35-0.53.hdf5
------ TRAIN ACCURACY:  weights.35-0.53.hdf5  ------
0.917426273458
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   2]
 [  0   0   0 ...   8  90   8]
 [  0   0   0 ...   6   1  96]]
------ TEST ACCURACY:  weights.35-0.53.hdf5  ------
0.417647058824
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 1]
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 3 0 1]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/3/weights.39-0.55.hdf5
------ TRAIN ACCURACY:  weights.39-0.55.hdf5  ------
0.922520107239
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  98   1   2]
 [  0   0   0 ...   5  97   2]
 [  0   0   0 ...   5   1  98]]
------ TEST ACCURACY:  weights.39-0.55.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 4 1 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 3 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/4/weights.36-0.55.hdf5
------ TRAIN ACCURACY:  weights.36-0.55.hdf5  ------
0.906166219839
[[105   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  91   0   7]
 [  0   0   0 ...   1  94   5]
 [  0   0   0 ...   4   0 100]]
------ TEST ACCURACY:  weights.36-0.55.hdf5  ------
0.576470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 2 3]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/5/weights.39-0.54.hdf5
------ TRAIN ACCURACY:  weights.39-0.54.hdf5  ------
0.922252010724
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  96   0   3]
 [  0   0   0 ...   2  97   4]
 [  0   0   0 ...   5   2  95]]
------ TEST ACCURACY:  weights.39-0.54.hdf5  ------
0.570588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 3 2]
 [0 0 0 ... 3 0 1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/6/weights.29-0.54.hdf5
------ TRAIN ACCURACY:  weights.29-0.54.hdf5  ------
0.890616621984
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  78   1   1]
 [  0   0   0 ...   3  96   1]
 [  0   0   0 ...   8   3  70]]
------ TEST ACCURACY:  weights.29-0.54.hdf5  ------
0.6
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/7/weights.18-0.51.hdf5
------ TRAIN ACCURACY:  weights.18-0.51.hdf5  ------
0.838337801609
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  75   1  10]
 [  0   0   0 ...   3  90   7]
 [  0   0   0 ...   5   9  84]]
------ TEST ACCURACY:  weights.18-0.51.hdf5  ------
0.623529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 5 0]
 [1 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/8/weights.47-0.53.hdf5
------ TRAIN ACCURACY:  weights.47-0.53.hdf5  ------
0.929222520107
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  96   1   3]
 [  0   0   0 ...   5 100   4]
 [  0   0   0 ...   5   0 102]]
------ TEST ACCURACY:  weights.47-0.53.hdf5  ------
0.564705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 4 1]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/9/weights.22-0.53.hdf5
------ TRAIN ACCURACY:  weights.22-0.53.hdf5  ------
0.860589812332
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  66   0   1]
 [  0   0   0 ...   1  94   1]
 [  0   0   0 ...   4   8  47]]
------ TEST ACCURACY:  weights.22-0.53.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/10/weights.34-0.53.hdf5
------ TRAIN ACCURACY:  weights.34-0.53.hdf5  ------
0.895710455764
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  82   0   5]
 [  0   0   0 ...   1  93   6]
 [  0   0   0 ...   1   3  89]]
------ TEST ACCURACY:  weights.34-0.53.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/11/weights.29-0.53.hdf5
------ TRAIN ACCURACY:  weights.29-0.53.hdf5  ------
0.883109919571
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   0]
 [  0   0   0 ...  12  84   1]
 [  0   0   0 ...  33   1  45]]
------ TEST ACCURACY:  weights.29-0.53.hdf5  ------
0.511764705882
[[4 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/12/weights.48-0.53.hdf5
------ TRAIN ACCURACY:  weights.48-0.53.hdf5  ------
0.932707774799
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  99   0   2]
 [  0   0   0 ...   5  92   3]
 [  0   0   0 ...   8   0  88]]
------ TEST ACCURACY:  weights.48-0.53.hdf5  ------
0.529411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 3 1]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/13/weights.49-0.53.hdf5
------ TRAIN ACCURACY:  weights.49-0.53.hdf5  ------
0.939946380697
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   1]
 [  0   0   0 ...   7  96   1]
 [  0   0   0 ...   6   0  94]]
------ TEST ACCURACY:  weights.49-0.53.hdf5  ------
0.541176470588
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/14/weights.35-0.55.hdf5
------ TRAIN ACCURACY:  weights.35-0.55.hdf5  ------
0.912600536193
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   1   1]
 [  0   0   0 ...   7  99   2]
 [  0   0   0 ...  11   3  91]]
------ TEST ACCURACY:  weights.35-0.55.hdf5  ------
0.564705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 5 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/15/weights.42-0.54.hdf5
------ TRAIN ACCURACY:  weights.42-0.54.hdf5  ------
0.931191432396
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   1]
 [  0   0   0 ...   5  94   7]
 [  0   0   0 ...   6   0  97]]
------ TEST ACCURACY:  weights.42-0.54.hdf5  ------
0.563636363636
[[4 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 2 0 3]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/16/weights.34-0.55.hdf5
------ TRAIN ACCURACY:  weights.34-0.55.hdf5  ------
0.913941018767
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  82   0  12]
 [  0   0   0 ...   3  90  13]
 [  0   0   0 ...   0   0 104]]
------ TEST ACCURACY:  weights.34-0.55.hdf5  ------
0.6
[[5 0 0 ... 0 0 0]
 [3 2 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 3]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/17/weights.30-0.54.hdf5
------ TRAIN ACCURACY:  weights.30-0.54.hdf5  ------
0.8978826052
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  86   0   2]
 [  0   0   0 ...   5  90   3]
 [  0   0   0 ...   3   1  77]]
------ TEST ACCURACY:  weights.30-0.54.hdf5  ------
0.568047337278
[[1 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/18/weights.16-0.56.hdf5
------ TRAIN ACCURACY:  weights.16-0.56.hdf5  ------
0.821179624665
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  61   1   5]
 [  0   0   0 ...   4  83   7]
 [  1   0   0 ...   6   5  77]]
------ TEST ACCURACY:  weights.16-0.56.hdf5  ------
0.582352941176
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 4]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/19/weights.33-0.54.hdf5
------ TRAIN ACCURACY:  weights.33-0.54.hdf5  ------
0.898927613941
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  68   6   7]
 [  0   0   0 ...   2  97   4]
 [  0   0   0 ...   0   5  94]]
------ TEST ACCURACY:  weights.33-0.54.hdf5  ------
0.605882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 1 2]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/20/weights.46-0.53.hdf5
------ TRAIN ACCURACY:  weights.46-0.53.hdf5  ------
0.934118907338
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   2]
 [  0   0   0 ...   7  96   2]
 [  0   0   0 ...   6   0  94]]
------ TEST ACCURACY:  weights.46-0.53.hdf5  ------
0.560240963855
[[5 0 0 ... 0 0 0]
 [1 3 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 2 2 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/21/weights.42-0.52.hdf5
------ TRAIN ACCURACY:  weights.42-0.52.hdf5  ------
0.931367292225
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0   5]
 [  0   0   0 ...   4  99   5]
 [  0   0   0 ...   4   2  92]]
------ TEST ACCURACY:  weights.42-0.52.hdf5  ------
0.582352941176
[[2 0 0 ... 0 0 0]
 [5 0 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 3 1]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/22/weights.20-0.58.hdf5
------ TRAIN ACCURACY:  weights.20-0.58.hdf5  ------
0.848793565684
[[106   0   0 ...   0   0   0]
 [  2 108   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  58   0   7]
 [  0   0   0 ...   2  80   7]
 [  0   0   0 ...   3   2  75]]
------ TEST ACCURACY:  weights.20-0.58.hdf5  ------
0.447058823529
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_svd_conv1d_10/23/weights.41-0.53.hdf5
------ TRAIN ACCURACY:  weights.41-0.53.hdf5  ------
0.931903485255
[[105   1   0 ...   0   0   0]
 [  5 105   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  87   0   2]
 [  0   0   0 ...   1  98   2]
 [  0   0   0 ...   5   0  77]]
------ TEST ACCURACY:  weights.41-0.53.hdf5  ------
0.605882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 2]]
[0.6058823529411764, 0.4176470588235294, 0.5117647058823529, 0.5764705882352941, 0.5705882352941176, 0.6, 0.6235294117647059, 0.5647058823529412, 0.5176470588235295, 0.5176470588235295, 0.5117647058823529, 0.5294117647058824, 0.5411764705882353, 0.5647058823529412, 0.5636363636363636, 0.6, 0.5680473372781065, 0.5823529411764706, 0.6058823529411764, 0.5602409638554217, 0.5823529411764706, 0.4470588235294118, 0.6058823529411764]
0.555147619696
0.0498109050091

Process finished with exit code 0

'''