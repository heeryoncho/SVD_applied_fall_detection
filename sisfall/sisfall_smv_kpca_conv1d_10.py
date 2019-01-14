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

X_kpca = pickle.load(open("data/X_sisfall_kpca.p", "rb"))
X_smv = pickle.load(open("data/X_sisfall_smv.p", "rb"))
X = np.concatenate((X_kpca, X_smv), axis=1)

y = pickle.load(open("data/y_sisfall_kpca.p", "rb"))

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
    print X_train.shape # (3730, 900)

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

    new_dir = 'model/sisfall_smv_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_smv_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.5764705882352941, 0.45294117647058824, 0.47058823529411764, 0.5588235294117647, 0.5352941176470588, 0.5529411764705883, 0.5647058823529412, 0.5235294117647059, 0.5176470588235295, 0.5705882352941176, 0.49411764705882355, 0.5411764705882353, 0.5352941176470588, 0.5, 0.5272727272727272, 0.5705882352941176, 0.5976331360946746, 0.5705882352941176, 0.5647058823529412, 0.6506024096385542, 0.5647058823529412, 0.47058823529411764, 0.6]
0.543947930028
0.0452937096121
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_smv_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/1/weights.28-0.52.hdf5
------ TRAIN ACCURACY:  weights.28-0.52.hdf5  ------
0.890348525469
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  88   1   0]
 [  0   0   0 ...   5  96   0]
 [  0   0   0 ...  13   5  46]]
------ TEST ACCURACY:  weights.28-0.52.hdf5  ------
0.576470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 3 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/2/weights.31-0.55.hdf5
------ TRAIN ACCURACY:  weights.31-0.55.hdf5  ------
0.887935656836
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  85   1   1]
 [  0   0   0 ...   2  94   2]
 [  0   0   0 ...  12   1  74]]
------ TEST ACCURACY:  weights.31-0.55.hdf5  ------
0.452941176471
[[0 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/3/weights.25-0.55.hdf5
------ TRAIN ACCURACY:  weights.25-0.55.hdf5  ------
0.900804289544
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  80   0   5]
 [  0   0   0 ...   1  92   7]
 [  0   0   0 ...   5   2  83]]
------ TEST ACCURACY:  weights.25-0.55.hdf5  ------
0.470588235294
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/4/weights.48-0.54.hdf5
------ TRAIN ACCURACY:  weights.48-0.54.hdf5  ------
0.930831099196
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   0]
 [  0   0   0 ...   9  83   4]
 [  0   0   0 ...  10   0  78]]
------ TEST ACCURACY:  weights.48-0.54.hdf5  ------
0.558823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/5/weights.36-0.54.hdf5
------ TRAIN ACCURACY:  weights.36-0.54.hdf5  ------
0.923056300268
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  87   0   2]
 [  0   0   0 ...   2  92   7]
 [  0   0   0 ...   1   0  97]]
------ TEST ACCURACY:  weights.36-0.54.hdf5  ------
0.535294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 4]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/6/weights.42-0.53.hdf5
------ TRAIN ACCURACY:  weights.42-0.53.hdf5  ------
0.928686327078
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  90   0   1]
 [  0   0   0 ...   2  96   4]
 [  0   0   0 ...   1   0  90]]
------ TEST ACCURACY:  weights.42-0.53.hdf5  ------
0.552941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/7/weights.22-0.52.hdf5
------ TRAIN ACCURACY:  weights.22-0.52.hdf5  ------
0.86890080429
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  77   0   0]
 [  0   0   0 ...   3  84   3]
 [  0   0   0 ...  15   3  50]]
------ TEST ACCURACY:  weights.22-0.52.hdf5  ------
0.564705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/8/weights.22-0.53.hdf5
------ TRAIN ACCURACY:  weights.22-0.53.hdf5  ------
0.855764075067
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  36   8  17]
 [  0   0   0 ...   0 101   7]
 [  0   0   0 ...   2  10  78]]
------ TEST ACCURACY:  weights.22-0.53.hdf5  ------
0.523529411765
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 3]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/9/weights.32-0.54.hdf5
------ TRAIN ACCURACY:  weights.32-0.54.hdf5  ------
0.909115281501
[[103   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  88   0   2]
 [  0   0   0 ...   1  90   4]
 [  0   0   0 ...   2   2  93]]
------ TEST ACCURACY:  weights.32-0.54.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/10/weights.36-0.54.hdf5
------ TRAIN ACCURACY:  weights.36-0.54.hdf5  ------
0.92144772118
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  89   1   3]
 [  0   0   0 ...   2  95   7]
 [  0   0   0 ...   4   3  85]]
------ TEST ACCURACY:  weights.36-0.54.hdf5  ------
0.570588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/11/weights.41-0.54.hdf5
------ TRAIN ACCURACY:  weights.41-0.54.hdf5  ------
0.932439678284
[[106   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  91   0   4]
 [  0   0   0 ...   5  83  15]
 [  0   0   0 ...   3   0  97]]
------ TEST ACCURACY:  weights.41-0.54.hdf5  ------
0.494117647059
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/12/weights.29-0.55.hdf5
------ TRAIN ACCURACY:  weights.29-0.55.hdf5  ------
0.906970509383
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  94 ...   0   0   0]
 ...
 [  0   0   0 ...  67   3  11]
 [  0   0   0 ...   2  99   6]
 [  0   0   0 ...   3   8  84]]
------ TEST ACCURACY:  weights.29-0.55.hdf5  ------
0.541176470588
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/13/weights.12-0.53.hdf5
------ TRAIN ACCURACY:  weights.12-0.53.hdf5  ------
0.744235924933
[[103   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  97 ...   0   0   0]
 ...
 [  0   0   0 ...  99   0   1]
 [  0   0   0 ...  23  37  17]
 [  0   0   0 ...  38   0  56]]
------ TEST ACCURACY:  weights.12-0.53.hdf5  ------
0.535294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 5 0 0]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/14/weights.40-0.53.hdf5
------ TRAIN ACCURACY:  weights.40-0.53.hdf5  ------
0.922520107239
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   2]
 [  0   0   0 ...   5  90   6]
 [  0   0   0 ...   3   0  93]]
------ TEST ACCURACY:  weights.40-0.53.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/15/weights.43-0.52.hdf5
------ TRAIN ACCURACY:  weights.43-0.52.hdf5  ------
0.909504685408
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ... 103   0   1]
 [  0   0   0 ...   9  89  10]
 [  0   0   0 ...  17   0  88]]
------ TEST ACCURACY:  weights.43-0.52.hdf5  ------
0.527272727273
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 3 0 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/16/weights.44-0.53.hdf5
------ TRAIN ACCURACY:  weights.44-0.53.hdf5  ------
0.925737265416
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 103   0   0]
 [  0   0   0 ...  10  86   7]
 [  0   0   0 ...   6   0  91]]
------ TEST ACCURACY:  weights.44-0.53.hdf5  ------
0.570588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 3 0 2]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/17/weights.39-0.56.hdf5
------ TRAIN ACCURACY:  weights.39-0.56.hdf5  ------
0.933261860091
[[105   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   0]
 [  0   0   0 ...   6  93   1]
 [  0   0   0 ...  12   0  85]]
------ TEST ACCURACY:  weights.39-0.56.hdf5  ------
0.597633136095
[[0 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/18/weights.23-0.53.hdf5
------ TRAIN ACCURACY:  weights.23-0.53.hdf5  ------
0.859249329759
[[103   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  48   6  18]
 [  0   0   0 ...   1  94  10]
 [  0   0   0 ...   0   7  97]]
------ TEST ACCURACY:  weights.23-0.53.hdf5  ------
0.570588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 2 2]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/19/weights.30-0.53.hdf5
------ TRAIN ACCURACY:  weights.30-0.53.hdf5  ------
0.89436997319
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  86   0   1]
 [  0   0   0 ...   4  69  10]
 [  0   0   0 ...   3   0  81]]
------ TEST ACCURACY:  weights.30-0.53.hdf5  ------
0.564705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/20/weights.34-0.53.hdf5
------ TRAIN ACCURACY:  weights.34-0.53.hdf5  ------
0.919657204071
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   1]
 [  0   0   0 ...   7  97   1]
 [  0   0   0 ...  11   5  75]]
------ TEST ACCURACY:  weights.34-0.53.hdf5  ------
0.650602409639
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/21/weights.16-0.52.hdf5
------ TRAIN ACCURACY:  weights.16-0.52.hdf5  ------
0.791689008043
[[101   8   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  58 ...   0   0   0]
 ...
 [  0   0   0 ...  67   4   0]
 [  0   0   0 ...   3  89   2]
 [  0   0   0 ...  14  11  27]]
------ TEST ACCURACY:  weights.16-0.52.hdf5  ------
0.564705882353
[[2 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/22/weights.47-0.60.hdf5
------ TRAIN ACCURACY:  weights.47-0.60.hdf5  ------
0.935924932976
[[108   0   0 ...   0   0   0]
 [  2 108   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  96   0   0]
 [  0   0   0 ...   3  99   0]
 [  0   0   0 ...   6   2  83]]
------ TEST ACCURACY:  weights.47-0.60.hdf5  ------
0.470588235294
[[3 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 2 1 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_kpca_conv1d_10/23/weights.27-0.53.hdf5
------ TRAIN ACCURACY:  weights.27-0.53.hdf5  ------
0.878284182306
[[104   2   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  59   1   3]
 [  0   0   0 ...   0  98   2]
 [  0   0   0 ...   1   4  68]]
------ TEST ACCURACY:  weights.27-0.53.hdf5  ------
0.6
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 0 1]]
[0.5764705882352941, 0.45294117647058824, 0.47058823529411764, 0.5588235294117647, 0.5352941176470588, 0.5529411764705883, 0.5647058823529412, 0.5235294117647059, 0.5176470588235295, 0.5705882352941176, 0.49411764705882355, 0.5411764705882353, 0.5352941176470588, 0.5, 0.5272727272727272, 0.5705882352941176, 0.5976331360946746, 0.5705882352941176, 0.5647058823529412, 0.6506024096385542, 0.5647058823529412, 0.47058823529411764, 0.6]
0.543947930028
0.0452937096121

Process finished with exit code 0

'''
