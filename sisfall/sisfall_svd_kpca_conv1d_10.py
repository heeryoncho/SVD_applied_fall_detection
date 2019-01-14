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
X_kpca = pickle.load(open("data/X_sisfall_kpca.p", "rb"))
X = np.concatenate((X_svd, X_kpca), axis=1)

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

    new_dir = 'model/sisfall_svd_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_svd_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.5470588235294118, 0.35294117647058826, 0.32941176470588235, 0.5, 0.4647058823529412, 0.5058823529411764, 0.47058823529411764, 0.5, 0.5, 0.43529411764705883, 0.4411764705882353, 0.5, 0.4588235294117647, 0.4647058823529412, 0.5454545454545454, 0.5411764705882353, 0.514792899408284, 0.5117647058823529, 0.5235294117647059, 0.536144578313253, 0.5411764705882353, 0.43529411764705883, 0.5470588235294118]
0.485520880803
0.0570768791743
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_svd_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/1/weights.46-0.51.hdf5
------ TRAIN ACCURACY:  weights.46-0.51.hdf5  ------
0.930831099196
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  92   1   1]
 [  0   0   0 ...   6  83   8]
 [  0   0   0 ...   4   0  91]]
------ TEST ACCURACY:  weights.46-0.51.hdf5  ------
0.547058823529
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/2/weights.16-0.50.hdf5
------ TRAIN ACCURACY:  weights.16-0.50.hdf5  ------
0.813404825737
[[105   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  74   4   5]
 [  0   0   0 ...  10  80   3]
 [  0   0   0 ...   7   5  80]]
------ TEST ACCURACY:  weights.16-0.50.hdf5  ------
0.352941176471
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/3/weights.47-0.49.hdf5
------ TRAIN ACCURACY:  weights.47-0.49.hdf5  ------
0.932439678284
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  93   1   2]
 [  0   0   0 ...   6  92   6]
 [  0   0   0 ...   2   1 101]]
------ TEST ACCURACY:  weights.47-0.49.hdf5  ------
0.329411764706
[[1 4 0 ... 0 0 0]
 [0 1 2 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 4 0 0]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/4/weights.45-0.50.hdf5
------ TRAIN ACCURACY:  weights.45-0.50.hdf5  ------
0.933243967828
[[105   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  91   2   4]
 [  0   0   0 ...   6  89   6]
 [  0   0   0 ...   3   1  98]]
------ TEST ACCURACY:  weights.45-0.50.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/5/weights.37-0.48.hdf5
------ TRAIN ACCURACY:  weights.37-0.48.hdf5  ------
0.917158176944
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  86   3   8]
 [  0   0   0 ...   7  90   6]
 [  0   0   1 ...   2   1  99]]
------ TEST ACCURACY:  weights.37-0.48.hdf5  ------
0.464705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 1 1 2]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/6/weights.29-0.50.hdf5
------ TRAIN ACCURACY:  weights.29-0.50.hdf5  ------
0.906434316354
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  74   2  11]
 [  0   0   0 ...   3  83   9]
 [  0   0   0 ...   2   1  97]]
------ TEST ACCURACY:  weights.29-0.50.hdf5  ------
0.505882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/7/weights.50-0.49.hdf5
------ TRAIN ACCURACY:  weights.50-0.49.hdf5  ------
0.932439678284
[[109   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  95   2   2]
 [  0   0   0 ...   6  89   6]
 [  0   0   0 ...   2   1  98]]
------ TEST ACCURACY:  weights.50-0.49.hdf5  ------
0.470588235294
[[5 0 0 ... 0 0 0]
 [1 3 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 1 1]
 [0 0 0 ... 2 3 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/8/weights.37-0.50.hdf5
------ TRAIN ACCURACY:  weights.37-0.50.hdf5  ------
0.922788203753
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  91   3   6]
 [  0   0   0 ...   7  89  11]
 [  0   0   0 ...   4   1  99]]
------ TEST ACCURACY:  weights.37-0.50.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 3]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/9/weights.17-0.50.hdf5
------ TRAIN ACCURACY:  weights.17-0.50.hdf5  ------
0.798659517426
[[104   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  39  25   5]
 [  0   0   0 ...   1  93   0]
 [  0   0   0 ...   1  31  52]]
------ TEST ACCURACY:  weights.17-0.50.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 3 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 2 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/10/weights.33-0.49.hdf5
------ TRAIN ACCURACY:  weights.33-0.49.hdf5  ------
0.903217158177
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  65   2   6]
 [  0   0   0 ...   2  84   5]
 [  0   0   0 ...   1   1  86]]
------ TEST ACCURACY:  weights.33-0.49.hdf5  ------
0.435294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/11/weights.27-0.49.hdf5
------ TRAIN ACCURACY:  weights.27-0.49.hdf5  ------
0.882573726542
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  97 ...   0   0   0]
 ...
 [  0   0   0 ...  88   2   5]
 [  0   0   0 ...   9  90   3]
 [  0   0   0 ...   7   3  90]]
------ TEST ACCURACY:  weights.27-0.49.hdf5  ------
0.441176470588
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 1]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/12/weights.29-0.50.hdf5
------ TRAIN ACCURACY:  weights.29-0.50.hdf5  ------
0.891689008043
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  83   1  11]
 [  0   0   0 ...   8  82  14]
 [  0   0   1 ...   3   0  97]]
------ TEST ACCURACY:  weights.29-0.50.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/13/weights.46-0.49.hdf5
------ TRAIN ACCURACY:  weights.46-0.49.hdf5  ------
0.929222520107
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  93   1   3]
 [  0   0   0 ...   4  93   4]
 [  0   0   0 ...   3   0  99]]
------ TEST ACCURACY:  weights.46-0.49.hdf5  ------
0.458823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/14/weights.15-0.47.hdf5
------ TRAIN ACCURACY:  weights.15-0.47.hdf5  ------
0.792225201072
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  96 ...   0   0   0]
 ...
 [  0   0   0 ...  84   5   2]
 [  0   0   0 ...  14  82   1]
 [  0   0   0 ...  24  10  50]]
------ TEST ACCURACY:  weights.15-0.47.hdf5  ------
0.464705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/15/weights.23-0.49.hdf5
------ TRAIN ACCURACY:  weights.23-0.49.hdf5  ------
0.864257028112
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ...  65  10   7]
 [  0   0   0 ...   5  89   0]
 [  0   0   0 ...   4   7  86]]
------ TEST ACCURACY:  weights.23-0.49.hdf5  ------
0.545454545455
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/16/weights.45-0.49.hdf5
------ TRAIN ACCURACY:  weights.45-0.49.hdf5  ------
0.927077747989
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  79   2   8]
 [  0   0   0 ...   4  91   9]
 [  0   0   0 ...   0   1 103]]
------ TEST ACCURACY:  weights.45-0.49.hdf5  ------
0.541176470588
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/17/weights.46-0.48.hdf5
------ TRAIN ACCURACY:  weights.46-0.48.hdf5  ------
0.92414902171
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  88   2   5]
 [  0   0   0 ...   3  88   5]
 [  0   0   0 ...   1   1 100]]
------ TEST ACCURACY:  weights.46-0.48.hdf5  ------
0.514792899408
[[0 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/18/weights.36-0.53.hdf5
------ TRAIN ACCURACY:  weights.36-0.53.hdf5  ------
0.909651474531
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  85   0   7]
 [  0   0   0 ...   9  85   7]
 [  0   0   0 ...   1   0 104]]
------ TEST ACCURACY:  weights.36-0.53.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/19/weights.15-0.48.hdf5
------ TRAIN ACCURACY:  weights.15-0.48.hdf5  ------
0.797050938338
[[105   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  51  11   9]
 [  0   0   0 ...   7  82   2]
 [  0   0   0 ...   4   5  78]]
------ TEST ACCURACY:  weights.15-0.48.hdf5  ------
0.523529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 2 2 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/20/weights.50-0.50.hdf5
------ TRAIN ACCURACY:  weights.50-0.50.hdf5  ------
0.933583288698
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  93   6   2]
 [  0   0   0 ...   5  97   1]
 [  0   0   0 ...   5   5  94]]
------ TEST ACCURACY:  weights.50-0.50.hdf5  ------
0.536144578313
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 4 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/21/weights.39-0.49.hdf5
------ TRAIN ACCURACY:  weights.39-0.49.hdf5  ------
0.916353887399
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  94   1   3]
 [  0   0   0 ...   4  95   4]
 [  0   0   0 ...   6   8  92]]
------ TEST ACCURACY:  weights.39-0.49.hdf5  ------
0.541176470588
[[2 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 2 0]
 [0 0 0 ... 1 3 0]
 [0 0 0 ... 3 1 0]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/22/weights.30-0.53.hdf5
------ TRAIN ACCURACY:  weights.30-0.53.hdf5  ------
0.906970509383
[[105   0   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  92   1   2]
 [  0   0   0 ...   9  86   4]
 [  0   0   0 ...  12   2  88]]
------ TEST ACCURACY:  weights.30-0.53.hdf5  ------
0.435294117647
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 1]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_svd_kpca_conv1d_10/23/weights.41-0.49.hdf5
------ TRAIN ACCURACY:  weights.41-0.49.hdf5  ------
0.928954423592
[[107   0   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  96   1   1]
 [  0   0   0 ...   4  93   1]
 [  0   0   0 ...   7   4  89]]
------ TEST ACCURACY:  weights.41-0.49.hdf5  ------
0.547058823529
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 1 0 2]]
[0.5470588235294118, 0.35294117647058826, 0.32941176470588235, 0.5, 0.4647058823529412, 0.5058823529411764, 0.47058823529411764, 0.5, 0.5, 0.43529411764705883, 0.4411764705882353, 0.5, 0.4588235294117647, 0.4647058823529412, 0.5454545454545454, 0.5411764705882353, 0.514792899408284, 0.5117647058823529, 0.5235294117647059, 0.536144578313253, 0.5411764705882353, 0.43529411764705883, 0.5470588235294118]
0.485520880803
0.0570768791743

Process finished with exit code 0

'''