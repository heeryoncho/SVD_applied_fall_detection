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
X_smv = pickle.load(open("data/X_sisfall_smv.p", "rb"))
X = np.concatenate((X_spca, X_smv), axis=1)

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

    new_dir = 'model/sisfall_smv_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_smv_spca_conv1d_10/' + str(i+1) + '/'
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
[0.5823529411764706, 0.4823529411764706, 0.47058823529411764, 0.5294117647058824, 0.5058823529411764, 0.5470588235294118, 0.5823529411764706, 0.5470588235294118, 0.4470588235294118, 0.5411764705882353, 0.5117647058823529, 0.49411764705882355, 0.5823529411764706, 0.47058823529411764, 0.48484848484848486, 0.5117647058823529, 0.5680473372781065, 0.5588235294117647, 0.6058823529411764, 0.5240963855421686, 0.5647058823529412, 0.5411764705882353, 0.5]
0.528411425909
0.0416791154476
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_smv_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/1/weights.46-0.53.hdf5
------ TRAIN ACCURACY:  weights.46-0.53.hdf5  ------
0.902144772118
[[103   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  93   3   4]
 [  0   0   0 ...   2  95   5]
 [  0   0   0 ...   2   5  84]]
------ TEST ACCURACY:  weights.46-0.53.hdf5  ------
0.582352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 1 2]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/2/weights.50-0.53.hdf5
------ TRAIN ACCURACY:  weights.50-0.53.hdf5  ------
0.89490616622
[[102   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  98   1   4]
 [  0   0   0 ...   3  87   8]
 [  0   0   0 ...   3   1  90]]
------ TEST ACCURACY:  weights.50-0.53.hdf5  ------
0.482352941176
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/3/weights.48-0.54.hdf5
------ TRAIN ACCURACY:  weights.48-0.54.hdf5  ------
0.903753351206
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ... 101   2   3]
 [  0   0   0 ...   7  96   2]
 [  0   0   0 ...  10   1  89]]
------ TEST ACCURACY:  weights.48-0.54.hdf5  ------
0.470588235294
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 3 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/4/weights.46-0.51.hdf5
------ TRAIN ACCURACY:  weights.46-0.51.hdf5  ------
0.882305630027
[[104   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   6]
 [  0   0   0 ...   1  80  21]
 [  0   0   0 ...   2   0 100]]
------ TEST ACCURACY:  weights.46-0.51.hdf5  ------
0.529411764706
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/5/weights.49-0.54.hdf5
------ TRAIN ACCURACY:  weights.49-0.54.hdf5  ------
0.894638069705
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ... 101   1   1]
 [  0   0   0 ...   5  92   1]
 [  0   0   0 ...   6   3  78]]
------ TEST ACCURACY:  weights.49-0.54.hdf5  ------
0.505882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 2 2]
 [0 0 0 ... 2 2 0]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/6/weights.49-0.52.hdf5
------ TRAIN ACCURACY:  weights.49-0.52.hdf5  ------
0.897319034853
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ... 104   1   3]
 [  0   0   0 ...   6  91   3]
 [  0   0   0 ...  11   1  76]]
------ TEST ACCURACY:  weights.49-0.52.hdf5  ------
0.547058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/7/weights.43-0.50.hdf5
------ TRAIN ACCURACY:  weights.43-0.50.hdf5  ------
0.829490616622
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  85   5   1]
 [  0   0   0 ...   0  95   0]
 [  0   0   0 ...   2  15  68]]
------ TEST ACCURACY:  weights.43-0.50.hdf5  ------
0.582352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 1 1]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/8/weights.49-0.52.hdf5
------ TRAIN ACCURACY:  weights.49-0.52.hdf5  ------
0.871045576408
[[ 99   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  93   2   8]
 [  0   0   0 ...   0 100   6]
 [  0   0   0 ...   1   4  95]]
------ TEST ACCURACY:  weights.49-0.52.hdf5  ------
0.547058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/9/weights.39-0.51.hdf5
------ TRAIN ACCURACY:  weights.39-0.51.hdf5  ------
0.891152815013
[[101   6   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  94   1   9]
 [  0   0   0 ...   1  97   6]
 [  0   0   0 ...   2   3  96]]
------ TEST ACCURACY:  weights.39-0.51.hdf5  ------
0.447058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 1 3]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/10/weights.42-0.55.hdf5
------ TRAIN ACCURACY:  weights.42-0.55.hdf5  ------
0.891152815013
[[103   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  97   2   3]
 [  0   0   0 ...   3  95   5]
 [  0   0   0 ...   3   5  84]]
------ TEST ACCURACY:  weights.42-0.55.hdf5  ------
0.541176470588
[[1 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 3 1 0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/11/weights.48-0.54.hdf5
------ TRAIN ACCURACY:  weights.48-0.54.hdf5  ------
0.907506702413
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   3]
 [  0   0   0 ...   2 101   2]
 [  0   0   0 ...   2   4  89]]
------ TEST ACCURACY:  weights.48-0.54.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/12/weights.40-0.55.hdf5
------ TRAIN ACCURACY:  weights.40-0.55.hdf5  ------
0.863002680965
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  99   0   3]
 [  0   0   0 ...   7  83   8]
 [  0   0   0 ...   9   0  83]]
------ TEST ACCURACY:  weights.40-0.55.hdf5  ------
0.494117647059
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 3 1]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/13/weights.50-0.52.hdf5
------ TRAIN ACCURACY:  weights.50-0.52.hdf5  ------
0.874798927614
[[105   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  95   2   1]
 [  0   0   0 ...   2  99   0]
 [  0   0   0 ...   5   5  77]]
------ TEST ACCURACY:  weights.50-0.52.hdf5  ------
0.582352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/14/weights.50-0.50.hdf5
------ TRAIN ACCURACY:  weights.50-0.50.hdf5  ------
0.875603217158
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ... 104   0   2]
 [  0   0   0 ...  16  65   7]
 [  0   0   0 ...  10   0  82]]
------ TEST ACCURACY:  weights.50-0.50.hdf5  ------
0.470588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/15/weights.45-0.54.hdf5
------ TRAIN ACCURACY:  weights.45-0.54.hdf5  ------
0.881927710843
[[105   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  91   0   4]
 [  0   0   0 ...   1  78  14]
 [  0   0   0 ...   1   0  95]]
------ TEST ACCURACY:  weights.45-0.54.hdf5  ------
0.484848484848
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/16/weights.48-0.52.hdf5
------ TRAIN ACCURACY:  weights.48-0.52.hdf5  ------
0.883914209115
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   2]
 [  0   0   0 ...   2  91   8]
 [  0   0   0 ...   2   0  95]]
------ TEST ACCURACY:  weights.48-0.52.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 1 0 3]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/17/weights.50-0.53.hdf5
------ TRAIN ACCURACY:  weights.50-0.53.hdf5  ------
0.878584829804
[[102   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  69   1   6]
 [  0   0   0 ...   0  90   3]
 [  0   0   0 ...   1   5  88]]
------ TEST ACCURACY:  weights.50-0.53.hdf5  ------
0.568047337278
[[4 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/18/weights.43-0.50.hdf5
------ TRAIN ACCURACY:  weights.43-0.50.hdf5  ------
0.865951742627
[[ 97   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  99 ...   0   0   0]
 ...
 [  0   0   0 ...  90   2  11]
 [  0   0   0 ...   2  86  13]
 [  0   0   0 ...   0   0 103]]
------ TEST ACCURACY:  weights.43-0.50.hdf5  ------
0.558823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 4]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/19/weights.46-0.53.hdf5
------ TRAIN ACCURACY:  weights.46-0.53.hdf5  ------
0.886327077748
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  88   0   1]
 [  0   0   0 ...   0  94   0]
 [  0   0   0 ...   0   6  65]]
------ TEST ACCURACY:  weights.46-0.53.hdf5  ------
0.605882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 2 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/20/weights.39-0.53.hdf5
------ TRAIN ACCURACY:  weights.39-0.53.hdf5  ------
0.85270487413
[[102   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  83  14   1]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   2  30  66]]
------ TEST ACCURACY:  weights.39-0.53.hdf5  ------
0.524096385542
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 4 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/21/weights.36-0.51.hdf5
------ TRAIN ACCURACY:  weights.36-0.51.hdf5  ------
0.844235924933
[[104   6   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  97   1   4]
 [  0   0   0 ...  13  75   5]
 [  0   0   0 ...  15   0  81]]
------ TEST ACCURACY:  weights.36-0.51.hdf5  ------
0.564705882353
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 1 2 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/22/weights.48-0.54.hdf5
------ TRAIN ACCURACY:  weights.48-0.54.hdf5  ------
0.863538873995
[[109   0   0 ...   0   0   0]
 [  3 107   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  91   1   2]
 [  0   0   0 ...   0 102   1]
 [  0   0   0 ...   0  10  85]]
------ TEST ACCURACY:  weights.48-0.54.hdf5  ------
0.541176470588
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_smv_spca_conv1d_10/23/weights.41-0.51.hdf5
------ TRAIN ACCURACY:  weights.41-0.51.hdf5  ------
0.839946380697
[[106   2   0 ...   0   0   0]
 [  5 105   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  67  10   1]
 [  0   0   0 ...   0  95   0]
 [  0   0   0 ...   1  15  62]]
------ TEST ACCURACY:  weights.41-0.51.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
[0.5823529411764706, 0.4823529411764706, 0.47058823529411764, 0.5294117647058824, 0.5058823529411764, 0.5470588235294118, 0.5823529411764706, 0.5470588235294118, 0.4470588235294118, 0.5411764705882353, 0.5117647058823529, 0.49411764705882355, 0.5823529411764706, 0.47058823529411764, 0.48484848484848486, 0.5117647058823529, 0.5680473372781065, 0.5588235294117647, 0.6058823529411764, 0.5240963855421686, 0.5647058823529412, 0.5411764705882353, 0.5]
0.528411425909
0.0416791154476

Process finished with exit code 0

'''
