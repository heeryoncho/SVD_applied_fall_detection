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

X_smv = pickle.load(open("data/X_sisfall_smv.p", "rb"))
X_raw = pickle.load(open("data/X_sisfall_raw.p", "rb"))
X = np.concatenate((X_smv, X_raw), axis=1)

y = pickle.load(open("data/y_sisfall_raw.p", "rb"))

n_classes = 34
signal_rows = 1800
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

    new_dir = 'sisfall/model/sisfall_raw_smv_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_raw_smv_conv1d_10/' + str(i+1) + '/'
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
[0.6352941176470588, 0.5411764705882353, 0.5411764705882353, 0.6352941176470588, 0.6588235294117647, 0.5352941176470588, 0.5588235294117647, 0.6647058823529411, 0.7235294117647059, 0.5647058823529412, 0.5058823529411764, 0.6647058823529411, 0.6058823529411764, 0.611764705882353, 0.49696969696969695, 0.6235294117647059, 0.5857988165680473, 0.6705882352941176, 0.6235294117647059, 0.6385542168674698, 0.6941176470588235, 0.4647058823529412, 0.6705882352941176]
0.605019146846
0.0667974180588
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_raw_smv_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/1/weights.36-0.61.hdf5
------ TRAIN ACCURACY:  weights.36-0.61.hdf5  ------
0.923860589812
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  93   0  12]
 [  0   0   0 ...   0 104   1]
 [  0   0   0 ...   1   0 107]]
------ TEST ACCURACY:  weights.36-0.61.hdf5  ------
0.635294117647
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/2/weights.32-0.60.hdf5
------ TRAIN ACCURACY:  weights.32-0.60.hdf5  ------
0.895174262735
[[102   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0   2]
 [  0   0   0 ...   0 104   0]
 [  0   0   0 ...   0   0  97]]
------ TEST ACCURACY:  weights.32-0.60.hdf5  ------
0.541176470588
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/3/weights.49-0.59.hdf5
------ TRAIN ACCURACY:  weights.49-0.59.hdf5  ------
0.923056300268
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   5]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   0   0 106]]
------ TEST ACCURACY:  weights.49-0.59.hdf5  ------
0.541176470588
[[3 2 0 ... 0 0 0]
 [0 1 4 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/4/weights.44-0.62.hdf5
------ TRAIN ACCURACY:  weights.44-0.62.hdf5  ------
0.94691689008
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  96   0  10]
 [  0   0   0 ...   0 106   1]
 [  0   0   0 ...   0   0 105]]
------ TEST ACCURACY:  weights.44-0.62.hdf5  ------
0.635294117647
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/5/weights.30-0.61.hdf5
------ TRAIN ACCURACY:  weights.30-0.61.hdf5  ------
0.915013404826
[[ 92   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  97   1   3]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   1   2 100]]
------ TEST ACCURACY:  weights.30-0.61.hdf5  ------
0.658823529412
[[2 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 4 1]
 [0 0 0 ... 1 1 1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/6/weights.45-0.58.hdf5
------ TRAIN ACCURACY:  weights.45-0.58.hdf5  ------
0.885522788204
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ... 103   0   2]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   1   0 100]]
------ TEST ACCURACY:  weights.45-0.58.hdf5  ------
0.535294117647
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/7/weights.19-0.60.hdf5
------ TRAIN ACCURACY:  weights.19-0.60.hdf5  ------
0.861930294906
[[ 98   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  90 ...   0   0   0]
 ...
 [  0   0   0 ...  65   3  23]
 [  0   0   0 ...   0 106   1]
 [  0   0   0 ...   0   2 105]]
------ TEST ACCURACY:  weights.19-0.60.hdf5  ------
0.558823529412
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/8/weights.46-0.62.hdf5
------ TRAIN ACCURACY:  weights.46-0.62.hdf5  ------
0.922252010724
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ...  99   0   6]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   0   0 107]]
------ TEST ACCURACY:  weights.46-0.62.hdf5  ------
0.664705882353
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/9/weights.25-0.61.hdf5
------ TRAIN ACCURACY:  weights.25-0.61.hdf5  ------
0.905361930295
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  87   0   8]
 [  0   0   0 ...   0 104   0]
 [  0   0   0 ...   0   0 104]]
------ TEST ACCURACY:  weights.25-0.61.hdf5  ------
0.723529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/10/weights.48-0.62.hdf5
------ TRAIN ACCURACY:  weights.48-0.62.hdf5  ------
0.946380697051
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 102   0   3]
 [  0   0   0 ...   1 103   0]
 [  0   0   0 ...   0   0 103]]
------ TEST ACCURACY:  weights.48-0.62.hdf5  ------
0.564705882353
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/11/weights.26-0.59.hdf5
------ TRAIN ACCURACY:  weights.26-0.59.hdf5  ------
0.885522788204
[[101   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  73   0  15]
 [  0   0   0 ...   0 102   2]
 [  0   0   0 ...   0   0 107]]
------ TEST ACCURACY:  weights.26-0.59.hdf5  ------
0.505882352941
[[0 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/12/weights.34-0.60.hdf5
------ TRAIN ACCURACY:  weights.34-0.60.hdf5  ------
0.90509383378
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0  10]
 [  0   0   0 ...   0 104   1]
 [  0   0   0 ...   0   0 104]]
------ TEST ACCURACY:  weights.34-0.60.hdf5  ------
0.664705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/13/weights.24-0.62.hdf5
------ TRAIN ACCURACY:  weights.24-0.62.hdf5  ------
0.908042895442
[[103   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0   4]
 [  0   0   0 ...   0 103   1]
 [  0   0   0 ...   1   0 102]]
------ TEST ACCURACY:  weights.24-0.62.hdf5  ------
0.605882352941
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 4 0 0]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/14/weights.38-0.60.hdf5
------ TRAIN ACCURACY:  weights.38-0.60.hdf5  ------
0.936193029491
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  98   0   4]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   1   1 101]]
------ TEST ACCURACY:  weights.38-0.60.hdf5  ------
0.611764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 2 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/15/weights.37-0.61.hdf5
------ TRAIN ACCURACY:  weights.37-0.61.hdf5  ------
0.931726907631
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  92   0   6]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   0   2 100]]
------ TEST ACCURACY:  weights.37-0.61.hdf5  ------
0.49696969697
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/16/weights.36-0.63.hdf5
------ TRAIN ACCURACY:  weights.36-0.63.hdf5  ------
0.921715817694
[[103   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   2]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   3   0 102]]
------ TEST ACCURACY:  weights.36-0.63.hdf5  ------
0.623529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/17/weights.49-0.60.hdf5
------ TRAIN ACCURACY:  weights.49-0.60.hdf5  ------
0.934333958724
[[108   2   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   9]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   1   0 104]]
------ TEST ACCURACY:  weights.49-0.60.hdf5  ------
0.585798816568
[[1 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/18/weights.42-0.59.hdf5
------ TRAIN ACCURACY:  weights.42-0.59.hdf5  ------
0.913404825737
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   3]
 [  0   0   0 ...   2 102   0]
 [  0   0   0 ...   1   0 100]]
------ TEST ACCURACY:  weights.42-0.59.hdf5  ------
0.670588235294
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/19/weights.50-0.60.hdf5
------ TRAIN ACCURACY:  weights.50-0.60.hdf5  ------
0.941554959786
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 105   0   2]
 [  0   0   0 ...   1 105   0]
 [  0   0   0 ...   3   0  98]]
------ TEST ACCURACY:  weights.50-0.60.hdf5  ------
0.623529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/20/weights.32-0.60.hdf5
------ TRAIN ACCURACY:  weights.32-0.60.hdf5  ------
0.90867702196
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  83   4  12]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   0   1 104]]
------ TEST ACCURACY:  weights.32-0.60.hdf5  ------
0.638554216867
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 3]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/21/weights.40-0.61.hdf5
------ TRAIN ACCURACY:  weights.40-0.61.hdf5  ------
0.939142091153
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  94   1   8]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   0   0 100]]
------ TEST ACCURACY:  weights.40-0.61.hdf5  ------
0.694117647059
[[1 0 0 ... 0 0 0]
 [2 1 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/22/weights.32-0.63.hdf5
------ TRAIN ACCURACY:  weights.32-0.63.hdf5  ------
0.875603217158
[[107   0   0 ...   0   0   0]
 [  1 108   0 ...   0   0   0]
 [  0   1   0 ...   0   0   0]
 ...
 [  0   0   0 ...  94   0   5]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   0   0 104]]
------ TEST ACCURACY:  weights.32-0.63.hdf5  ------
0.464705882353
[[2 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 4]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_raw_smv_conv1d_10/23/weights.47-0.62.hdf5
------ TRAIN ACCURACY:  weights.47-0.62.hdf5  ------
0.951742627346
[[104   4   0 ...   0   0   0]
 [  2 107   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   4]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   0   1  96]]
------ TEST ACCURACY:  weights.47-0.62.hdf5  ------
0.670588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 2 0 3]]
[0.6352941176470588, 0.5411764705882353, 0.5411764705882353, 0.6352941176470588, 0.6588235294117647, 0.5352941176470588, 0.5588235294117647, 0.6647058823529411, 0.7235294117647059, 0.5647058823529412, 0.5058823529411764, 0.6647058823529411, 0.6058823529411764, 0.611764705882353, 0.49696969696969695, 0.6235294117647059, 0.5857988165680473, 0.6705882352941176, 0.6235294117647059, 0.6385542168674698, 0.6941176470588235, 0.4647058823529412, 0.6705882352941176]
0.605019146846
0.0667974180588

Process finished with exit code 0

'''