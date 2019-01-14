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

X = pickle.load(open("data/X_sisfall_raw.p", "rb"))
y = pickle.load(open("data/y_sisfall_raw.p", "rb"))

n_classes = 34
signal_rows = 1350
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
    print X_train.shape # (3730, 1350)

    # input layer
    input_signal = Input(shape=(signal_rows, 1))
    print K.int_shape(input_signal) # (None, 1350, 1)

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

    new_dir = 'model/sisfall_raw_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_raw_conv1d_10/' + str(i+1) + '/'
    for path, dirs, files in os.walk(path_str):
        dirs.sort()
        files.sort()
        top_acc = []
        top_acc.append(files[-1])
        files = top_acc
        for file in files:
            print file
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
[0.6705882352941176, 0.5117647058823529, 0.6058823529411764, 0.611764705882353, 0.6529411764705882, 0.6588235294117647, 0.6235294117647059, 0.6, 0.6705882352941176, 0.5470588235294118, 0.45294117647058824, 0.6764705882352942, 0.6411764705882353, 0.5411764705882353, 0.4484848484848485, 0.6, 0.6390532544378699, 0.7352941176470589, 0.6647058823529411, 0.6686746987951807, 0.7, 0.5235294117647059, 0.6705882352941176]
0.613697231788
0.074710425358
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_raw_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
weights.46-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/1/weights.46-0.63.hdf5
------ TRAIN ACCURACY:  weights.46-0.63.hdf5  ------
0.945040214477
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   6]
 [  0   0   0 ...   0 104   2]
 [  0   0   0 ...   0   1 104]]
------ TEST ACCURACY:  weights.46-0.63.hdf5  ------
0.670588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 2 0 3]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
weights.23-0.60.hdf5
========================================
model/sisfall_raw_conv1d_10/2/weights.23-0.60.hdf5
------ TRAIN ACCURACY:  weights.23-0.60.hdf5  ------
0.875603217158
[[106   4   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ...  82   2  12]
 [  0   0   0 ...   0 107   1]
 [  0   0   0 ...   0   2  97]]
------ TEST ACCURACY:  weights.23-0.60.hdf5  ------
0.511764705882
[[2 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
weights.42-0.64.hdf5
========================================
model/sisfall_raw_conv1d_10/3/weights.42-0.64.hdf5
------ TRAIN ACCURACY:  weights.42-0.64.hdf5  ------
0.953351206434
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ... 102   0   2]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   2   1 102]]
------ TEST ACCURACY:  weights.42-0.64.hdf5  ------
0.605882352941
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
weights.48-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/4/weights.48-0.63.hdf5
------ TRAIN ACCURACY:  weights.48-0.63.hdf5  ------
0.951206434316
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   2]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   0   0 105]]
------ TEST ACCURACY:  weights.48-0.63.hdf5  ------
0.611764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
weights.39-0.62.hdf5
========================================
model/sisfall_raw_conv1d_10/5/weights.39-0.62.hdf5
------ TRAIN ACCURACY:  weights.39-0.62.hdf5  ------
0.927077747989
[[102   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  94   1   5]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   0  99]]
------ TEST ACCURACY:  weights.39-0.62.hdf5  ------
0.652941176471
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
weights.44-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/6/weights.44-0.63.hdf5
------ TRAIN ACCURACY:  weights.44-0.63.hdf5  ------
0.947184986595
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ... 102   0   5]
 [  0   0   0 ...   0 104   1]
 [  0   0   0 ...   1   0 104]]
------ TEST ACCURACY:  weights.44-0.63.hdf5  ------
0.658823529412
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 0 5]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
weights.47-0.64.hdf5
========================================
model/sisfall_raw_conv1d_10/7/weights.47-0.64.hdf5
------ TRAIN ACCURACY:  weights.47-0.64.hdf5  ------
0.943699731903
[[108   2   0 ...   0   0   0]
 [  0 109   1 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   2]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   0   0 107]]
------ TEST ACCURACY:  weights.47-0.64.hdf5  ------
0.623529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
weights.50-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/8/weights.50-0.63.hdf5
------ TRAIN ACCURACY:  weights.50-0.63.hdf5  ------
0.952010723861
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   5]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   0   0 103]]
------ TEST ACCURACY:  weights.50-0.63.hdf5  ------
0.6
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
weights.37-0.62.hdf5
========================================
model/sisfall_raw_conv1d_10/9/weights.37-0.62.hdf5
------ TRAIN ACCURACY:  weights.37-0.62.hdf5  ------
0.943967828418
[[105   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  97   2   5]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   3 100]]
------ TEST ACCURACY:  weights.37-0.62.hdf5  ------
0.670588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
weights.21-0.61.hdf5
========================================
model/sisfall_raw_conv1d_10/10/weights.21-0.61.hdf5
------ TRAIN ACCURACY:  weights.21-0.61.hdf5  ------
0.870509383378
[[ 95   3   0 ...   0   0   0]
 [  2 107   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  74   2  20]
 [  0   0   0 ...   0 107   1]
 [  0   0   0 ...   0   1 106]]
------ TEST ACCURACY:  weights.21-0.61.hdf5  ------
0.547058823529
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
weights.38-0.62.hdf5
========================================
model/sisfall_raw_conv1d_10/11/weights.38-0.62.hdf5
------ TRAIN ACCURACY:  weights.38-0.62.hdf5  ------
0.93726541555
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  95   0   8]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   0 106]]
------ TEST ACCURACY:  weights.38-0.62.hdf5  ------
0.452941176471
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
weights.48-0.64.hdf5
========================================
model/sisfall_raw_conv1d_10/12/weights.48-0.64.hdf5
------ TRAIN ACCURACY:  weights.48-0.64.hdf5  ------
0.956568364611
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   3]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   1   0 100]]
------ TEST ACCURACY:  weights.48-0.64.hdf5  ------
0.676470588235
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 4]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
weights.43-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/13/weights.43-0.63.hdf5
------ TRAIN ACCURACY:  weights.43-0.63.hdf5  ------
0.946648793566
[[105   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  99   1   2]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   2 100]]
------ TEST ACCURACY:  weights.43-0.63.hdf5  ------
0.641176470588
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
weights.38-0.60.hdf5
========================================
model/sisfall_raw_conv1d_10/14/weights.38-0.60.hdf5
------ TRAIN ACCURACY:  weights.38-0.60.hdf5  ------
0.935924932976
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  97   0   3]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   0   1 101]]
------ TEST ACCURACY:  weights.38-0.60.hdf5  ------
0.541176470588
[[4 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 2 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
weights.45-0.61.hdf5
========================================
model/sisfall_raw_conv1d_10/15/weights.45-0.61.hdf5
------ TRAIN ACCURACY:  weights.45-0.61.hdf5  ------
0.933333333333
[[107   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   4]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   0   0 103]]
------ TEST ACCURACY:  weights.45-0.61.hdf5  ------
0.448484848485
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
weights.41-0.64.hdf5
========================================
model/sisfall_raw_conv1d_10/16/weights.41-0.64.hdf5
------ TRAIN ACCURACY:  weights.41-0.64.hdf5  ------
0.952010723861
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   5]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   1   1 104]]
------ TEST ACCURACY:  weights.41-0.64.hdf5  ------
0.6
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
weights.49-0.64.hdf5
========================================
model/sisfall_raw_conv1d_10/17/weights.49-0.64.hdf5
------ TRAIN ACCURACY:  weights.49-0.64.hdf5  ------
0.952291610828
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   2]
 [  0   0   0 ...   0 107   0]
 [  0   0   0 ...   0   0 102]]
------ TEST ACCURACY:  weights.49-0.64.hdf5  ------
0.639053254438
[[1 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
weights.41-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/18/weights.41-0.63.hdf5
------ TRAIN ACCURACY:  weights.41-0.63.hdf5  ------
0.93726541555
[[104   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  96   0   7]
 [  0   0   0 ...   0 105   1]
 [  0   0   0 ...   1   0 105]]
------ TEST ACCURACY:  weights.41-0.63.hdf5  ------
0.735294117647
[[5 0 0 ... 0 0 0]
 [1 3 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
weights.49-0.62.hdf5
========================================
model/sisfall_raw_conv1d_10/19/weights.49-0.62.hdf5
------ TRAIN ACCURACY:  weights.49-0.62.hdf5  ------
0.944772117962
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ... 102   0   2]
 [  0   0   0 ...   0 106   0]
 [  0   0   0 ...   0   2 104]]
------ TEST ACCURACY:  weights.49-0.62.hdf5  ------
0.664705882353
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 3]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
weights.40-0.61.hdf5
========================================
model/sisfall_raw_conv1d_10/20/weights.40-0.61.hdf5
------ TRAIN ACCURACY:  weights.40-0.61.hdf5  ------
0.94322442421
[[106   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 100   0   5]
 [  0   0   0 ...   0 108   0]
 [  0   0   0 ...   0   1 104]]
------ TEST ACCURACY:  weights.40-0.61.hdf5  ------
0.668674698795
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 3 0 1]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
weights.47-0.61.hdf5
========================================
model/sisfall_raw_conv1d_10/21/weights.47-0.61.hdf5
------ TRAIN ACCURACY:  weights.47-0.61.hdf5  ------
0.93726541555
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   7]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   1   0 103]]
------ TEST ACCURACY:  weights.47-0.61.hdf5  ------
0.7
[[2 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
weights.36-0.67.hdf5
========================================
model/sisfall_raw_conv1d_10/22/weights.36-0.67.hdf5
------ TRAIN ACCURACY:  weights.36-0.67.hdf5  ------
0.916890080429
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  99   0   3]
 [  0   0   0 ...   0 109   0]
 [  0   0   0 ...   0   2 100]]
------ TEST ACCURACY:  weights.36-0.67.hdf5  ------
0.523529411765
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 2]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
weights.44-0.63.hdf5
========================================
model/sisfall_raw_conv1d_10/23/weights.44-0.63.hdf5
------ TRAIN ACCURACY:  weights.44-0.63.hdf5  ------
0.945576407507
[[104   2   0 ...   0   0   0]
 [  0 105   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  98   0   8]
 [  0   0   0 ...   0 105   2]
 [  0   0   0 ...   0   0 104]]
------ TEST ACCURACY:  weights.44-0.63.hdf5  ------
0.670588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 3]
 [0 0 0 ... 0 5 0]
 [0 0 0 ... 0 0 5]]
[0.6705882352941176, 0.5117647058823529, 0.6058823529411764, 0.611764705882353, 0.6529411764705882, 0.6588235294117647, 0.6235294117647059, 0.6, 0.6705882352941176, 0.5470588235294118, 0.45294117647058824, 0.6764705882352942, 0.6411764705882353, 0.5411764705882353, 0.4484848484848485, 0.6, 0.6390532544378699, 0.7352941176470589, 0.6647058823529411, 0.6686746987951807, 0.7, 0.5235294117647059, 0.6705882352941176]
0.613697231788
0.074710425358

Process finished with exit code 0

'''