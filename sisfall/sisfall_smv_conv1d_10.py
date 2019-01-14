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

X = pickle.load(open("data/X_sisfall_smv.p", "rb"))
y = pickle.load(open("data/y_sisfall_smv.p", "rb"))

n_classes = 34
signal_rows = 450
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
    print X_train.shape # (3730, 450)

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

    new_dir = 'model/sisfall_smv_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_smv_conv1d_10/' + str(i+1) + '/'
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
[0.5470588235294118, 0.4117647058823529, 0.4764705882352941, 0.5235294117647059, 0.40588235294117647, 0.4235294117647059, 0.5235294117647059, 0.49411764705882355, 0.37058823529411766, 0.5705882352941176, 0.4588235294117647, 0.4411764705882353, 0.5411764705882353, 0.45294117647058824, 0.4303030303030303, 0.45294117647058824, 0.44970414201183434, 0.40588235294117647, 0.5176470588235295, 0.463855421686747, 0.5411764705882353, 0.4176470588235294, 0.4647058823529412]
0.468914741939
0.0530682597949
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_smv_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
weights.31-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/1/weights.31-0.46.hdf5
------ TRAIN ACCURACY:  weights.31-0.46.hdf5  ------
0.742895442359
[[ 97   0   0 ...   0   0   0]
 [  3 107   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  39   3  11]
 [  0   0   0 ...   0  79   2]
 [  0   0   0 ...   0   2  63]]
------ TEST ACCURACY:  weights.31-0.46.hdf5  ------
0.547058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
weights.18-0.48.hdf5
========================================
model/sisfall_smv_conv1d_10/2/weights.18-0.48.hdf5
------ TRAIN ACCURACY:  weights.18-0.48.hdf5  ------
0.755764075067
[[ 96   3   0 ...   1   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  92 ...   0   0   0]
 ...
 [  0   0   0 ...  73   0   4]
 [  0   0   0 ...   2  75   8]
 [  0   0   0 ...   6   6  59]]
------ TEST ACCURACY:  weights.18-0.48.hdf5  ------
0.411764705882
[[0 5 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
weights.16-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/3/weights.16-0.46.hdf5
------ TRAIN ACCURACY:  weights.16-0.46.hdf5  ------
0.724396782842
[[ 93   6   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  45   0  16]
 [  0   0   0 ...   1  72   7]
 [  0   0   0 ...   4   2  66]]
------ TEST ACCURACY:  weights.16-0.46.hdf5  ------
0.476470588235
[[4 1 0 ... 0 0 0]
 [0 2 3 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
weights.18-0.48.hdf5
========================================
model/sisfall_smv_conv1d_10/4/weights.18-0.48.hdf5
------ TRAIN ACCURACY:  weights.18-0.48.hdf5  ------
0.72144772118
[[ 96   1   0 ...   0   0   0]
 [  5 105   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  46   1   3]
 [  0   0   0 ...   0  59   4]
 [  0   0   0 ...   2   2  45]]
------ TEST ACCURACY:  weights.18-0.48.hdf5  ------
0.523529411765
[[0 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
weights.47-0.48.hdf5
========================================
model/sisfall_smv_conv1d_10/5/weights.47-0.48.hdf5
------ TRAIN ACCURACY:  weights.47-0.48.hdf5  ------
0.793565683646
[[ 79   0   0 ...   8   0   0]
 [  2 102   0 ...   1   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ... 102   2   2]
 [  0   0   0 ...   3  99   1]
 [  0   0   0 ...   6   6  82]]
------ TEST ACCURACY:  weights.47-0.48.hdf5  ------
0.405882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 4 1]
 [0 0 0 ... 0 5 0]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
weights.13-0.44.hdf5
========================================
model/sisfall_smv_conv1d_10/6/weights.13-0.44.hdf5
------ TRAIN ACCURACY:  weights.13-0.44.hdf5  ------
0.640750670241
[[ 21  15   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  97 ...   0   0   0]
 ...
 [  0   0   0 ...  39   4   1]
 [  0   0   0 ...   0  90   0]
 [  0   0   0 ...   8  14  14]]
------ TEST ACCURACY:  weights.13-0.44.hdf5  ------
0.423529411765
[[0 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
weights.21-0.48.hdf5
========================================
model/sisfall_smv_conv1d_10/7/weights.21-0.48.hdf5
------ TRAIN ACCURACY:  weights.21-0.48.hdf5  ------
0.727077747989
[[ 52  21   0 ...   1   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  59   0   3]
 [  0   0   0 ...   5  73   5]
 [  0   0   0 ...   3   0  63]]
------ TEST ACCURACY:  weights.21-0.48.hdf5  ------
0.523529411765
[[1 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 4 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
weights.49-0.47.hdf5
========================================
model/sisfall_smv_conv1d_10/8/weights.49-0.47.hdf5
------ TRAIN ACCURACY:  weights.49-0.47.hdf5  ------
0.820643431635
[[ 91  18   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  96   1   2]
 [  0   0   0 ...   2  96   0]
 [  0   0   0 ...   4   8  76]]
------ TEST ACCURACY:  weights.49-0.47.hdf5  ------
0.494117647059
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
weights.08-0.42.hdf5
========================================
model/sisfall_smv_conv1d_10/9/weights.08-0.42.hdf5
------ TRAIN ACCURACY:  weights.08-0.42.hdf5  ------
0.595978552279
[[95  3  0 ...  0  0  0]
 [11 99  0 ...  0  0  0]
 [ 0  0 89 ...  0  0  0]
 ...
 [ 9  0  0 ... 20  0  1]
 [ 0  0  0 ...  1 57  1]
 [ 5  0  0 ...  2  0 13]]
------ TEST ACCURACY:  weights.08-0.42.hdf5  ------
0.370588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
weights.30-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/10/weights.30-0.46.hdf5
------ TRAIN ACCURACY:  weights.30-0.46.hdf5  ------
0.745576407507
[[ 74   8   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  87   5  10]
 [  0   0   0 ...   2  95   2]
 [  0   0   0 ...   4  14  79]]
------ TEST ACCURACY:  weights.30-0.46.hdf5  ------
0.570588235294
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
weights.28-0.47.hdf5
========================================
model/sisfall_smv_conv1d_10/11/weights.28-0.47.hdf5
------ TRAIN ACCURACY:  weights.28-0.47.hdf5  ------
0.758445040214
[[104   0   0 ...   0   0   0]
 [ 11  99   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  53   0   4]
 [  0   0   0 ...   0  69   2]
 [  0   0   0 ...   1   1  67]]
------ TEST ACCURACY:  weights.28-0.47.hdf5  ------
0.458823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
weights.39-0.51.hdf5
========================================
model/sisfall_smv_conv1d_10/12/weights.39-0.51.hdf5
------ TRAIN ACCURACY:  weights.39-0.51.hdf5  ------
0.816085790885
[[ 73   3   0 ...   1   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ... 101   0   2]
 [  0   0   0 ...   4  95   2]
 [  0   0   0 ...   9  10  73]]
------ TEST ACCURACY:  weights.39-0.51.hdf5  ------
0.441176470588
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 2 1]
 [0 0 0 ... 0 4 1]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
weights.15-0.44.hdf5
========================================
model/sisfall_smv_conv1d_10/13/weights.15-0.44.hdf5
------ TRAIN ACCURACY:  weights.15-0.44.hdf5  ------
0.68471849866
[[ 99   0   0 ...   0   0   0]
 [ 19  84   5 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  75   0   6]
 [  0   0   0 ...  10  49  15]
 [  0   0   0 ...   9   0  65]]
------ TEST ACCURACY:  weights.15-0.44.hdf5  ------
0.541176470588
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
weights.29-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/14/weights.29-0.46.hdf5
------ TRAIN ACCURACY:  weights.29-0.46.hdf5  ------
0.72654155496
[[ 88   0   0 ...   0   0   0]
 [  1 109   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  46   4   5]
 [  0   0   0 ...   1  98   0]
 [  0   0   0 ...   0  14  61]]
------ TEST ACCURACY:  weights.29-0.46.hdf5  ------
0.452941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
weights.37-0.45.hdf5
========================================
model/sisfall_smv_conv1d_10/15/weights.37-0.45.hdf5
------ TRAIN ACCURACY:  weights.37-0.45.hdf5  ------
0.745381526104
[[ 93   1   0 ...   1   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  97   1   1]
 [  0   0   0 ...  11  77   4]
 [  0   0   0 ...  11   0  65]]
------ TEST ACCURACY:  weights.37-0.45.hdf5  ------
0.430303030303
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 0]
 [0 0 0 ... 5 0 0]
 [0 0 0 ... 3 0 2]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
weights.38-0.52.hdf5
========================================
model/sisfall_smv_conv1d_10/16/weights.38-0.52.hdf5
------ TRAIN ACCURACY:  weights.38-0.52.hdf5  ------
0.797587131367
[[ 85   9   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  68   0   1]
 [  0   0   0 ...   2  79   0]
 [  0   0   0 ...   4   5  54]]
------ TEST ACCURACY:  weights.38-0.52.hdf5  ------
0.452941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
weights.21-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/17/weights.21-0.46.hdf5
------ TRAIN ACCURACY:  weights.21-0.46.hdf5  ------
0.659072634682
[[ 98   0   0 ...   0   0   0]
 [  5 105   0 ...   0   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ...  62  23   1]
 [  0   0   0 ...   0 105   0]
 [  0   0   0 ...   4  60  26]]
------ TEST ACCURACY:  weights.21-0.46.hdf5  ------
0.449704142012
[[0 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 2 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 3 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
weights.39-0.44.hdf5
========================================
model/sisfall_smv_conv1d_10/18/weights.39-0.44.hdf5
------ TRAIN ACCURACY:  weights.39-0.44.hdf5  ------
0.652815013405
[[ 44   0   0 ...   2   1   0]
 [  3  90   0 ...   0   1   0]
 [  0   0 102 ...   1   0   0]
 ...
 [  0   0   0 ...  61   6   0]
 [  0   0   0 ...   0  88   0]
 [  0   0   0 ...   1  12  47]]
------ TEST ACCURACY:  weights.39-0.44.hdf5  ------
0.405882352941
[[2 0 0 ... 0 0 0]
 [2 0 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
weights.45-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/19/weights.45-0.46.hdf5
------ TRAIN ACCURACY:  weights.45-0.46.hdf5  ------
0.83672922252
[[107   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  65   2   3]
 [  0   0   0 ...   1  97   1]
 [  0   0   0 ...   1   6  82]]
------ TEST ACCURACY:  weights.45-0.46.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
weights.19-0.42.hdf5
========================================
model/sisfall_smv_conv1d_10/20/weights.19-0.42.hdf5
------ TRAIN ACCURACY:  weights.19-0.42.hdf5  ------
0.694429566149
[[ 90   2   0 ...   0   0   0]
 [  3 107   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  26   0  31]
 [  0   0   0 ...   0  67  17]
 [  0   0   0 ...   1   4  81]]
------ TEST ACCURACY:  weights.19-0.42.hdf5  ------
0.463855421687
[[1 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 5]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
weights.21-0.49.hdf5
========================================
model/sisfall_smv_conv1d_10/21/weights.21-0.49.hdf5
------ TRAIN ACCURACY:  weights.21-0.49.hdf5  ------
0.751206434316
[[ 92   2   0 ...   2   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0  98 ...   0   0   0]
 ...
 [  0   0   0 ...  89   0   5]
 [  0   0   0 ...  14  52  20]
 [  0   0   0 ...  17   1  73]]
------ TEST ACCURACY:  weights.21-0.49.hdf5  ------
0.541176470588
[[1 0 0 ... 0 0 0]
 [5 0 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 2]
 [0 0 0 ... 2 2 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
weights.35-0.44.hdf5
========================================
model/sisfall_smv_conv1d_10/22/weights.35-0.44.hdf5
------ TRAIN ACCURACY:  weights.35-0.44.hdf5  ------
0.68471849866
[[ 33  22   0 ...   1   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  38   9   4]
 [  0   0   0 ...   1 101   0]
 [  0   0   0 ...   0  13  60]]
------ TEST ACCURACY:  weights.35-0.44.hdf5  ------
0.417647058824
[[0 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 2 1]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
weights.26-0.46.hdf5
========================================
model/sisfall_smv_conv1d_10/23/weights.26-0.46.hdf5
------ TRAIN ACCURACY:  weights.26-0.46.hdf5  ------
0.676943699732
[[76  0  0 ...  0  0  0]
 [ 8 99  0 ...  0  0  0]
 [ 0  0 99 ...  0  0  0]
 ...
 [ 0  0  0 ... 31 10  1]
 [ 0  0  0 ...  0 85  0]
 [ 0  0  0 ...  0 17 22]]
------ TEST ACCURACY:  weights.26-0.46.hdf5  ------
0.464705882353
[[1 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]]
[0.5470588235294118, 0.4117647058823529, 0.4764705882352941, 0.5235294117647059, 0.40588235294117647, 0.4235294117647059, 0.5235294117647059, 0.49411764705882355, 0.37058823529411766, 0.5705882352941176, 0.4588235294117647, 0.4411764705882353, 0.5411764705882353, 0.45294117647058824, 0.4303030303030303, 0.45294117647058824, 0.44970414201183434, 0.40588235294117647, 0.5176470588235295, 0.463855421686747, 0.5411764705882353, 0.4176470588235294, 0.4647058823529412]
0.468914741939
0.0530682597949

Process finished with exit code 0

'''
