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

X = pickle.load(open("data/X_sisfall_spca.p", "rb"))
y = pickle.load(open("data/y_sisfall_spca.p", "rb"))

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

    new_dir = 'model/sisfall_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_spca_conv1d_10/' + str(i+1) + '/'
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
[0.4823529411764706, 0.35294117647058826, 0.36470588235294116, 0.3941176470588235, 0.49411764705882355, 0.5058823529411764, 0.5176470588235295, 0.5058823529411764, 0.45294117647058824, 0.4, 0.4470588235294118, 0.4647058823529412, 0.5176470588235295, 0.37058823529411766, 0.4303030303030303, 0.4823529411764706, 0.5266272189349113, 0.5176470588235295, 0.5, 0.4939759036144578, 0.5176470588235295, 0.3764705882352941, 0.5117647058823529]
0.462059858308
0.056728954103
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/1/weights.18-0.43.hdf5
------ TRAIN ACCURACY:  weights.18-0.43.hdf5  ------
0.724128686327
[[109   0   0 ...   0   0   0]
 [ 11  99   0 ...   0   0   0]
 [  0   0  96 ...   0   0   0]
 ...
 [  0   0   0 ...  43   9   4]
 [  0   0   0 ...   2  85   1]
 [  0   0   0 ...   3  11  47]]
------ TEST ACCURACY:  weights.18-0.43.hdf5  ------
0.482352941176
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 2]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/2/weights.26-0.49.hdf5
------ TRAIN ACCURACY:  weights.26-0.49.hdf5  ------
0.839142091153
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  62   3   2]
 [  0   0   0 ...   2  90   1]
 [  0   0   0 ...   4   6  68]]
------ TEST ACCURACY:  weights.26-0.49.hdf5  ------
0.352941176471
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/3/weights.16-0.47.hdf5
------ TRAIN ACCURACY:  weights.16-0.47.hdf5  ------
0.762198391421
[[100   2   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   0 104 ...   0   0   0]
 ...
 [  0   0   0 ...  55   6   4]
 [  0   0   0 ...   3  84   3]
 [  0   0   0 ...   7   6  60]]
------ TEST ACCURACY:  weights.16-0.47.hdf5  ------
0.364705882353
[[0 5 0 ... 0 0 0]
 [0 4 1 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 3 0 1]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/4/weights.33-0.45.hdf5
------ TRAIN ACCURACY:  weights.33-0.45.hdf5  ------
0.826273458445
[[109   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  87   2   1]
 [  0   0   0 ...   3  94   0]
 [  0   0   0 ...  13  12  62]]
------ TEST ACCURACY:  weights.33-0.45.hdf5  ------
0.394117647059
[[5 0 0 ... 0 0 0]
 [2 2 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 1 2 0]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/5/weights.30-0.46.hdf5
------ TRAIN ACCURACY:  weights.30-0.46.hdf5  ------
0.849865951743
[[102   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  73   3   2]
 [  0   0   0 ...   2  90   3]
 [  0   0   0 ...   8   4  64]]
------ TEST ACCURACY:  weights.30-0.46.hdf5  ------
0.494117647059
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 1 2]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/6/weights.41-0.45.hdf5
------ TRAIN ACCURACY:  weights.41-0.45.hdf5  ------
0.877479892761
[[105   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  80   1   3]
 [  0   0   0 ...   0  88   2]
 [  0   0   0 ...   5   3  76]]
------ TEST ACCURACY:  weights.41-0.45.hdf5  ------
0.505882352941
[[3 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 2]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/7/weights.22-0.47.hdf5
------ TRAIN ACCURACY:  weights.22-0.47.hdf5  ------
0.778820375335
[[102   5   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  56  14  14]
 [  0   0   0 ...   1  98   3]
 [  0   0   0 ...   6  14  80]]
------ TEST ACCURACY:  weights.22-0.47.hdf5  ------
0.517647058824
[[4 1 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 3 0]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/8/weights.37-0.46.hdf5
------ TRAIN ACCURACY:  weights.37-0.46.hdf5  ------
0.868096514745
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   1 106 ...   0   0   0]
 ...
 [  0   0   0 ...  80   0   0]
 [  0   0   0 ...   1  93   2]
 [  0   0   0 ...   3   2  77]]
------ TEST ACCURACY:  weights.37-0.46.hdf5  ------
0.505882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/9/weights.50-0.44.hdf5
------ TRAIN ACCURACY:  weights.50-0.44.hdf5  ------
0.882841823056
[[105   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  91   0   0]
 [  0   0   0 ...   0  94   1]
 [  0   0   0 ...   9   3  58]]
------ TEST ACCURACY:  weights.50-0.44.hdf5  ------
0.452941176471
[[4 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 1 1]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/10/weights.37-0.46.hdf5
------ TRAIN ACCURACY:  weights.37-0.46.hdf5  ------
0.85308310992
[[109   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  76   0   7]
 [  0   0   0 ...   0  82   9]
 [  0   0   0 ...   2   1  85]]
------ TEST ACCURACY:  weights.37-0.46.hdf5  ------
0.4
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 2 0 1]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/11/weights.18-0.45.hdf5
------ TRAIN ACCURACY:  weights.18-0.45.hdf5  ------
0.758445040214
[[110   0   0 ...   0   0   0]
 [  6 104   0 ...   0   0   0]
 [  0   0 103 ...   0   0   0]
 ...
 [  0   0   0 ...  40   4   5]
 [  0   0   0 ...   0  80   1]
 [  0   0   0 ...   3   3  60]]
------ TEST ACCURACY:  weights.18-0.45.hdf5  ------
0.447058823529
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 3 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/12/weights.28-0.47.hdf5
------ TRAIN ACCURACY:  weights.28-0.47.hdf5  ------
0.831903485255
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  65   2   5]
 [  0   0   0 ...   0  89   4]
 [  0   0   0 ...   3   2  78]]
------ TEST ACCURACY:  weights.28-0.47.hdf5  ------
0.464705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/13/weights.31-0.46.hdf5
------ TRAIN ACCURACY:  weights.31-0.46.hdf5  ------
0.860321715818
[[108   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  67   1   9]
 [  0   0   0 ...   0  88  10]
 [  0   0   0 ...   0   3  85]]
------ TEST ACCURACY:  weights.31-0.46.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 3]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/14/weights.44-0.45.hdf5
------ TRAIN ACCURACY:  weights.44-0.45.hdf5  ------
0.856836461126
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  70   1   2]
 [  0   0   0 ...   0  94   3]
 [  0   0   0 ...   3   2  85]]
------ TEST ACCURACY:  weights.44-0.45.hdf5  ------
0.370588235294
[[4 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/15/weights.33-0.46.hdf5
------ TRAIN ACCURACY:  weights.33-0.46.hdf5  ------
0.831860776439
[[106   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  73   0   4]
 [  0   0   0 ...   2  87   8]
 [  0   0   0 ...   6   4  80]]
------ TEST ACCURACY:  weights.33-0.46.hdf5  ------
0.430303030303
[[2 3 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/16/weights.18-0.45.hdf5
------ TRAIN ACCURACY:  weights.18-0.45.hdf5  ------
0.703753351206
[[102   3   0 ...   0   0   0]
 [  0 109   0 ...   0   0   0]
 [  0   1 105 ...   0   0   0]
 ...
 [  0   0   0 ...  45   0   3]
 [  0   0   0 ...   2  30   4]
 [  0   0   0 ...   8   1  31]]
------ TEST ACCURACY:  weights.18-0.45.hdf5  ------
0.482352941176
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 2 0 0]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/17/weights.48-0.44.hdf5
------ TRAIN ACCURACY:  weights.48-0.44.hdf5  ------
0.903243098365
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  72   6   2]
 [  0   0   0 ...   0  98   4]
 [  0   0   0 ...   0   8  83]]
------ TEST ACCURACY:  weights.48-0.44.hdf5  ------
0.526627218935
[[0 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 1 1]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/18/weights.38-0.48.hdf5
------ TRAIN ACCURACY:  weights.38-0.48.hdf5  ------
0.869973190349
[[106   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  53  11   3]
 [  0   0   0 ...   0  98   1]
 [  0   0   0 ...   2  13  73]]
------ TEST ACCURACY:  weights.38-0.48.hdf5  ------
0.517647058824
[[5 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/19/weights.14-0.46.hdf5
------ TRAIN ACCURACY:  weights.14-0.46.hdf5  ------
0.707238605898
[[102   1   0 ...   0   0   0]
 [  2 108   0 ...   0   0   0]
 [  0   0  97 ...   0   0   0]
 ...
 [  0   0   0 ...  45   5   8]
 [  0   0   0 ...   3  72   8]
 [  0   0   0 ...   4   4  67]]
------ TEST ACCURACY:  weights.14-0.46.hdf5  ------
0.5
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/20/weights.48-0.47.hdf5
------ TRAIN ACCURACY:  weights.48-0.47.hdf5  ------
0.910551687199
[[104   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  84   4   3]
 [  0   0   0 ...   0 102   2]
 [  0   0   0 ...   4   5  86]]
------ TEST ACCURACY:  weights.48-0.47.hdf5  ------
0.493975903614
[[4 0 1 ... 0 0 0]
 [0 4 1 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 4 0]
 [0 0 0 ... 0 2 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/21/weights.35-0.45.hdf5
------ TRAIN ACCURACY:  weights.35-0.45.hdf5  ------
0.863538873995
[[110   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 101 ...   0   0   0]
 ...
 [  0   0   0 ...  85   0   5]
 [  0   0   0 ...   2  97   4]
 [  0   0   0 ...   4   4  88]]
------ TEST ACCURACY:  weights.35-0.45.hdf5  ------
0.517647058824
[[3 0 0 ... 0 0 0]
 [4 1 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 4 1]
 [0 0 0 ... 1 3 0]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/22/weights.23-0.49.hdf5
------ TRAIN ACCURACY:  weights.23-0.49.hdf5  ------
0.781501340483
[[107   0   0 ...   0   0   0]
 [  4 106   0 ...   0   0   0]
 [  0   0 109 ...   0   0   0]
 ...
 [  0   0   0 ...  49  10   6]
 [  0   0   0 ...   0  92   1]
 [  0   0   0 ...   3  15  55]]
------ TEST ACCURACY:  weights.23-0.49.hdf5  ------
0.376470588235
[[4 1 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_spca_conv1d_10/23/weights.47-0.49.hdf5
------ TRAIN ACCURACY:  weights.47-0.49.hdf5  ------
0.889544235925
[[107   1   0 ...   0   0   0]
 [  2 108   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  87   0  14]
 [  0   0   0 ...   4  95   4]
 [  0   0   0 ...   3   4  95]]
------ TEST ACCURACY:  weights.47-0.49.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 0 4]]
[0.4823529411764706, 0.35294117647058826, 0.36470588235294116, 0.3941176470588235, 0.49411764705882355, 0.5058823529411764, 0.5176470588235295, 0.5058823529411764, 0.45294117647058824, 0.4, 0.4470588235294118, 0.4647058823529412, 0.5176470588235295, 0.37058823529411766, 0.4303030303030303, 0.4823529411764706, 0.5266272189349113, 0.5176470588235295, 0.5, 0.4939759036144578, 0.5176470588235295, 0.3764705882352941, 0.5117647058823529]
0.462059858308
0.056728954103

Process finished with exit code 0

'''