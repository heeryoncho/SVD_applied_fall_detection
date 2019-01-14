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

X = pickle.load(open("data/X_sisfall_kpca.p", "rb"))
y = pickle.load(open("data/y_sisfall_kpca.p", "rb"))

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

    new_dir = 'model/sisfall_kpca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/sisfall_kpca_conv1d_10/' + str(i+1) + '/'
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
[0.4647058823529412, 0.4176470588235294, 0.4176470588235294, 0.4764705882352941, 0.5058823529411764, 0.4764705882352941, 0.4764705882352941, 0.4647058823529412, 0.5588235294117647, 0.4294117647058823, 0.40588235294117647, 0.5058823529411764, 0.4823529411764706, 0.4235294117647059, 0.5515151515151515, 0.48823529411764705, 0.4970414201183432, 0.5470588235294118, 0.5117647058823529, 0.5060240963855421, 0.5176470588235295, 0.45294117647058824, 0.5352941176470588]
0.483191486845
0.0435936790659
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/sisfall/sisfall_kpca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/1/weights.34-0.49.hdf5
------ TRAIN ACCURACY:  weights.34-0.49.hdf5  ------
0.876943699732
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  60   3  10]
 [  0   0   0 ...   1  86  11]
 [  0   0   0 ...   1   3  93]]
------ TEST ACCURACY:  weights.34-0.49.hdf5  ------
0.464705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 1 0]
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 2 1]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/2/weights.47-0.52.hdf5
------ TRAIN ACCURACY:  weights.47-0.52.hdf5  ------
0.91018766756
[[104   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  83   0   2]
 [  0   0   0 ...   4  85   1]
 [  0   0   0 ...   4   1  79]]
------ TEST ACCURACY:  weights.47-0.52.hdf5  ------
0.417647058824
[[1 4 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/3/weights.46-0.50.hdf5
------ TRAIN ACCURACY:  weights.46-0.50.hdf5  ------
0.906702412869
[[102   8   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  66   3   6]
 [  0   0   0 ...   1  92   2]
 [  0   0   0 ...   0   1  97]]
------ TEST ACCURACY:  weights.46-0.50.hdf5  ------
0.417647058824
[[0 5 0 ... 0 0 0]
 [0 3 1 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 1 0 2]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/4/weights.34-0.50.hdf5
------ TRAIN ACCURACY:  weights.34-0.50.hdf5  ------
0.859785522788
[[109   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  1   1 105 ...   0   0   0]
 ...
 [  0   0   0 ...  69   1   6]
 [  0   0   0 ...   5  77   3]
 [  0   0   0 ...   4   0  81]]
------ TEST ACCURACY:  weights.34-0.50.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 1 0 0]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/5/weights.50-0.51.hdf5
------ TRAIN ACCURACY:  weights.50-0.51.hdf5  ------
0.920107238606
[[108   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  89   1   2]
 [  0   0   0 ...   7  90   4]
 [  0   0   0 ...   3   1  91]]
------ TEST ACCURACY:  weights.50-0.51.hdf5  ------
0.505882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 1 2 1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/6/weights.26-0.49.hdf5
------ TRAIN ACCURACY:  weights.26-0.49.hdf5  ------
0.841554959786
[[107   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  1   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  65   2   2]
 [  0   0   0 ...   7  81   3]
 [  0   0   0 ...   9   3  60]]
------ TEST ACCURACY:  weights.26-0.49.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/7/weights.48-0.49.hdf5
------ TRAIN ACCURACY:  weights.48-0.49.hdf5  ------
0.901608579088
[[107   2   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  80   3   8]
 [  0   0   0 ...   3  94   3]
 [  0   0   0 ...   0   5  95]]
------ TEST ACCURACY:  weights.48-0.49.hdf5  ------
0.476470588235
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/8/weights.44-0.51.hdf5
------ TRAIN ACCURACY:  weights.44-0.51.hdf5  ------
0.898123324397
[[104   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  78   2   4]
 [  0   0   0 ...   4  88   9]
 [  0   0   0 ...   1   1  98]]
------ TEST ACCURACY:  weights.44-0.51.hdf5  ------
0.464705882353
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/9/weights.38-0.50.hdf5
------ TRAIN ACCURACY:  weights.38-0.50.hdf5  ------
0.892225201072
[[105   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 100 ...   0   0   0]
 ...
 [  0   0   0 ...  81   0   5]
 [  0   0   0 ...   5  76   5]
 [  0   0   0 ...   4   1  88]]
------ TEST ACCURACY:  weights.38-0.50.hdf5  ------
0.558823529412
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 1 1 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/10/weights.47-0.49.hdf5
------ TRAIN ACCURACY:  weights.47-0.49.hdf5  ------
0.905898123324
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  77   4   1]
 [  0   0   0 ...   2  89   2]
 [  0   0   0 ...   0   4  91]]
------ TEST ACCURACY:  weights.47-0.49.hdf5  ------
0.429411764706
[[5 0 0 ... 0 0 0]
 [0 4 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 2 0]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/11/weights.47-0.49.hdf5
------ TRAIN ACCURACY:  weights.47-0.49.hdf5  ------
0.906970509383
[[109   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  84   1   6]
 [  0   0   0 ...   5  88   4]
 [  0   0   0 ...   3   0  95]]
------ TEST ACCURACY:  weights.47-0.49.hdf5  ------
0.405882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 2 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 3 0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/12/weights.50-0.50.hdf5
------ TRAIN ACCURACY:  weights.50-0.50.hdf5  ------
0.917426273458
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  99   0   1]
 [  0   0   0 ...  10  84   4]
 [  0   0   0 ...  10   0  86]]
------ TEST ACCURACY:  weights.50-0.50.hdf5  ------
0.505882352941
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 0 2 2]
 [0 0 0 ... 1 0 1]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/13/weights.50-0.50.hdf5
------ TRAIN ACCURACY:  weights.50-0.50.hdf5  ------
0.919839142091
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  91   1   2]
 [  0   0   0 ...   8  86   5]
 [  0   0   0 ...   2   2  95]]
------ TEST ACCURACY:  weights.50-0.50.hdf5  ------
0.482352941176
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 1 1 2]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 1 2]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/14/weights.32-0.50.hdf5
------ TRAIN ACCURACY:  weights.32-0.50.hdf5  ------
0.879892761394
[[105   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 102 ...   0   0   0]
 ...
 [  0   0   0 ...  76   4   6]
 [  0   0   0 ...   5  88   5]
 [  0   0   0 ...   9   3  80]]
------ TEST ACCURACY:  weights.32-0.50.hdf5  ------
0.423529411765
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/15/weights.36-0.52.hdf5
------ TRAIN ACCURACY:  weights.36-0.52.hdf5  ------
0.859705488621
[[104   4   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  69   1   6]
 [  0   0   0 ...   4  76   8]
 [  0   0   0 ...   7   0  84]]
------ TEST ACCURACY:  weights.36-0.52.hdf5  ------
0.551515151515
[[3 2 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/16/weights.30-0.49.hdf5
------ TRAIN ACCURACY:  weights.30-0.49.hdf5  ------
0.85764075067
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  70   1   5]
 [  0   0   0 ...   7  70   8]
 [  0   0   0 ...   7   1  81]]
------ TEST ACCURACY:  weights.30-0.49.hdf5  ------
0.488235294118
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 2]
 [0 0 0 ... 1 0 1]
 [0 0 0 ... 0 0 1]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/17/weights.49-0.49.hdf5
------ TRAIN ACCURACY:  weights.49-0.49.hdf5  ------
0.905655320289
[[109   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  81   2   5]
 [  0   0   0 ...   4  96   3]
 [  0   0   0 ...   1   1  91]]
------ TEST ACCURACY:  weights.49-0.49.hdf5  ------
0.497041420118
[[0 0 0 ... 0 0 0]
 [3 2 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 2]
 [0 0 0 ... 0 1 0]
 [0 0 0 ... 1 3 0]]

>>>>>>>>>>>>>> 18 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/18/weights.43-0.49.hdf5
------ TRAIN ACCURACY:  weights.43-0.49.hdf5  ------
0.89490616622
[[107   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 105 ...   0   0   0]
 ...
 [  0   0   0 ...  94   0   3]
 [  0   0   0 ...   8  85   7]
 [  0   0   0 ...   8   1  91]]
------ TEST ACCURACY:  weights.43-0.49.hdf5  ------
0.547058823529
[[5 0 0 ... 0 0 0]
 [2 3 0 ... 0 0 0]
 [0 0 4 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 1]
 [0 0 0 ... 0 1 2]
 [0 0 0 ... 2 0 2]]

>>>>>>>>>>>>>> 19 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/19/weights.41-0.50.hdf5
------ TRAIN ACCURACY:  weights.41-0.50.hdf5  ------
0.89490616622
[[106   3   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  1   0 107 ...   0   0   0]
 ...
 [  0   0   0 ...  67   1   4]
 [  0   0   0 ...   1  87   3]
 [  0   0   0 ...   0   2  84]]
------ TEST ACCURACY:  weights.41-0.50.hdf5  ------
0.511764705882
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 1]
 [0 0 0 ... 3 0 0]]

>>>>>>>>>>>>>> 20 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/20/weights.37-0.51.hdf5
------ TRAIN ACCURACY:  weights.37-0.51.hdf5  ------
0.884574183182
[[108   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 108 ...   0   0   0]
 ...
 [  0   0   0 ...  81   0   6]
 [  0   0   0 ...  10  74   8]
 [  0   0   0 ...   4   0  95]]
------ TEST ACCURACY:  weights.37-0.51.hdf5  ------
0.506024096386
[[5 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 3 0]
 [0 0 0 ... 0 0 0]]

>>>>>>>>>>>>>> 21 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/21/weights.45-0.51.hdf5
------ TRAIN ACCURACY:  weights.45-0.51.hdf5  ------
0.902680965147
[[106   1   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  1   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  85   0   4]
 [  0   0   0 ...   3  86   7]
 [  0   0   0 ...   4   1  92]]
------ TEST ACCURACY:  weights.45-0.51.hdf5  ------
0.517647058824
[[1 0 0 ... 0 0 0]
 [1 4 0 ... 0 0 0]
 [0 0 3 ... 0 0 0]
 ...
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 0 0]
 [0 0 0 ... 1 1 0]]

>>>>>>>>>>>>>> 22 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/22/weights.34-0.54.hdf5
------ TRAIN ACCURACY:  weights.34-0.54.hdf5  ------
0.879892761394
[[106   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 110 ...   0   0   0]
 ...
 [  0   0   0 ...  82   3   3]
 [  0   0   0 ...   5  92   1]
 [  0   0   0 ...  10   8  74]]
------ TEST ACCURACY:  weights.34-0.54.hdf5  ------
0.452941176471
[[5 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 1 ... 0 0 0]
 ...
 [0 0 0 ... 3 0 0]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 0 0 2]]

>>>>>>>>>>>>>> 23 -fold <<<<<<<<<<<<<<<<
========================================
model/sisfall_kpca_conv1d_10/23/weights.44-0.48.hdf5
------ TRAIN ACCURACY:  weights.44-0.48.hdf5  ------
0.904557640751
[[106   0   0 ...   0   0   0]
 [  0 110   0 ...   0   0   0]
 [  0   0 106 ...   0   0   0]
 ...
 [  0   0   0 ...  87   1   9]
 [  0   0   0 ...   3  91   6]
 [  0   0   0 ...   5   3  96]]
------ TEST ACCURACY:  weights.44-0.48.hdf5  ------
0.535294117647
[[2 0 0 ... 0 0 0]
 [0 5 0 ... 0 0 0]
 [0 0 5 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 2 0 1]
 [0 0 0 ... 1 0 2]]
[0.4647058823529412, 0.4176470588235294, 0.4176470588235294, 0.4764705882352941, 0.5058823529411764, 0.4764705882352941, 0.4764705882352941, 0.4647058823529412, 0.5588235294117647, 0.4294117647058823, 0.40588235294117647, 0.5058823529411764, 0.4823529411764706, 0.4235294117647059, 0.5515151515151515, 0.48823529411764705, 0.4970414201183432, 0.5470588235294118, 0.5117647058823529, 0.5060240963855421, 0.5176470588235295, 0.45294117647058824, 0.5352941176470588]
0.483191486845
0.0435936790659

Process finished with exit code 0


'''