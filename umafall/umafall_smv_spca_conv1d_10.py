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

X_spca = pickle.load(open("data/X_umafall_spca.p", "rb"))
X_smv = pickle.load(open("data/X_umafall_smv.p", "rb"))
X = np.concatenate((X_spca, X_smv), axis=1)

y = pickle.load(open("data/y_umafall_spca.p", "rb"))

n_classes = 11
signal_rows = 450 * 2
signal_columns = 1
n_subject = 17


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
    print X_train.shape # (2465, 900)

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

    new_dir = 'model/umafall_smv_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_smv_spca_conv1d_10/' + str(i+1) + '/'
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
[0.5894736842105263, 0.5533333333333333, 0.5567567567567567, 0.5, 0.6285714285714286, 0.4, 0.7, 0.6210526315789474, 0.5388888888888889, 0.37142857142857144, 0.55, 0.425, 0.4421052631578947, 0.4909090909090909, 0.47619047619047616, 0.48833333333333334, 0.49444444444444446]
0.519205170753
0.083779009663
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_smv_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/1/weights.50-0.41.hdf5
------ TRAIN ACCURACY:  weights.50-0.41.hdf5  ------
0.758620689655
[[236   2   0   0   0   0   0   0   0   2   0]
 [  0 122   0   2   0   0   0   0   0   1   0]
 [  0   0  75   0   0   1  25   6  29 108   6]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  89   0   0   0   2   6   3]
 [  0   0   0   0   0  95   0   0   2   2   1]
 [  2   0   1   0   0   0 224   1   4   7   1]
 [  2   0   2   0   0   0  54 101  15  66   5]
 [  0   0   0   0   0   0   0   0 173  97  20]
 [  0   0   0   0   0   0   3   0  11 298   8]
 [  2   0   0   0   0   0   0   0  33  62 238]]
------ TEST ACCURACY:  weights.50-0.41.hdf5  ------
0.589473684211
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  3  0  0  0  0  0  0]
 [ 0  0  2  0  0  5  3  1  4  0]
 [ 0  0  0 14  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 11  3  0  1  0]
 [ 0  0  0  0  0  3  7  2  2  1]
 [ 0  0  0  0  0  0  0  7 16  7]
 [ 0  0  0  0  0  0  0  5 33  2]
 [ 0  0  0  0  0  0  0  8 11 11]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/2/weights.46-0.44.hdf5
------ TRAIN ACCURACY:  weights.46-0.44.hdf5  ------
0.760079840319
[[235   0   0   0   0   0   2   0   0   3   0]
 [  0 121   0   3   0   0   0   0   0   1   0]
 [  0   0 112   0   0   0   2   8  50  65  13]
 [  0   0   0 217   0   0   0   0   2   1   0]
 [  0   0   0   0  90   0   0   0   4   3   3]
 [  0   0   0   0   0  91   0   0   1   4   4]
 [  0   0   1   0   0   0 188   8  13  20  10]
 [  0   0  11   0   0   0  19 112  54  37  12]
 [  0   0   0   0   0   0   0   0 201  61  43]
 [  0   0   0   0   0   0   1   0  32 273  24]
 [  0   0   0   0   0   0   0   0  39  47 264]]
------ TEST ACCURACY:  weights.46-0.44.hdf5  ------
0.553333333333
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0  1  1  5  6  1]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  1  0  8  3  0  0  2]
 [ 0  0  5  0  0  0  1  0  8  1  0]
 [ 0  0  0  0  0  0  0  0  8  2  5]
 [ 0  0  0  0  0  1  0  0  9 13  7]
 [ 0  0  0  0  0  0  0  0  3  4  8]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/3/weights.35-0.41.hdf5
------ TRAIN ACCURACY:  weights.35-0.41.hdf5  ------
0.770850202429
[[234   1   0   0   0   0   2   0   0   3   0]
 [  0 125   0   0   0   0   0   0   0   0   0]
 [  0   0 222   0   0   0   0   1   5  16   1]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  0   0   1   0  89   0   0   0   1   8   1]
 [  0   0   0   0   1  89   0   0   2   3   5]
 [  0   0  36   0   0   0 188   9   1   4   2]
 [  0   0 119   0   0   0  30  88   1   6   1]
 [  0   0  55   0   0   1   0   0 129  59  46]
 [  0   0  32   0   0   0   3   0   4 268  23]
 [  0   0  16   0   0   0   0   0  12  54 253]]
------ TEST ACCURACY:  weights.35-0.41.hdf5  ------
0.556756756757
[[15  0  0  0  0  0  0  0  0  0]
 [ 3 11  0  0  0  0  0  0  1  0]
 [ 0  0 20  0  0  0  0  0  0  0]
 [ 0  0  0 14  0  0  0  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 1  0  1  0  0 11  0  0  2  0]
 [ 0  0  1  0  0 11  1  0  0  2]
 [ 0  0  3  0  2  0  0  4  7 14]
 [ 0  0  3  0  0  0  0  5  7 15]
 [ 0  0  3  0  0  0  0  0  7 20]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/4/weights.48-0.44.hdf5
------ TRAIN ACCURACY:  weights.48-0.44.hdf5  ------
0.790669371197
[[230   0   0   0   0   0   2   0   0   3   0]
 [  0 124   0   0   0   0   0   0   0   1   0]
 [  0   0 123   0   0   0   9  37  10  67   4]
 [  0   0   0 219   1   0   0   0   0   0   0]
 [  0   0   0   0  93   0   0   0   2   4   1]
 [  0   0   0   0   1  92   0   0   1   5   1]
 [  0   0   2   0   0   0 197  17   0  12   2]
 [  0   0  15   0   0   0  25 176   4  24   1]
 [  0   0   0   0   0   0   0   1 158  95  36]
 [  0   0   0   0   0   0   4   5   7 302  17]
 [  1   0   0   0   0   0   0   0  20  79 235]]
------ TEST ACCURACY:  weights.48-0.44.hdf5  ------
0.5
[[20  0  0  0  0  0  0  0  0]
 [ 0  5  0 10  0  0  0  0  0]
 [ 0  0  7  0  0  2  0  6  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  2  0 11  7  0  5  0]
 [ 0  0  5  0  1  8  0  1  0]
 [ 0  0  1  0  0  1  6 17  5]
 [ 0  0  0  0  0  1  8 13  3]
 [ 0  0  0  0  0  0  9 11 10]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/5/weights.41-0.47.hdf5
------ TRAIN ACCURACY:  weights.41-0.47.hdf5  ------
0.789019607843
[[235   0   0   0   0   0   3   0   0   2   0]
 [  0 138   0   1   0   0   0   1   0   0   0]
 [  0   0 121   0   0   0  36  58  24  11   0]
 [  0   0   0 218   0   0   0   0   2   0   0]
 [  0   0   0   0  91   1   0   1   3   2   2]
 [  0   0   0   0   0  95   0   0   0   1   4]
 [  0   0   0   0   0   0 229  10   0   0   1]
 [  0   0   9   0   0   0  58 172   4   1   1]
 [  1   0   9   0   0   0   1  19 227  35  28]
 [  0   0   8   0   0   0   8  13  55 257  19]
 [  0   0   4   0   0   2   1   0  74  25 229]]
------ TEST ACCURACY:  weights.41-0.47.hdf5  ------
0.628571428571
[[15  0  0  0  0  0  0  0  0]
 [ 0  7  0  0  1  5  1  1  0]
 [ 0  0 15  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 1  1  0  0 11  1  1  0  0]
 [ 0  0  0  0  7  8  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0]
 [ 3  0  0  1  0  5  2  9 10]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/6/weights.49-0.42.hdf5
------ TRAIN ACCURACY:  weights.49-0.42.hdf5  ------
0.719001919386
[[246   2   0   0   0   0   0   0   0   2   0]
 [  0 139   0   0   0   0   0   0   0   1   0]
 [  0   0  45   0   0   0  12  15  49  94  45]
 [  0   0   0 234   0   0   0   0   0   0   1]
 [  0   1   0   0  85   0   0   0   1   2  11]
 [  0   0   0   0   0  81   0   0   2   1  16]
 [  1   0   0   0   0   0 216   2   6  13  17]
 [  4   0   1   0   0   0  31  77  49  56  32]
 [  0   0   0   0   0   0   0   0 165  51  94]
 [  0   0   0   0   0   0   2   0  28 266  54]
 [  0   0   0   0   0   0   0   0  13  23 319]]
------ TEST ACCURACY:  weights.49-0.42.hdf5  ------
0.4
[[5 0 0 0 0 0 0]
 [0 3 0 0 1 0 1]
 [0 0 0 0 0 0 0]
 [4 0 5 0 0 1 0]
 [0 0 0 0 4 2 4]
 [0 0 0 0 3 2 5]
 [0 0 0 0 4 0 6]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/7/weights.43-0.42.hdf5
------ TRAIN ACCURACY:  weights.43-0.42.hdf5  ------
0.750491159136
[[238   0   0   0   0   0   0   0   0   2   0]
 [  0   5   0 135   0   0   0   0   0   0   0]
 [  0   0 188   0   0   0   1  12  14  32   3]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  78   0   0   0   1   4   2]
 [  0   0   0   0   0  82   0   0   0   2   1]
 [  0   0  13   0   0   0 195  25   0   2   0]
 [  2   0  70   0   0   0  20 144   0   8   1]
 [  2   0   5   0   0   1   0   1 189 100  22]
 [  0   0   1   0   0   0   4   4  15 324  12]
 [  0   0   1   0   0   0   0   1  38  78 247]]
------ TEST ACCURACY:  weights.43-0.42.hdf5  ------
0.7
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  0  1  1  2  0]
 [ 0  0  7  0  3  0  0  1  3  1]
 [ 3  0  0  8  4  0  0  0  0  0]
 [ 0  0  0  0 13  0  0  2  0  0]
 [ 0  4  0  0  0 12  4  0  0  0]
 [ 0  4  0  0  0  0 11  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/8/weights.49-0.43.hdf5
------ TRAIN ACCURACY:  weights.49-0.43.hdf5  ------
0.79765625
[[235   2   0   0   0   0   2   0   0   1   0]
 [  0 139   0   0   0   0   1   0   0   0   0]
 [  0   0 131   0   0   0  61  18  20  18   2]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  85   0   0   0   2   3   0]
 [  0   0   0   0   0  89   0   0   1   0   0]
 [  0   0   0   0   0   0 239   1   0   0   0]
 [  1   0  11   0   0   0  85 142   2   1   3]
 [  0   0   6   0   0   0   4   5 224  52  29]
 [  0   0   7   0   0   0  10   3  34 284  22]
 [  2   0   3   0   1   0   1   0  65  39 254]]
------ TEST ACCURACY:  weights.49-0.43.hdf5  ------
0.621052631579
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  1  0  0  0  2  8  4  0  0]
 [ 0  1  0 14  0  0  0  0  0  0  0]
 [ 0  0  0  0  2  1  0  0  2  2  3]
 [ 0  0  0  2  1  4  0  0  2  0  1]
 [ 0  0  0  0  0  0 15  0  0  0  0]
 [ 0  0  2  0  0  0  3  8  2  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/9/weights.50-0.38.hdf5
------ TRAIN ACCURACY:  weights.50-0.38.hdf5  ------
0.802828282828
[[231   4   0   0   0   0   3   0   0   2   0]
 [  0 124   0   0   0   0   0   0   0   1   0]
 [  0   0 121   0   0   0  13  62   5  41   8]
 [  0   0   0 218   0   0   0   0   0   2   0]
 [  0   0   0   0  89   0   0   0   2   7   2]
 [  0   0   0   0   0  92   0   0   1   6   1]
 [  0   0   0   0   0   0 217  20   0   3   0]
 [  0   0   8   0   0   0  23 207   0   6   1]
 [  0   0   0   0   0   0   0  13 141 101  35]
 [  0   0   0   0   0   0   4  15   1 297  13]
 [  0   0   0   0   0   0   1   0  11  73 250]]
------ TEST ACCURACY:  weights.50-0.38.hdf5  ------
0.538888888889
[[14  0  0  0  1  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0  7  0  2  3  1  2  0]
 [ 0 10  0  5  0  0  0  0  0]
 [ 0  0  0  0 12  3  0  0  0]
 [ 0  0  1  0  5  9  0  0  0]
 [ 0  0  0  0  0  4  4 16  6]
 [ 0  0  0  0  0  0  1 20  9]
 [ 0  0  0  0  0  0 10  9 11]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/10/weights.48-0.45.hdf5
------ TRAIN ACCURACY:  weights.48-0.45.hdf5  ------
0.726666666667
[[232   2   0   1   0   0   2   0   0   3   0]
 [  0 139   0   0   0   0   0   0   0   1   0]
 [  0   0  22   0   0   0  12  46  27 118  25]
 [  0   0   0 218   0   0   0   0   0   1   1]
 [  0   0   0   0  59   0   0   0   1  10  15]
 [  0   0   0   1   0  62   0   0   2   6  14]
 [  0   0   0   0   0   0 195  18   6  16   5]
 [  0   0   0   0   0   0  13 156  16  49  11]
 [  0   0   0   0   0   0   0   0 174  99  47]
 [  0   0   0   0   0   0   3   1   9 314  33]
 [  0   0   0   0   0   0   0   0  19  64 282]]
------ TEST ACCURACY:  weights.48-0.45.hdf5  ------
0.371428571429
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  1  0  1  8  5]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 1  0  0  3  0  0  0  1  0 10]
 [ 0  0  0 10  3  0  0  0  0  2]
 [ 0  0  0  0  0  2  7  0  5  1]
 [ 0  0  0  0  0  0  1  5  5  4]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/11/weights.25-0.41.hdf5
------ TRAIN ACCURACY:  weights.25-0.41.hdf5  ------
0.712720156556
[[236   0   0   0   0   0   2   2   0   0   0]
 [  1 129   0   9   0   0   0   1   0   0   0]
 [  0   0 235   0   0   0   1   7   0   1   1]
 [  0   0   0 219   0   0   0   0   0   0   1]
 [  1   0   6   0  76   2   0   0   0   3   2]
 [  1   0   1   0   2  80   0   0   0   1   5]
 [  7   0  35   0   0   0 139  57   0   1   1]
 [  2   0  88   0   0   0   6 148   1   0   0]
 [  1   0 103   0   0   0   0   9 121  37  49]
 [  4   0  92   0   1   2   3   6  16 194  42]
 [  4   0  54   0   2   0   0   2  29  30 244]]
------ TEST ACCURACY:  weights.25-0.41.hdf5  ------
0.55
[[15  0  0  0  0  0  0]
 [ 0 19  0  0  0  0  1]
 [ 0  0 15  0  0  0  0]
 [ 2  0  0  4  4  0  0]
 [ 2  0  0  7  1  0  0]
 [ 0  3  0  0  0  1 11]
 [ 0 15  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/12/weights.26-0.38.hdf5
------ TRAIN ACCURACY:  weights.26-0.38.hdf5  ------
0.680561122244
[[234   0   0   0   0   0   2   1   0   3   0]
 [  2   0   0 136   0   0   0   0   0   2   0]
 [  0   0 171   0   0   0   0   6   3  40  25]
 [  0   0   0 217   0   0   0   0   0   2   1]
 [  0   0   1   0  69   0   0   0   0   8   7]
 [  3   0   0   0  11  56   0   0   1   1   8]
 [  0   0  25   0   0   0 142  51   0  12  10]
 [  0   0  75   0   0   0   7 137   1  13  12]
 [  0   0   1   0   0   0   0   0 103  91 110]
 [  0   0   1   0   0   0   1   3   9 280  51]
 [  0   0   1   0   0   0   0   0  14  46 289]]
------ TEST ACCURACY:  weights.26-0.38.hdf5  ------
0.425
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  0  0  0  0  0  7  1]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 9  0  0  4  2  0  0  0  0  0]
 [ 3  0 11  0  1  0  0  0  0  5]
 [ 0  2  0  0  0  1  4  0  7  1]
 [ 0  6  0  0  0  0  5  0  4  0]
 [ 0  0  0  0  0  0  0  0  5 10]
 [ 0  0  0  0  0  0  0  1  7  7]
 [ 0  0  0  0  0  0  0  2  5  8]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/13/weights.50-0.45.hdf5
------ TRAIN ACCURACY:  weights.50-0.45.hdf5  ------
0.78515625
[[243   1   0   0   0   0   0   0   0   1   0]
 [  0 133   0   1   0   0   0   0   0   1   0]
 [  0   0 173   0   0   0   0   3   8  74   2]
 [  0   0   0 230   0   0   0   0   0   0   0]
 [  0   0   0   0  91   0   0   0   2   5   2]
 [  0   0   0   0   0  92   0   0   2   4   2]
 [  0   0  12   0   0   0 198  18   2  20   0]
 [  0   0  65   0   0   0  10 139   3  37   1]
 [  0   0   0   0   0   0   0   0 156 128  16]
 [  0   0   1   0   0   0   2   1   3 323  10]
 [  0   0   0   0   0   0   0   0  19  94 232]]
------ TEST ACCURACY:  weights.50-0.45.hdf5  ------
0.442105263158
[[ 5  0  0  0  2  0  0  3  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0  0  1  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  1  0  0  0  0  4  0]
 [ 0  0  1  0  0  1  0  3  0]
 [ 0  0  0  0  0  0  1 19  0]
 [ 0  0  0  0  0  0  3 13  4]
 [ 0  0  0  0  0  0  2 10  8]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/14/weights.40-0.39.hdf5
------ TRAIN ACCURACY:  weights.40-0.39.hdf5  ------
0.737692307692
[[245   0   0   0   0   0   2   0   0   3   0]
 [  0 137   0   2   0   0   0   0   0   1   0]
 [  0   0  73   0   0   0   4  90   0  80   8]
 [  0   0   0 234   0   0   0   0   0   0   1]
 [  0   0   0   0  89   0   0   0   0   8   3]
 [  0   0   0   0   0  90   0   0   1   6   3]
 [  0   0   2   0   0   0 173  70   0   9   1]
 [  0   0   4   0   0   0   9 222   0  13   2]
 [  0   0   0   0   0   0   0  12  85 139  79]
 [  0   0   0   0   0   0   2   8   4 305  26]
 [  0   0   0   0   0   0   0   2   4  84 265]]
------ TEST ACCURACY:  weights.40-0.39.hdf5  ------
0.490909090909
[[5 0 0 0 0 0 0 0]
 [0 0 0 1 9 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 1 8 0 1 0]
 [0 0 0 0 0 2 2 1]
 [0 0 1 0 2 3 5 4]
 [0 0 0 0 0 2 1 7]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/15/weights.45-0.48.hdf5
------ TRAIN ACCURACY:  weights.45-0.48.hdf5  ------
0.753333333333
[[241   1   0   0   0   0   0   0   0   3   0]
 [  0 134   0   0   0   0   0   0   0   1   0]
 [  0   0 109   0   0   0   8   7  40  89   7]
 [  0   4   0 218   1   0   0   0   1   1   0]
 [  1   0   0   0  90   0   0   0   2   5   2]
 [  0   0   0   0   0  92   0   0   1   3   4]
 [  1   0   1   0   0   0 208   8   7  17   3]
 [ 10   0  20   0   0   0  34  92  40  46   8]
 [  0   0   0   0   0   0   0   0 191  69  40]
 [  0   0   0   0   0   0   2   1  29 286  27]
 [  0   0   0   0   0   0   0   0  35  50 260]]
------ TEST ACCURACY:  weights.45-0.48.hdf5  ------
0.47619047619
[[10  0  0  0  0  0  0  0  0]
 [ 0  5  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  5  0]
 [ 1  9  0  0  0  0  0  0  0]
 [ 0  0  0  0  9  1  0  0  0]
 [ 0  0  1  0  0  4  1  4  0]
 [ 1  0  0  0  0  0  6  6  7]
 [ 0  0  0  0  0  0  3  9  3]
 [ 0  0  0  0  0  0  8  5  7]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/16/weights.50-0.47.hdf5
------ TRAIN ACCURACY:  weights.50-0.47.hdf5  ------
0.777128953771
[[212   0   0   1   0   0   0   0   0   2   0]
 [  1   4   0  95   0   0   0   0   0   0   0]
 [  0   0 163   0   0   0   0   4   6  45   2]
 [  0   0   0 195   0   0   0   0   0   0   0]
 [  0   0   0   0  62   0   0   0   1   0   2]
 [  0   0   0   0   0  65   0   0   0   2   3]
 [  0   0  14   0   1   0 170  19   1   3   2]
 [  1   0  56   0   0   0   8 138   1   9   2]
 [  0   0   2   0   0   0   1   0 145  65  22]
 [  0   0   1   0   0   1   3   3   6 228  18]
 [  0   0   0   0   0   0   0   0  14  41 215]]
------ TEST ACCURACY:  weights.50-0.47.hdf5  ------
0.488333333333
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0 40  0  0  0  0  0  0  0]
 [ 0  0 32  0  0  0  2  5  1  3  2]
 [ 0  0  0 40  0  0  0  0  0  0  0]
 [ 0  0  0  0 17  8  1  0  2  6  1]
 [ 0  0  0  2  4 12  0  0  4  1  7]
 [ 0  0  5  0  0  0 24 15  0  0  1]
 [ 0  0 18  0  0  0 12 13  0  2  0]
 [ 2  0  0  0  0  0  0  0 14 41 28]
 [ 5  1  0  3  3  0  1  0 21 53 13]
 [ 0  0  0  0  0  1  0  0 12 34 48]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_spca_conv1d_10/17/weights.36-0.33.hdf5
------ TRAIN ACCURACY:  weights.36-0.33.hdf5  ------
0.767676767677
[[235   0   0   0   0   0   3   1   0   1   0]
 [  0 124   0   0   0   0   0   1   0   0   0]
 [  0   0 196   0   0   0   6  36   4   7   1]
 [  0   0   0 217   0   0   0   0   2   1   0]
 [  0   0   0   0  83   7   0   1   6   3   0]
 [  0   0   0   0   0  97   0   0   2   1   0]
 [  0   0   7   0   0   0 196  37   0   0   0]
 [  0   0  40   0   0   0  16 189   0   0   0]
 [  0   0  36   0   0   0   0  24 195  24  11]
 [  4   0  40   0   0   4   5  11  62 188  16]
 [  0   0  23   0   0   1   0   5  85  41 180]]
------ TEST ACCURACY:  weights.36-0.33.hdf5  ------
0.494444444444
[[15  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  3  0  0  0  0  0  0]
 [ 0  0  4  0  0  1  2  5  3  0]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 15  0  0  0  0]
 [ 0  0  3  0  0 11  1  0  0  0]
 [ 0  0  5  0  0  0  3 13  5  4]
 [ 0  0  8  0  0  3  5  4  9  1]
 [ 1  0  0  0  1  1  0 13  9  5]]
[0.5894736842105263, 0.5533333333333333, 0.5567567567567567, 0.5, 0.6285714285714286, 0.4, 0.7, 0.6210526315789474, 0.5388888888888889, 0.37142857142857144, 0.55, 0.425, 0.4421052631578947, 0.4909090909090909, 0.47619047619047616, 0.48833333333333334, 0.49444444444444446]
0.519205170753
0.083779009663

Process finished with exit code 0

'''