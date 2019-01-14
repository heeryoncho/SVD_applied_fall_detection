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

X = pickle.load(open("data/X_umafall_spca.p", "rb"))
y = pickle.load(open("data/y_umafall_spca.p", "rb"))

n_classes = 11
signal_rows = 450
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
    print X_train.shape # (2465, 450)

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

    new_dir = 'model/umafall_spca_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_spca_conv1d_10/' + str(i+1) + '/'
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
[0.47368421052631576, 0.4666666666666667, 0.5243243243243243, 0.39473684210526316, 0.4857142857142857, 0.3, 0.7181818181818181, 0.43157894736842106, 0.4444444444444444, 0.5142857142857142, 0.4, 0.49375, 0.45263157894736844, 0.4, 0.38095238095238093, 0.5116666666666667, 0.38333333333333336]
0.457408894913
0.0869818493507
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_spca_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/1/weights.38-0.45.hdf5
------ TRAIN ACCURACY:  weights.38-0.45.hdf5  ------
0.814604462475
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 118   0   7   0   0   0   0   0   0   0]
 [  2   0 227   0   0   1   2   4   6   4   4]
 [  0   0   0 218   0   0   0   0   0   1   1]
 [  1   0   0   0  96   1   0   0   2   0   0]
 [  0   0   0   0   0  99   0   0   0   0   1]
 [  1   0  30   0   1   0 154  48   4   2   0]
 [  2   0  47   0   0   0  27 165   3   1   0]
 [  2   1  32   0   4   1   3   7 187  35  18]
 [  3   0  10   0   1   1  11   5  13 245  31]
 [  2   2  14   2   0   2   0   0  30  22 261]]
------ TEST ACCURACY:  weights.38-0.45.hdf5  ------
0.473684210526
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 12  0  3  0  0  0  0  0  0  0]
 [ 0  0  7  0  0  0  0  4  1  1  2]
 [ 0  0  0  9  0  4  0  0  0  2  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  0 10  0  1  0  0]
 [ 0  0  2  0  0  0 10  3  0  0  0]
 [ 2  0  9  0  3  0  0  0  7  2  7]
 [ 0  0  7  0  0  1  0  0  8 14 10]
 [ 4  0  0  0  0  1  0  0  8  4 13]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/2/weights.23-0.46.hdf5
------ TRAIN ACCURACY:  weights.23-0.46.hdf5  ------
0.731736526946
[[235   0   0   0   0   0   1   1   0   3   0]
 [  6  96   0  18   0   0   0   0   0   1   4]
 [  0   0 162   0   0   0   1  40  24  17   6]
 [  2   0   0 211   1   0   0   0   0   3   3]
 [  0   0   0   0  90   0   0   0   4   2   4]
 [  2   0   0   0   2  79   0   0   4   8   5]
 [  0   0  25   0   2   0 101  99  10   3   0]
 [  0   0  45   0   0   0  19 173   5   1   2]
 [  1   0  13   0   3   0   6  12 185  68  17]
 [  0   0   3   0   1   0   4  16  18 276  12]
 [  0   0   9   2   1   1   0   4  34  74 225]]
------ TEST ACCURACY:  weights.23-0.46.hdf5  ------
0.466666666667
[[15  0  0  0  0  0  0  0  0]
 [14  0  0  0  0  0  0  1  0]
 [ 0  0  9  0  0  5  0  1  0]
 [ 0  0  0  7  0  0  0  1  7]
 [ 0  0  0  0  2 13  0  0  0]
 [ 0  0  5  0  0  9  1  0  0]
 [ 0  0  1  0  0  0  6  3  5]
 [ 0  0  0  0  0  0  3 16 11]
 [ 0  0  0  0  0  0  2  7  6]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/3/weights.49-0.49.hdf5
------ TRAIN ACCURACY:  weights.49-0.49.hdf5  ------
0.789473684211
[[237   0   0   1   0   1   1   0   0   0   0]
 [  0 124   0   1   0   0   0   0   0   0   0]
 [  1   0 228   0   0   0   2   0   5   0   9]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   1   0  95   0   0   0   0   1   3]
 [  0   0   0   0   0 100   0   0   0   0   0]
 [  0   0  53   0   0   0 178   2   4   2   1]
 [  2   0 101   0   0   0  72  62   3   0   5]
 [  1   0  23   0   1   1   6   2 169  11  76]
 [  2   0   9   0   0   0  10   1   6 230  72]
 [  0   0  11   2   0   2   0   0   3  10 307]]
------ TEST ACCURACY:  weights.49-0.49.hdf5  ------
0.524324324324
[[15  0  0  0  0  0  0  0  0]
 [ 4 11  0  0  0  0  0  0  0]
 [ 0  0 18  0  1  0  0  0  1]
 [ 0  1  0 14  0  0  0  0  0]
 [ 0  0  1  0 12  0  1  0  1]
 [ 0  0  0  0 13  2  0  0  0]
 [ 2  1  2  0  0  0  0  2 23]
 [ 1  0  3  0  0  0  1  5 20]
 [ 1  1  0  0  0  0  0  8 20]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/4/weights.17-0.46.hdf5
------ TRAIN ACCURACY:  weights.17-0.46.hdf5  ------
0.650709939148
[[233   0   0   0   0   0   2   0   0   0   0]
 [ 47  45   0  32   0   0   0   0   0   0   1]
 [  5   0  77   0   1   0  19  87  36  16   9]
 [ 11   0   0 206   0   0   0   0   0   1   2]
 [ 14   0   0   0  81   2   0   0   2   0   1]
 [ 19   0   0   0   0  77   0   0   1   0   3]
 [  2   0   3   0   2   0 164  48   9   2   0]
 [  3   0  18   0   0   0  71 138  14   1   0]
 [  8   0   1   0   4   1  11  13 177  45  30]
 [ 23   0   2   0   2   1  15   9  35 210  38]
 [ 15   0   2   2   2   6   0   6  63  43 196]]
------ TEST ACCURACY:  weights.17-0.46.hdf5  ------
0.394736842105
[[20  0  0  0  0  0  0  0  0  0  0]
 [15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  5  0  0  0  0  7  1  1  1]
 [15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  0 18  4  1  0  0]
 [ 0  0  0  0  0  0 10  5  0  0  0]
 [ 5  0  0  0  3  1  0  0 10  6  5]
 [ 4  0  0  0  0  0  0  0 10  4  7]
 [ 7  0  0  0  0  0  0  0  4  6 13]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/5/weights.48-0.53.hdf5
------ TRAIN ACCURACY:  weights.48-0.53.hdf5  ------
0.767843137255
[[240   0   0   0   0   0   0   0   0   0   0]
 [  2 131   0   6   0   0   0   0   0   0   1]
 [  3   0 186   0   0   0   3  23  32   2   1]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  1   0   0   0  97   0   0   0   2   0   0]
 [  1   0   0   0   0  96   0   0   3   0   0]
 [ 14   0  17   0   1   0 150  30  25   3   0]
 [  3   0  34   0   0   0  50 134  23   1   0]
 [  8   0  10   2   2   0   0   6 281   9   2]
 [  6   0   1   0   1   0   4   2 112 227   7]
 [  4   0   0   2   0   2   0   0 127   3 197]]
------ TEST ACCURACY:  weights.48-0.53.hdf5  ------
0.485714285714
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0 12  0  0  0  1  1  0  1]
 [ 0  2  0 13  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 3  0  0  0  1  3  1  7  0  0]
 [ 0  0  1  0  0  7  7  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 1  0 13  0  0  0  0  9  6  1]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/6/weights.50-0.50.hdf5
------ TRAIN ACCURACY:  weights.50-0.50.hdf5  ------
0.776583493282
[[246   0   0   3   0   0   1   0   0   0   0]
 [  1 128   0  11   0   0   0   0   0   0   0]
 [  4   0 228   0   0   0   6   0  20   1   1]
 [  0   0   0 235   0   0   0   0   0   0   0]
 [  0   0   0   0  97   0   0   0   2   1   0]
 [  1   0   0   0   0  98   0   0   1   0   0]
 [  0   0  24   0   0   0 224   0   4   3   0]
 [  0   0 106   0   0   0 106  21  16   1   0]
 [  7   1  13   4   2   0   7   0 258  13   5]
 [  9   0   5   0   1   1   9   0  69 248   8]
 [ 10   5   7   9   0   5   1   0  61  17 240]]
------ TEST ACCURACY:  weights.50-0.50.hdf5  ------
0.3
[[5 0 0 0 0 0 0 0 0 0]
 [0 3 0 1 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [4 3 0 0 0 3 0 0 0 0]
 [2 0 1 0 1 0 0 4 0 2]
 [1 0 0 0 0 0 0 6 2 1]
 [0 0 0 0 1 0 0 7 1 1]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/7/weights.49-0.50.hdf5
------ TRAIN ACCURACY:  weights.49-0.50.hdf5  ------
0.85068762279
[[238   0   0   0   0   0   2   0   0   0   0]
 [  0 136   0   3   0   0   0   0   0   0   1]
 [  0   0 220   0   0   0   7  11   5   3   4]
 [  0   0   0 220   0   0   0   0   0   0   0]
 [  0   0   0   0  82   0   1   0   1   0   1]
 [  0   0   0   0   0  84   0   0   0   0   1]
 [  0   0  14   0   0   0 182  37   1   1   0]
 [  0   0  34   0   0   0  44 164   3   0   0]
 [  2   0  17   0   2   1  11  10 216  28  33]
 [  2   0   3   0   0   0  15   7   7 306  20]
 [  0   0   8   2   0   3   0   0  14  21 317]]
------ TEST ACCURACY:  weights.49-0.50.hdf5  ------
0.718181818182
[[14  0  0  0  0  0  0  1  0  0]
 [ 0 13  0  0  0  0  0  1  1  0]
 [ 0  0  9  0  0  0  0  0  1  5]
 [ 2  0  0  8  0  0  0  4  0  1]
 [ 2  0  0  0 10  0  0  0  0  3]
 [ 0  4  0  0  0 15  0  1  0  0]
 [ 0  0  0  0  0  5 10  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/8/weights.32-0.54.hdf5
------ TRAIN ACCURACY:  weights.32-0.54.hdf5  ------
0.761328125
[[237   0   0   0   0   0   1   1   0   1   0]
 [  0 137   0   3   0   0   0   0   0   0   0]
 [  2   1 183   0   0   1   2   6  40   8   7]
 [  0   0   0 216   0   0   0   0   0   3   1]
 [  2   4   0   0  81   0   0   0   2   0   1]
 [  0   0   0   0   0  89   0   0   1   0   0]
 [  6   0  22   0   0   0  97  52  49  13   1]
 [  0   0  40   0   0   0  25 117  49  11   3]
 [  3   1  17   0   1   1   2   7 236  27  25]
 [  4   0   6   0   0   0   2   3  44 282  19]
 [  3   0   8   1   0   4   0   0  41  34 274]]
------ TEST ACCURACY:  weights.32-0.54.hdf5  ------
0.431578947368
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  1  1  5  0  0]
 [ 0  8  0  2  0  1  0  0  0  0  4]
 [ 0  0  0  0  2  3  0  0  4  0  1]
 [ 0  0  0  0  0  8  0  0  0  0  2]
 [ 0  0  0  0  0  0  1  2  8  3  1]
 [ 0  0  8  0  0  0  1  5  0  0  1]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/9/weights.23-0.46.hdf5
------ TRAIN ACCURACY:  weights.23-0.46.hdf5  ------
0.633939393939
[[239   0   0   0   0   0   1   0   0   0   0]
 [  2 123   0   0   0   0   0   0   0   0   0]
 [ 12   0 207   0   1   0   8   3  17   0   2]
 [ 44  75   0  74   1  20   0   0   0   1   5]
 [ 15   2   1   0  78   1   0   0   3   0   0]
 [ 18   0   0   0   0  79   0   0   3   0   0]
 [  4   0  39   0   1   0 161  27   8   0   0]
 [  4   0  85   0   0   0  61  88   7   0   0]
 [ 22   2  32   0   3   1   6   1 195  14  14]
 [ 33   3  27   0   1   0  11   1  80 138  36]
 [ 29  13  18   0   0   2   0   0  74  12 187]]
------ TEST ACCURACY:  weights.23-0.46.hdf5  ------
0.444444444444
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0  0]
 [ 1  0 13  0  0  0  1  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  3  0  1  0 11  0  0  0  0]
 [ 0  0  3  0  0  0  7  4  1  0  0]
 [ 2  0  5  0  1  0  1  0  8  4  9]
 [ 7  0  5  0  0  0  0  0  4  5  9]
 [ 2  4  0  0  0  1  0  0 13  1  9]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/10/weights.27-0.51.hdf5
------ TRAIN ACCURACY:  weights.27-0.51.hdf5  ------
0.707450980392
[[239   0   0   0   0   0   0   0   0   1   0]
 [ 14 109   0  17   0   0   0   0   0   0   0]
 [ 14   0 160   0   0   0   5   2  55  13   1]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  3   0   0   0  79   1   0   0   1   0   1]
 [  2   0   0   0   0  80   0   0   2   0   1]
 [  9   0  15   0   2   0 153  18  39   4   0]
 [  5   0  58   0   1   0  73  78  29   0   1]
 [ 31   0   6   2   5   1   1   2 250  15   7]
 [ 32   0   0   0   2   1   3   0  81 228  13]
 [ 34   0   1   4   2   5   0   0  82  28 209]]
------ TEST ACCURACY:  weights.27-0.51.hdf5  ------
0.514285714286
[[10  0  5  0  0  0  0  0  0  0]
 [ 1  0  0  2  2  0  0  5  3  2]
 [ 0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0 13  2  0  0  0  0  0]
 [ 1  0  0  6  7  0  0  1  0  0]
 [ 0  3  0  0  0  9  3  0  0  0]
 [ 0  3  0  1  0  1  0 10  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/11/weights.44-0.51.hdf5
------ TRAIN ACCURACY:  weights.44-0.51.hdf5  ------
0.777299412916
[[238   0   0   0   0   0   2   0   0   0   0]
 [ 17 117   0   6   0   0   0   0   0   0   0]
 [  1   0 138   0   2   0  48  26  28   1   1]
 [ 19   0   0 186   1   5   0   0   4   1   4]
 [  1   0   0   0  88   0   0   0   1   0   0]
 [  0   0   0   0   0  87   0   0   3   0   0]
 [  0   0   1   0   0   0 223  16   0   0   0]
 [  0   0   5   0   0   0  91 142   7   0   0]
 [  2   0   4   0   5   1  17   9 265   7  10]
 [  1   0   0   0   1   0  34   3  72 231  18]
 [  5   0   1   0   2   2   7   0  72   5 271]]
------ TEST ACCURACY:  weights.44-0.51.hdf5  ------
0.4
[[15  0  0  0  0  0  0  0  0  0]
 [ 1  8  0  0  0  3  1  4  2  1]
 [ 5  0  4  0  6  0  0  0  0  0]
 [ 2  0  0  8  0  0  0  0  0  0]
 [ 4  0  0  4  2  0  0  0  0  0]
 [ 0  0  0  0  0  3 12  0  0  0]
 [ 0 13  0  0  0  1  0  1  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/12/weights.47-0.47.hdf5
------ TRAIN ACCURACY:  weights.47-0.47.hdf5  ------
0.791182364729
[[234   0   0   4   0   0   1   1   0   0   0]
 [  1 126   0  13   0   0   0   0   0   0   0]
 [  0   0 149   0   0   0  60  13  17   5   1]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  80   0   3   0   2   0   0]
 [  0   0   0   0   0  80   0   0   0   0   0]
 [  0   0   1   0   0   0 200  36   2   1   0]
 [  0   0  19   0   0   0  86 137   3   0   0]
 [  2   0  12   1   0   1  20  11 236  16   6]
 [  0   0   5   0   0   0  31   6  34 262   7]
 [  1   0   7   2   0   3   5   3  49  29 251]]
------ TEST ACCURACY:  weights.47-0.47.hdf5  ------
0.49375
[[15  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 13  0  0  0  0  0  3  3  1]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [12  0  0  1  0  2  0  0  0  0  0]
 [ 7  0  0  9  0  2  0  0  2  0  0]
 [ 0  0  0  0  0  0 14  1  0  0  0]
 [ 0  0  0  0  0  0  5 10  0  0  0]
 [ 0  0  1  0  0  0  0  0  6  4  4]
 [ 0  0  0  0  0  1  1  0  8  3  2]
 [ 0  1  2  0  0  0  1  0  6  4  1]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/13/weights.34-0.50.hdf5
------ TRAIN ACCURACY:  weights.34-0.50.hdf5  ------
0.7125
[[244   0   0   1   0   0   0   0   0   0   0]
 [  4  97   0  34   0   0   0   0   0   0   0]
 [ 11   0 150   0   1   0  11  18  62   6   1]
 [  0   0   0 229   0   0   0   0   0   1   0]
 [  0   0   0   0  96   0   0   0   3   1   0]
 [  3   0   0   1   0  92   0   0   3   0   1]
 [  6   0  29   0   4   0 159   5  42   4   1]
 [  6   0  68   0   0   1  68  74  38   0   0]
 [ 10   0   1   3   4   0   5   4 253  12   8]
 [ 17   0   1   0   7   0   2   1  82 185  45]
 [ 17   0   3   5   0   3   0   0  71   1 245]]
------ TEST ACCURACY:  weights.34-0.50.hdf5  ------
0.452631578947
[[ 6  0  0  0  0  2  0  0  2  0]
 [ 0  3  0  2  0  0  0  0  0  0]
 [ 0  0  3  0  0  0  0  2  0  0]
 [ 0  0  0  5  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  2  0  3  0  0]
 [ 0  0  0  0  0  4  1  0  0  0]
 [ 0  0  0  1  0  0  0 10  1  8]
 [ 3  0  0  1  0  0  0  1  6  9]
 [ 1  1  0  1  1  0  0  7  2  7]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/14/weights.32-0.47.hdf5
------ TRAIN ACCURACY:  weights.32-0.47.hdf5  ------
0.760769230769
[[244   2   0   2   0   0   1   0   0   1   0]
 [  0 135   0   5   0   0   0   0   0   0   0]
 [  2   0 188   0   5   0  13   0  27  12   8]
 [  0   0   0 232   0   0   0   0   0   1   2]
 [  0   0   0   0  97   0   0   0   1   1   1]
 [  1   0   0   0   9  86   0   0   0   0   4]
 [  1   0  24   0   2   0 208   1  13   6   0]
 [  2   0  70   0   1   0 117  34  16   6   4]
 [  2   0   9   1  11   1   4   0 191  47  49]
 [  1   0   1   0   8   0   7   0  13 275  40]
 [  2   0   5   2   6   2   0   0  25  25 288]]
------ TEST ACCURACY:  weights.32-0.47.hdf5  ------
0.4
[[ 4  1  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  4  4  0  2  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  3  6  0  1  0  0]
 [ 0  0  1  0  0  0  2  2]
 [ 0  0  0  1  0  2 10  2]
 [ 0  0  0  0  0  3  3  4]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/15/weights.41-0.52.hdf5
------ TRAIN ACCURACY:  weights.41-0.52.hdf5  ------
0.741568627451
[[245   0   0   0   0   0   0   0   0   0   0]
 [  1 134   0   0   0   0   0   0   0   0   0]
 [  4   0 232   0   0   0   0  17   7   0   0]
 [  2  14   0 206   0   0   0   0   1   1   1]
 [  1   0   0   0  97   0   0   0   2   0   0]
 [  2   0   0   0   0  92   0   0   5   0   1]
 [ 19   0  54   0   1   0 116  47   8   0   0]
 [  5   0  79   0   0   0  13 149   4   0   0]
 [  9   3  37   0   3   1   1  12 227   5   2]
 [  8   2  26   0   0   0   5  16  82 186  20]
 [  7   7  22   1   2   2   0   1  95   1 207]]
------ TEST ACCURACY:  weights.41-0.52.hdf5  ------
0.380952380952
[[10  0  0  0  0  0  0  0  0]
 [ 0  1  0  4  0  0  0  0  0]
 [ 0  0  0  0  0  5  0  0  0]
 [ 0  9  0  1  0  0  0  0  0]
 [ 2  0  3  0  1  4  0  0  0]
 [ 0  0  3  0  0  7  0  0  0]
 [ 0  0  2  0  0  4 12  0  2]
 [ 2  0  0  0  0  0  7  4  2]
 [ 3  1  1  0  0  1  7  3  4]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/16/weights.32-0.50.hdf5
------ TRAIN ACCURACY:  weights.32-0.50.hdf5  ------
0.779562043796
[[211   3   0   0   0   0   1   0   0   0   0]
 [  0  98   0   2   0   0   0   0   0   0   0]
 [  0   0 190   0   0   0   2   5   7   9   7]
 [  0   2   0 191   0   0   0   0   0   2   0]
 [  1   0   0   0  61   0   0   0   1   0   2]
 [  1   0   0   0   0  64   0   0   0   0   5]
 [  2   0  50   0   0   0 111  39   3   4   1]
 [  3   0  73   0   0   0  25 109   3   1   1]
 [  1   0  26   0   0   0   2   3 141  26  36]
 [  2   0  12   0   0   0   8   2   6 204  26]
 [  0   1  17   1   0   2   0   0  11  16 222]]
------ TEST ACCURACY:  weights.32-0.50.hdf5  ------
0.511666666667
[[40  0  0  0  0  0  0  0  0  0  0]
 [10 27  0  0  0  0  0  0  0  0  3]
 [ 0  0 41  0  0  0  0  1  0  2  1]
 [ 4  0  0 35  0  0  0  0  1  0  0]
 [ 0  4  8  0 16  0  1  0  3  1  2]
 [ 1  0  2  0  2  6  0  0  3  3 13]
 [ 0  0  4  0  0  0 22 19  0  0  0]
 [ 0  0 22  0  0  0  3 19  0  1  0]
 [ 1  2 20  2  0  0  5  2 19 22 12]
 [ 2  4 17  0  0  0  3  2  7 46 19]
 [ 2  4  5  0  0  0  0  0 10 38 36]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_spca_conv1d_10/17/weights.43-0.44.hdf5
------ TRAIN ACCURACY:  weights.43-0.44.hdf5  ------
0.742222222222
[[239   0   0   0   0   0   0   0   0   1   0]
 [  2 122   0   1   0   0   0   0   0   0   0]
 [  0   0 199   0   1   0   3   4   5  27  11]
 [  0   0   0 219   0   0   0   0   0   1   0]
 [  0   0   0   0  97   0   0   0   1   1   1]
 [  0   0   0   0   1  93   0   0   0   5   1]
 [  0   0  34   0   2   0 114  10   9  67   4]
 [  1   0  62   0   0   0  33  75  19  43  12]
 [  0   0   3   1   2   0   2   0 149  96  37]
 [  1   0   0   0   1   0   2   2   5 303  16]
 [  2   0   4   1   2   0   0   0   5  94 227]]
------ TEST ACCURACY:  weights.43-0.44.hdf5  ------
0.383333333333
[[14  0  0  1  0  0  0  0  0  0  0]
 [ 3  0  0 12  0  0  0  0  0  0  0]
 [ 0  0  3  0  1  0  1  0  4  4  2]
 [ 0  0  0 15  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  5  0  0 10  0]
 [ 0  0  2  0  0  0  2  2  3  5  1]
 [ 1  0  2  0  3  0  0  2  7 13  2]
 [ 3  0  3  0  0  0  4  0  6 12  2]
 [ 0  0  0  2  2  2  0  0  3 10 11]]
[0.47368421052631576, 0.4666666666666667, 0.5243243243243243, 0.39473684210526316, 0.4857142857142857, 0.3, 0.7181818181818181, 0.43157894736842106, 0.4444444444444444, 0.5142857142857142, 0.4, 0.49375, 0.45263157894736844, 0.4, 0.38095238095238093, 0.5116666666666667, 0.38333333333333336]
0.457408894913
0.0869818493507

Process finished with exit code 0

'''