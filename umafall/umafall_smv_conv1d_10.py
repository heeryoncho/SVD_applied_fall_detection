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

X = pickle.load(open("data/X_umafall_smv.p", "rb"))
y = pickle.load(open("data/y_umafall_smv.p", "rb"))

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

    new_dir = 'model/umafall_smv_conv1d_10/' + str(i+1) + '/'
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
    path_str = 'model/umafall_smv_conv1d_10/' + str(i+1) + '/'
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
[0.5368421052631579, 0.48, 0.5405405405405406, 0.5157894736842106, 0.6571428571428571, 0.32, 0.21818181818181817, 0.5789473684210527, 0.5166666666666667, 0.4, 0.33, 0.34375, 0.2736842105263158, 0.38181818181818183, 0.5047619047619047, 0.48333333333333334, 0.40555555555555556]
0.44041258917
0.114797737566
-----

/usr/bin/python2.7 /mnt/0cdd1a7e-3686-4cf7-9055-145eb5fe70f3/hcilab/OSS/electronicsletters2018/umafall/umafall_smv_conv1d_10.py
/home/hcilab/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.

>>>>>>>>>>>>>> 1 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/1/weights.37-0.42.hdf5
------ TRAIN ACCURACY:  weights.37-0.42.hdf5  ------
0.675862068966
[[211   0   1   0   0   0  23   2   0   0   3]
 [  0 119   0   5   0   0   0   0   0   0   1]
 [  0   0 179   0   0   0   1   2   1   0  67]
 [  0   0   0 214   0   0   0   0   0   0   6]
 [  0   0   5   0  70   0   3   1   0   1  20]
 [  0   0   0   0   0  75   1   0   0   0  24]
 [  0   0  23   0   0   0 129  48   0   0  40]
 [  0   0  30   0   0   0   8 140   2   3  62]
 [  0   0   0   0   0   0   0   1  88   5 196]
 [  0   0   1   0   0   0   0   1   7 112 199]
 [  0   0   0   0   0   0   0   0   4   2 329]]
------ TEST ACCURACY:  weights.37-0.42.hdf5  ------
0.536842105263
[[14  0  0  0  0  1  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  1  0]
 [ 0  0 14  0  0  0  0  0  0  1]
 [ 0  0  0 11  1  0  0  0  0  3]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  3  0  0  2  5  0  2  3]
 [ 0  0  1  0  0  0 13  0  0  1]
 [ 0  0  0  0  0  0  0  2  1 27]
 [ 0  0  0  0  0  0  0  5  5 30]
 [ 0  0  0  0  0  0  0  3  0 27]]

>>>>>>>>>>>>>> 2 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/2/weights.19-0.56.hdf5
------ TRAIN ACCURACY:  weights.19-0.56.hdf5  ------
0.550898203593
[[234   3   0   0   0   0   0   0   3   0   0]
 [  1 117   0   6   0   0   0   0   1   0   0]
 [  2   0 123   0   0   0   6   4 102   0  13]
 [  0   2   0 209   0   0   0   0   8   0   1]
 [ 46   0   0   0  24   0   0   0  12   0  18]
 [ 11   0   0   0   0  49   0   0  27   0  13]
 [ 56   0   0   0   0   0 105  16  50   0  13]
 [ 21   0  15   0   0   0  14  75 111   0   9]
 [  0   0   0   0   0   0   4   0 291   0  10]
 [  7   2   1   0   0   0   1   0 273  15  31]
 [  4   0   0   0   0   0   0   1 207   0 138]]
------ TEST ACCURACY:  weights.19-0.56.hdf5  ------
0.48
[[ 5 10  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0  8  0  5  0  2  0  0]
 [ 0  1  0 14  0  0  0  0  0]
 [ 7  0  0  0  8  0  0  0  0]
 [ 0  0  3  0 10  2  0  0  0]
 [ 0  0  0  0  0  0 15  0  0]
 [ 0  0  0  0  0  0 22  0  8]
 [ 0  0  0  0  0  0 10  0  5]]

>>>>>>>>>>>>>> 3 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/3/weights.17-0.40.hdf5
------ TRAIN ACCURACY:  weights.17-0.40.hdf5  ------
0.649392712551
[[227   1   2   0   0   0   9   1   0   0   0]
 [  0 114   0  11   0   0   0   0   0   0   0]
 [  0   0 148   0   0   0  40  41   2  14   0]
 [  0   0   0 218   0   0   0   0   1   1   0]
 [  3   0   0   0  70   3  13   1   1   5   4]
 [  2   0   0   1   2  77   2   3   5   4   4]
 [  0   0   4   0   0   0 209  26   0   1   0]
 [  0   0  24   0   0   0  65 154   0   2   0]
 [  0   0  16   0   0   0  93  12 108  51  10]
 [  2   1   4   0   0   0  77  56  36 143  11]
 [  0   0   7   0   0   0  52  46  36  58 136]]
------ TEST ACCURACY:  weights.17-0.40.hdf5  ------
0.540540540541
[[15  0  0  0  0  0  0  0  0]
 [ 3 10  1  0  1  0  0  0  0]
 [ 0  0 20  0  0  0  0  0  0]
 [ 0  1  0 12  0  0  1  0  1]
 [ 0  0  1  0 10  2  0  1  1]
 [ 0  0  0  0  8  6  0  0  1]
 [ 0  2  2  0  5  3  6  4  8]
 [ 0  0  0  0  2  6  4  5 13]
 [ 0  0  4  0  0  0  1  9 16]]

>>>>>>>>>>>>>> 4 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/4/weights.11-0.60.hdf5
------ TRAIN ACCURACY:  weights.11-0.60.hdf5  ------
0.558215010142
[[207  15   5   0   0   1   4   3   0   0   0]
 [  0 108   1  15   0   0   0   1   0   0   0]
 [  0   0 135   0   0   0   0   1  64  50   0]
 [  0   0   0 214   0   0   0   0   2   3   1]
 [  4   0   0   0  35  18   0   0  21  16   6]
 [  0   0   0   2   0  63   0   0  16  14   5]
 [  0   0  12   0   0   0  68  23  75  51   1]
 [  0   0  36   0   0   0   6  72  99  31   1]
 [  0   3   6   0   0   0   0   0 228  44   9]
 [  1   4   2   0   0   1   0   2 165 150  10]
 [  0   0   3   0   0   1   0   1 135  99  96]]
------ TEST ACCURACY:  weights.11-0.60.hdf5  ------
0.515789473684
[[18  2  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0 15  0  0  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  2  0  7  4  9  3  0]
 [ 0  0  8  0  0  4  3  0  0]
 [ 0  0  5  0  0  0 19  6  0]
 [ 0  0  0  0  0  0 21  4  0]
 [ 0  0  0  0  0  0 26  3  1]]

>>>>>>>>>>>>>> 5 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/5/weights.42-0.66.hdf5
------ TRAIN ACCURACY:  weights.42-0.66.hdf5  ------
0.751764705882
[[234   0   2   0   0   0   2   2   0   0   0]
 [  0 134   0   5   0   0   0   1   0   0   0]
 [  0   0 177   0   0   0   2  20   2  48   1]
 [  0   0   0 217   0   0   0   0   0   3   0]
 [  0   0   2   0  82   0   0   1   0  11   4]
 [  0   0   0   0   0  82   0   0   3  14   1]
 [  0   0   3   0   0   0 159  64   6   7   1]
 [  0   0  12   0   0   0   4 208   5  16   0]
 [  0   0   5   0   0   0   0   1 220  87   7]
 [  0   0   0   0   0   0   2   2  66 289   1]
 [  0   0   0   0   0   0   0   0  84 136 115]]
------ TEST ACCURACY:  weights.42-0.66.hdf5  ------
0.657142857143
[[15  0  0  0  0  0  0  0]
 [ 0  9  0  0  3  3  0  0]
 [ 0  0 15  0  0  0  0  0]
 [ 0  1  0  6  6  0  2  0]
 [ 0  1  0  1 13  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  5  2 12 11]]

>>>>>>>>>>>>>> 6 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/6/weights.13-0.38.hdf5
------ TRAIN ACCURACY:  weights.13-0.38.hdf5  ------
0.629942418426
[[224   8   6   9   1   0   2   0   0   0   0]
 [  0 117   1  21   0   0   1   0   0   0   0]
 [  0   0 185   0   0   0   1  20   1  49   4]
 [  0   0   0 233   0   0   0   0   0   1   1]
 [  1   0   1   3  72   0   0   0   0  16   7]
 [  0   0   0   6   6  61   1   0   0  23   3]
 [  0   0  36   0   0   0 120  65   1  28   5]
 [  0   0  50   0   0   0  17 134   2  43   4]
 [  0   0  38   0   0   0   2   5  77 175  13]
 [  0   0  19   2   0   1   3   3  13 295  14]
 [  0   0  49   3   0   0   0   5  20 155 123]]
------ TEST ACCURACY:  weights.13-0.38.hdf5  ------
0.32
[[0 0 2 1 2 0 0 0]
 [0 5 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 4 0 1 5 0 0 0]
 [0 6 0 0 0 0 3 1]
 [0 3 0 0 0 3 4 0]
 [0 0 0 0 0 1 7 2]]

>>>>>>>>>>>>>> 7 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/7/weights.28-0.50.hdf5
------ TRAIN ACCURACY:  weights.28-0.50.hdf5  ------
0.451866404715
[[240   0   0   0   0   0   0   0   0   0   0]
 [  2 138   0   0   0   0   0   0   0   0   0]
 [190   0  43   0   0   0   6  11   0   0   0]
 [  7  11   0 195   0   1   0   0   4   0   2]
 [ 77   0   0   0   7   0   0   0   1   0   0]
 [ 57   0   0   0   0  24   0   0   2   0   2]
 [153   0   0   0   0   0  80   1   1   0   0]
 [189   0   2   0   0   0  16  38   0   0   0]
 [ 82   0   0   0   0   0   3   8 220   1   6]
 [128   1   0   0   0   0   0   0 155  53  23]
 [169   0   0   0   0   0   0   7  76   1 112]]
------ TEST ACCURACY:  weights.28-0.50.hdf5  ------
0.218181818182
[[15  0  0  0  0  0  0  0]
 [15  0  0  0  0  0  0  0]
 [ 2  0  7  1  2  0  0  3]
 [15  0  0  0  0  0  0  0]
 [14  0  0  0  0  0  0  1]
 [18  0  0  0  0  2  0  0]
 [14  0  0  0  0  1  0  0]
 [ 0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 8 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/8/weights.20-0.39.hdf5
------ TRAIN ACCURACY:  weights.20-0.39.hdf5  ------
0.669140625
[[236   0   3   0   0   0   0   1   0   0   0]
 [  1 125   1  12   0   0   0   1   0   0   0]
 [  0   0 195   0   0   0  11   5   0   0  39]
 [  0   0   0 215   0   0   1   0   1   1   2]
 [  7   0   8   0  67   0   0   0   0   0   8]
 [  3   0   0   0  12  55   1   0   1   0  18]
 [  0   0  18   0   0   0 161  47   1   0  13]
 [  0   0  49   0   0   0  30 139   2   0  25]
 [  0   0   5   0   0   0  12  17  98   5 183]
 [  3   0   2   0   0   0   5  10  33  90 217]
 [  0   0   2   0   0   0  11   6  12   2 332]]
------ TEST ACCURACY:  weights.20-0.39.hdf5  ------
0.578947368421
[[15  0  0  0  0  0  0  0]
 [ 0 10  0  0  0  0  2  3]
 [ 0  0 15  0  0  0  0  0]
 [ 0  0  0  2  0  0  0  8]
 [ 0  0  0  2  4  0  0  4]
 [ 0  2  0  0  0  6  6  1]
 [ 0  7  0  0  0  0  3  5]
 [ 0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 9 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/9/weights.35-0.40.hdf5
------ TRAIN ACCURACY:  weights.35-0.40.hdf5  ------
0.698181818182
[[233   2   1   0   0   0   2   1   0   1   0]
 [  0 123   0   1   0   0   0   1   0   0   0]
 [  0   0  76   0   0   0  43  54   0  73   4]
 [  0   0   0 218   1   0   0   0   0   1   0]
 [  0   0   0   0  91   1   0   0   0   8   0]
 [  0   0   0   0   0  88   0   0   0   9   3]
 [  0   0   0   0   0   0 228   5   0   4   3]
 [  0   0   2   0   0   0  84 122   1  35   1]
 [  0   1   0   0   0   0   0   4  83 186  16]
 [  0   1   0   0   0   0   3   2   2 317   5]
 [  0   0   0   0   0   0   0   0   6 180 149]]
------ TEST ACCURACY:  weights.35-0.40.hdf5  ------
0.516666666667
[[13  0  0  0  1  1  0  0  0]
 [ 0 15  0  0  0  0  0  0  0]
 [ 0  0  4  0  0  8  1  2  0]
 [ 0  8  0  7  0  0  0  0  0]
 [ 0  0  0  0 10  1  0  4  0]
 [ 0  0  0  0  6  7  0  2  0]
 [ 0  0  0  0  0  0  1 26  3]
 [ 0  0  0  0  0  0  1 28  1]
 [ 0  0  0  0  0  0  1 21  8]]

>>>>>>>>>>>>>> 10 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/10/weights.28-0.41.hdf5
------ TRAIN ACCURACY:  weights.28-0.41.hdf5  ------
0.663529411765
[[233   0   4   0   0   0   1   2   0   0   0]
 [  0 138   1   0   0   0   0   1   0   0   0]
 [  0   0 206   0   0   0  11   5   0  28   0]
 [  0   4   0 209   0   0   1   0   2   4   0]
 [  0   0   7   0  65   0   0   3   0  10   0]
 [  2   0   0   0   4  55   3   0   2  19   0]
 [  0   0  18   0   0   0 173  49   0   0   0]
 [  0   0  57   0   0   0  22 162   0   4   0]
 [  0   0  43   0   0   0  15  71  87 103   1]
 [  0   0  29   0   0   0   5  41   0 285   0]
 [  0   0  48   0   0   0  14  30  25 169  79]]
------ TEST ACCURACY:  weights.28-0.41.hdf5  ------
0.4
[[11  0  0  0  0  4  0  0]
 [ 0 13  0  0  0  0  0  2]
 [ 2  0  1 10  2  0  0  0]
 [ 0  4  0  6  0  3  2  0]
 [ 0  0  0 12  0  3  0  0]
 [ 0  5  0  0  0  6  4  0]
 [ 0  9  0  0  0  0  5  1]
 [ 0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 11 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/11/weights.17-0.46.hdf5
------ TRAIN ACCURACY:  weights.17-0.46.hdf5  ------
0.424657534247
[[ 57  91   0  30   0   0  35   5   1   5  16]
 [  0 119   0  19   0   0   0   0   0   1   1]
 [  0   0  14   0   0   0   0   0  80   0 151]
 [  0   0   0 214   0   0   0   0   1   1   4]
 [  0   1   0   9  27   3   0   0   7   2  41]
 [  0   1   0   7   0  44   0   0  16   2  20]
 [  0   0   0   0   0   0  19   5 111  23  82]
 [  0   0   0   0   0   0   1  29 126   8  81]
 [  0   0   0   0   0   0   0   0 243   6  71]
 [  0   0   0   0   0   0   0   0 192  93  75]
 [  0   0   0   0   0   0   0   0 136   3 226]]
------ TEST ACCURACY:  weights.17-0.46.hdf5  ------
0.33
[[12  0  0  0  0  0  0  1  0  2]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0 10 10]
 [ 0  0  0 15  0  0  0  0  0  0]
 [ 0  0  0  0  3  1  0  0  0  6]
 [ 0  1  0  0  1  3  1  0  0  4]
 [ 0  0  0  0  0  0  0  0 10  5]
 [ 0  0  0  0  0  0  0  0 13  2]
 [ 0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0]]

>>>>>>>>>>>>>> 12 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/12/weights.10-0.42.hdf5
------ TRAIN ACCURACY:  weights.10-0.42.hdf5  ------
0.492184368737
[[224  16   0   0   0   0   0   0   0   0   0]
 [  3 136   0   1   0   0   0   0   0   0   0]
 [ 51   0   8   0   0   0 127  59   0   0   0]
 [  1  19   0 192   0   2   2   0   1   1   2]
 [ 15   0   0   0  24  10  23   0   5   1   7]
 [  4   0   0   0   1  46   3   0  15   0  11]
 [ 10   0   0   0   0   0 222   8   0   0   0]
 [ 21   0   0   0   0   0 131  93   0   0   0]
 [  9   2   0   0   0   2 117  33 120   6  16]
 [  4   5   0   0   0   1 154  23  74  56  28]
 [ 10   1   0   0   0   0 142  28  57   5 107]]
------ TEST ACCURACY:  weights.10-0.42.hdf5  ------
0.34375
[[15  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 17  3  0  0  0]
 [ 0  0 15  0  0  0  0  0  0  0]
 [14  0  0  0  1  0  0  0  0  0]
 [ 0  0 12  0  3  5  0  0  0  0]
 [ 0  0  0  0  0 13  2  0  0  0]
 [ 0  0  0  0  0 14  1  0  0  0]
 [ 0  0  0  0  0  7  0  5  1  2]
 [ 0  0  0  0  0  9  0  5  0  1]
 [ 0  0  0  0  0  4  4  4  0  3]]

>>>>>>>>>>>>>> 13 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/13/weights.50-0.46.hdf5
------ TRAIN ACCURACY:  weights.50-0.46.hdf5  ------
0.5421875
[[163   0   2  72   8   0   0   0   0   0   0]
 [  0  77   1  57   0   0   0   0   0   0   0]
 [  0   0 237   0   0   0   5   0  12   0   6]
 [  0   0   0 230   0   0   0   0   0   0   0]
 [  0   0  16  59  25   0   0   0   0   0   0]
 [  0   0   3  80   0  13   0   1   3   0   0]
 [  0   0  47   7   1   0 112  79   4   0   0]
 [  0   0  66   1   1   0  10 174   2   0   1]
 [  0   0  41  19   0   0   2   5 206   5  22]
 [  0   0  53  71   0   0   4   2 152  53   5]
 [  0   0  60  64   0   0   0   7 115   1  98]]
------ TEST ACCURACY:  weights.50-0.46.hdf5  ------
0.273684210526
[[ 5  0  5  0  0  0  0  0  0]
 [ 0  1  0  4  0  0  0  0  0]
 [ 0  0  5  0  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0]
 [ 0  0  2  0  1  0  0  0  2]
 [ 0  0  4  0  0  1  0  0  0]
 [ 0  0 12  2  0  0  6  0  0]
 [ 0  0  0  3  0  0 13  1  3]
 [ 0  0 11  6  0  0  2  0  1]]

>>>>>>>>>>>>>> 14 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/14/weights.20-0.41.hdf5
------ TRAIN ACCURACY:  weights.20-0.41.hdf5  ------
0.537692307692
[[238   1   0   0   0   0   8   3   0   0   0]
 [  0 139   0   0   0   0   0   1   0   0   0]
 [  0   0  99   0   0   0  23 133   0   0   0]
 [  2  49   1 176   0   0   0   0   1   0   6]
 [ 19   1   0   0  25   0  36  16   1   0   2]
 [ 11   7   0   0   3  27  23   7   7   1  14]
 [  0   0   0   0   0   0 205  50   0   0   0]
 [  0   0  10   0   0   0  44 196   0   0   0]
 [  0   2   0   0   0   0  25 149 120   2  17]
 [  0   9   0   0   0   0  26 164  77  41  28]
 [  1   2   0   0   0   0  23 157  40   0 132]]
------ TEST ACCURACY:  weights.20-0.41.hdf5  ------
0.381818181818
[[0 5 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 8 0 2 0 0 0]
 [0 0 0 0 0 0 0 0]
 [0 0 1 0 9 0 0 0]
 [0 0 0 0 1 1 1 2]
 [0 2 0 1 2 3 0 7]
 [0 0 0 0 4 3 0 3]]

>>>>>>>>>>>>>> 15 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/15/weights.12-0.42.hdf5
------ TRAIN ACCURACY:  weights.12-0.42.hdf5  ------
0.661568627451
[[237   0   3   0   1   0   3   1   0   0   0]
 [  0 119   1  13   0   0   2   0   0   0   0]
 [  0   0 126   0   0   0  25  33   0  66  10]
 [  0   0   0 220   0   1   0   0   0   2   2]
 [  3   0   0   0  80   4   0   0   0  13   0]
 [  2   0   0   0   7  75   2   0   0  12   2]
 [  0   0   1   0   0   1 214  24   0   4   1]
 [  0   0  15   0   0   0  80 128   0  26   1]
 [  0   0   4   0   1   0  71   4  89 100  31]
 [  3   0   0   0   1   8  43   4  18 249  19]
 [  0   0   0   0   0   1  19   9  19 147 150]]
------ TEST ACCURACY:  weights.12-0.42.hdf5  ------
0.504761904762
[[ 5  0  0  0  1  4  0  0  0  0  0]
 [ 0  0  0  5  0  0  0  0  0  0  0]
 [ 0  0  2  0  0  0  0  3  0  0  0]
 [ 0  0  0 10  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0  0]
 [ 0  0  1  0  0  0  0  9  0  0  0]
 [ 0  0  0  0  0  0  3  0  4  6  7]
 [ 0  0  0  0  0  0  0  0  0 10  5]
 [ 0  0  0  0  0  0 10  0  7  0  3]]

>>>>>>>>>>>>>> 16 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/16/weights.23-0.46.hdf5
------ TRAIN ACCURACY:  weights.23-0.46.hdf5  ------
0.672506082725
[[210   0   4   0   0   0   0   1   0   0   0]
 [  0  86   1  12   0   0   0   1   0   0   0]
 [  0   0 170   0   0   0   2   0   0   0  48]
 [  0   0   0 193   0   0   0   0   0   0   2]
 [  2   0   2   0  48   0   0   0   0   0  13]
 [  2   0   0   2   8  41   1   0   1   1  14]
 [  0   0  19   0   0   0 154  20   1   0  16]
 [  0   0  34   0   0   0  18 112   2   4  45]
 [  0   0  44   0   0   0   0   1  60  13 117]
 [  0   0  22   0   0   0   3   1   6  76 152]
 [  0   0  25   0   0   0   2   3   3   5 232]]
------ TEST ACCURACY:  weights.23-0.46.hdf5  ------
0.483333333333
[[40  0  0  0  0  0  0  0  0  0  0]
 [ 0 40  0  0  0  0  0  0  0  0  0]
 [ 0  0 27  0  0  0  4  3  0  0 11]
 [ 1  0  0 39  0  0  0  0  0  0  0]
 [ 2  0  2  0 14  2  0  0  0  1 14]
 [ 0  0  0  3  2  7  0  0  0  0 18]
 [ 0  0  2  0  0  0 23 15  0  0  5]
 [ 0  0 12  0  0  0 14 10  0  0  9]
 [ 0  0  0  0  0  0  1  1  5  8 70]
 [ 3  0  0  0  0  0  3  2 14 19 59]
 [ 0  0 21  0  0  0  0  0  3  5 66]]

>>>>>>>>>>>>>> 17 -fold <<<<<<<<<<<<<<<<
========================================
model/umafall_smv_conv1d_10/17/weights.20-0.46.hdf5
------ TRAIN ACCURACY:  weights.20-0.46.hdf5  ------
0.672323232323
[[215   5   3   0   0   0  16   1   0   0   0]
 [  0 121   1   2   0   0   1   0   0   0   0]
 [  0   0 151   0   0   0  45  29   0   4  21]
 [  0   0   0 217   0   0   1   0   1   0   1]
 [  2   0   0   0  76   2   8   0   2   1   9]
 [  0   0   0   0   1  78   6   0   5   0  10]
 [  0   0   2   0   0   0 233   3   1   0   1]
 [  0   0  16   0   0   0 137  80   2   1   9]
 [  0   0   3   0   0   0  39   5 183  31  29]
 [  0   1   0   0   0   1  50   2  87 133  56]
 [  0   0   0   1   0   0  56   3  54  44 177]]
------ TEST ACCURACY:  weights.20-0.46.hdf5  ------
0.405555555556
[[15  0  0  0  0  0  0  0  0]
 [ 0  8  0  7  0  0  0  0  0]
 [ 0  0  4  0 11  0  0  0  0]
 [ 0  0  0 15  0  0  0  0  0]
 [ 0  0  0  0 15  0  0  0  0]
 [ 0  0  0  0 15  0  0  0  0]
 [ 0  0  0  0 11  0 11  3  5]
 [ 0  0  0  0 14  0  5  1 10]
 [ 0  0  0  0 14  0  8  4  4]]
[0.5368421052631579, 0.48, 0.5405405405405406, 0.5157894736842106, 0.6571428571428571, 0.32, 0.21818181818181817, 0.5789473684210527, 0.5166666666666667, 0.4, 0.33, 0.34375, 0.2736842105263158, 0.38181818181818183, 0.5047619047619047, 0.48333333333333334, 0.40555555555555556]
0.44041258917
0.114797737566

Process finished with exit code 0

'''
