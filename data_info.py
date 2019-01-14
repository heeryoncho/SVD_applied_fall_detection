import pandas as pd
import cPickle as pickle

X = pickle.load(open("unimib/data/X_unimib_smv.p", "rb"))
y = pickle.load(open("unimib/data/y_unimib_smv.p", "rb"))
y = pd.DataFrame(y)

print "X_unimib:", X.shape
print "y_unimib:", y.shape


X = pickle.load(open("sisfall/data/X_sisfall_smv.p", "rb"))
y = pickle.load(open("sisfall/data/y_sisfall_smv.p", "rb"))
y = pd.DataFrame(y)

print "X_sisfall:", X.shape
print "y_sisfall:", y.shape


X = pickle.load(open("umafall/data/X_umafall_smv.p", "rb"))
y = pickle.load(open("umafall/data/y_umafall_smv.p", "rb"))
y = pd.DataFrame(y)

print "X_umafall:", X.shape
print "y_umafall:", y.shape

'''

/usr/bin/python2.7 /code/data_info.py
X_unimib: (11771, 151)
y_unimib: (11771, 3)
X_sisfall: (3900, 450)
y_sisfall: (3900, 2)
X_umafall: (2655, 450)
y_umafall: (2655, 2)

Process finished with exit code 0

'''
