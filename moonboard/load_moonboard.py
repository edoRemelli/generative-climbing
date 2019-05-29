import numpy as np
import sys

def load_moonboard(filename='moon2019.npz'):
    f = np.load(filename)
    
    
    x, y = f['x_train'], f['y_train']
    
    nproblems = len(y)
    randidx = np.random.randint(nproblems, size=nproblems)
    trainidx   = randidx[:int(4*nproblems/5)]
    testidx    = randidx[ int(4*nproblems/5):]
    x_train = np.take(x, trainidx, axis=0)
    y_train   = np.take(y  , trainidx, axis=0)
    x_test  = np.take(x, testidx, axis=0)
    y_test    = np.take(y  , testidx, axis=0)
    return (x_train, y_train), (x_test, y_test)

