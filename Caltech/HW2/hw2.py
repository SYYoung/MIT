import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')


def takeLine():
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    y1 = np.random.uniform(-1, 1)
    y2 = np.random.uniform(-1, 1)

    ydiff = y2 - y1
    xdiff = x2 - x1

    slope = ydiff / xdiff
    inter = (xdiff * y2 - ydiff * x2) / xdiff

    return slope, inter


def getY(slope, inter, x1, x2):
    est_x2 = slope * x1 + inter
    y = np.sign(x2 - est_x2)
    return y


def takeSample(numPoint):
    x1 = np.random.uniform(-1, 1, numPoint * 2)
    x = x1.reshape(numPoint, 2)
    temp = np.zeros((numPoint, 1)) + 1
    x = np.hstack((temp, x))
    return x


N = 10
def test1(m, c):
    x1 = np.random.uniform(-10, 10, N)
    x2 = np.random.uniform(-10, 10, N)
    y = np.sign(x2 - (m * x1 + c))
    return np.hstack((np.zeros((N, 1))+1, x1.reshape((N, 1)), x2.reshape((N, 1)))), y

def pred_y(w, x):
    pred = np.sign(np.transpose(x @ w))
    return pred

def perceptron(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 2

    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + eta * X[i] * Y[i]

    return w

corrupt = 0.1
corruptNum = int(corrupt * N)

def noiseY(Y):
    choice = np.random.randint(0, N, corruptNum)
    for i in range(corruptNum):
        Y[i] = Y[i] * -1
    return Y

def hw(noise):
    m, c = takeLine()
    trainx, trainy = test1(m, c)
    if noise:
        trainy = noiseY(trainy)
    w = perceptron(trainx, trainy)
    estY = pred_y(w, trainx)
    err = 1 - sum(estY == trainy) / N
    Ein = err

    testx, testy = test1(m, c)
    estY = pred_y(w, testx)
    err = 1 - sum(estY == testy) / N
    Eout = err

    return Ein, Eout

def Q1():
    for j in range(5):
        numIter = 1000
        totalEin, totalEout = 0, 0
        for i in range(numIter):
            ein, eout = hw(True)
            totalEin = totalEin + ein
            totalEout = totalEout + eout
        print('avgEin = ', str(totalEin/numIter), ', avgEout = ', str(totalEout/numIter))

def Q2():
    for j in range(10):
        numIter = 2
        totalEin, totalEout = 0, 0
        for i in range(numIter):
            ein, eout = hw(False)
            totalEin = totalEin + ein
            totalEout = totalEout + eout
        print('avgEin = ', str(totalEin/numIter), ', avgEout = ', str(totalEout/numIter))


Q2()