import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')


def funcError(u, v):
    t1 = np.float64(u*np.exp(v) - 2 * v * np.exp(-u))
    t2 = np.float64(t1**2)
    return t2

def grad_u(u, v):
    t1 = np.float64(u*np.exp(v) - 2 * v * np.exp(-u))
    t2 = np.float64(np.exp(v) + 2 * v * np.exp(-u))
    return 2*t1*t2

def grad_v(u, v):
    t1 = np.float64(u*np.exp(v) - 2 * v * np.exp(-u))
    t2 = np.float64(u*np.exp(v) - 2*np.exp(-u))
    return 2*t1*t2

def ans():
    u = np.float64(1.0)
    v = np.float64(1.0)
    eta = 0.1
    count = 0
    err = funcError(u, v)
    while (count < 10):
        count += 1
        u1 = u - eta * grad_u(u, v)
        v1 = v - eta * grad_v(u, v)
        u, v = u1, v1
        err = funcError(u, v)
        print('iter = ', str(count), ' , error = ', str(err))

#ans()

def getSample(numPoint):
    x1 = np.random.uniform(-1, 1, numPoint)
    x2 = np.random.uniform(-1, 1, numPoint)
    return x1, x2

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

def drawLine(slope, inter):
    xcoord = np.array([1, -1])
    pt = slope * xcoord + inter
    plt.plot(xcoord, pt)

N = 20
def test1(n):
    x1, x2 = getSample(n)
    m, c = takeLine()
    y = np.sign(x2 - (m * x1 + c))
    plt.scatter(x1, x2, c = y, cmap = 'Spectral')

    xcoord = np.array([1, -1])
    pt = m * xcoord + c
    plt.plot(xcoord, pt)
    plt.show()
    return x1, x2, y

def logregGrad(xn, yn, w):
    t1 = np.dot(xn, w) * (yn)
    numer = -1 * yn * xn
    denom = 1 + np.exp(t1)
    return numer/denom


def test2():
    n = 20
    x1, x2, Y = test1(n)
    x0 = np.ones((1, n))
    trainX = np.vstack((x0, x1.reshape((1, n)), x2.reshape((1, n))))
    m, c = takeLine()
    drawLine(m, c)

    #permutation
    newIndex = np.random.permutation(n)
    terminate = False
    eta = 0.1
    count = 0
    while (not terminate):
        w = np.zeros(3)
        count += 1
        for i in newIndex:
            grad = logregGrad(trainX[:,i], Y[i], w)
            w_new = w - eta * grad
            mag = np.sum((w_new - w)**2)
            if (mag < 0.0001):
                terminate = True
                break
            else:
                w = w_new
    print('count = ', str(count))

test2()