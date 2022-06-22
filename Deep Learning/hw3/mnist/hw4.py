import numpy as np


def gamma(X, Y, th, t0):
    '''
    X: 2xn
    Y: 1xn
    th: 2xn
    t0: scalar
    return gamma values of : 1xn
    '''
    norm = np.sqrt(np.dot(th.T, th))
    margin_dist = Y * ((th.T @ X) + t0) / norm
    return margin_dist

def cal_margin(X, Y, th, t0):
    margin = gamma(X, Y, th, t0)
    _, N = X.shape
    margin = margin.reshape(N)
    sum_margin = np.sum(margin)
    min_margin = min(margin)
    min_idx = np.argmin(margin)
    max_margin = max(margin)
    max_idx = np.argmax(margin)
    ans = "Sum of margin = {}, min of margin = {}, index = {}, max of margin = {}, index = {}".format\
          (sum_margin, min_margin, min_idx, max_margin, max_idx)
    print(ans)
    return margin

def q1():
    data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                     [1, 1, 2, 2, 2, 2, 2, 2]])
    labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
    blue_th = np.array([[0, 1]]).T
    blue_th0 = -1.5
    red_th = np.array([[1, 0]]).T
    red_th0 = -2.5
    print("blue separator")
    margin = cal_margin(data, labels, blue_th, blue_th0)
    print("red separator")
    margin = cal_margin(data, labels, red_th, red_th0)

def hinge_loss(X, Y, th, t0, gamma_ref):
    margin = cal_margin(X, Y, th, t0)
    loss = [1 - m/gamma_ref if m < gamma_ref else 0 for m in margin]
    L_h = np.array(loss)
    return L_h

def q3():
    data = np.array([[1.1, 1, 4], [3.1, 1, 2]])
    labels = np.array([[1, -1, -1]])
    th = np.array([[1, 1]]).T
    th0 = -4
    gamma_ref = np.sqrt(2)/2

    margin = cal_margin(data, labels, th, th0)
    Lh = hinge_loss(data, labels, th, th0, gamma_ref)
    print(Lh)

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def gd(f, df, x0, step_size_fn, max_iter):
    '''
    f: a function whose input is an x, a column vector, and returns a scalar
    df: a function whose input is an x, a column vector, and returns a column vector representing the
        gradient of f at x
    x0: an initial value of x, x0, which is a column vector
    step_size_fn: a funciton that is given the iteration index and returns a step size
    max_iter: the number of iterations to perform
    return: a tuple
    x: the value at the final step
    fs: the list of values of f found during all the iterations (including f(x0))
    xs: the list of values of x found during all the iterations (including x0)
    '''
#q1()
q3()

