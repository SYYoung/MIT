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

def hinge_loss_old(X, Y, th, t0, gamma_ref):
    margin = cal_margin(X, Y, th, t0)
    loss = [ max(0, 1 - m/gamma_ref) for m in margin]
    #loss = [1 - m/gamma_ref if m < gamma_ref else 0 for m in margin]
    L_h = np.array(loss)
    return L_h

def hinge(v):
    val = np.where(v<1, 1 - v, 0)
    return val

def hinge_loss(x, y, th, th0):
    margin = y * ((th.T @ x) + th0)
    #margin = gamma(X, Y, th, t0)
    #loss = [hinge(m) for m in margin]
    loss = hinge(margin)
    L_h = np.array(loss)
    return L_h

def q3():
    data = np.array([[1.1, 1, 4], [3.1, 1, 2]])
    labels = np.array([[1, -1, -1]])
    th = np.array([[1, 1]]).T
    th0 = -4
    gamma_ref = np.sqrt(2)/2

    margin = cal_margin(data, labels, th, th0)
    Lh = hinge_loss_old(data, labels, th, th0, gamma_ref)
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
    xs = []
    fs = []
    xs.append(x0)
    fs.append(f(x0))
    for iter in range(max_iter):
        current_x = xs[-1]
        new_x = current_x - step_size_fn(iter) * df(current_x)
        xs.append(new_x)
        fs.append(f(new_x))
    return (new_x, fs, xs)

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])


def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

def q6():
    # Test case 1
    ans=package_ans(gd(f1, df1, cv([0.]), lambda i: 0.1, 10))
    print(ans)

    # Test case 2
    #ans=package_ans(gd(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))

def num_grad(f, delta=0.001):
    def df(x):
        d, n = x.shape
        df_v = np.zeros((d, n))
        for i in range(d):
            delta_v = np.zeros((d, n))
            delta_v[i, :] = delta_v[i,:] + delta
            val = (f(x + delta_v) - f(x-delta_v)) / (2 * delta)
            df_v[i, :] = val
        return df_v
    return df

def minimize_old(f, x0, step_size_fn, max_iter):
    '''
    f: a function whose input is an x, a column vector, and returns a scalar
    x0: an initial value of x, x0, which is a column vector
    step_size_fn: a funciton that is given the iteration index and returns a step size
    max_iter: the number of iterations to perform
    return: a tuple
    x: the value at the final step
    fs: the list of values of f found during all the iterations (including f(x0))
    xs: the list of values of x found during all the iterations (including x0)
    '''
    xs = []
    fs = []
    xs.append(x0)
    fs.append(f(x0))
    for iter in range(max_iter):
        current_x = xs[-1]
        new_x = current_x - step_size_fn(iter) * num_grad(f)(current_x)
        xs.append(new_x)
        fs.append(f(new_x))
    return (new_x, fs, xs)

def minimize(f, x0, step_size_fn, max_iter):
    '''
    f: a function whose input is an x, a column vector, and returns a scalar
    x0: an initial value of x, x0, which is a column vector
    step_size_fn: a funciton that is given the iteration index and returns a step size
    max_iter: the number of iterations to perform
    return: a tuple
    x: the value at the final step
    fs: the list of values of f found during all the iterations (including f(x0))
    xs: the list of values of x found during all the iterations (including x0)
    '''
    return gd(f, num_grad(f), x0, step_size_fn, max_iter)

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def svm_obj(x, y, th, th0, lam):
    d, n = x.shape
    Lh = hinge_loss(x, y, th, th0).reshape(n)
    norm_sq = float(np.dot(th.T, th))
    val = np.mean(Lh) + lam * norm_sq
    return val

def q7_1():
    sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

    # Test case 1
    x_1, y_1 = super_simple_separable()
    th1, th1_0 = sep_e_separator
    ans = svm_obj(x_1, y_1, th1, th1_0, .1)
    print("in q7_1(), ans = ", str(ans))

    # Test case 2
    ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
    print("in q7_1(), ans = ", str(ans))

def q6_2():
    x = cv([0.])
    ans = (num_grad(f1)(x).tolist(), x.tolist())

    x = cv([0.1])
    ans = (num_grad(f1)(x).tolist(), x.tolist())

    x = cv([0., 0.])
    ans = (num_grad(f2)(x).tolist(), x.tolist())

    x = cv([0.1, -0.1])
    ans = (num_grad(f2)(x).tolist(), x.tolist())

def q6_3():
    ans = package_ans(minimize(f1, cv([0.]), lambda i: 0.1, 1000))
    print(ans)
    ans = package_ans(minimize(f2, cv([0., 0.]), lambda i: 0.01, 1000))
    print(ans)

q6_3()
q1()
q3()
q6_2()
q7_1()

