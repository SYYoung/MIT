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

# Returns the gradient of hinge(v) with respect to v.
def d_hinge(v):
    val = np.where(v < 1, -1, 0)
    return val

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th
def d_hinge_loss_th(x, y, th, th0):
    margin = y * ((th.T @ x) + th0)
    loss = d_hinge(margin)
    loss = loss * y * x
    return loss

# Returns the gradient of hinge_loss(x, y, th, th0) with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    margin = y * ((th.T @ x) + th0)
    loss = d_hinge(margin)
    loss = loss * y
    return loss

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    dLh_dth = d_hinge_loss_th(x, y, th, th0)
    val = np.mean(dLh_dth) + 2 * lam * th
    return val

# Returns the gradient of svm_obj(x, y, th, th0) with respect to th0
def d_svm_obj_th0(x, y, th, th0, lam):
    dLh_dt0 = d_hinge_loss_th0(x, y, th, th0)
    val = np.mean(dLh_dt0)
    return val

# Returns the full gradient as a single vector (which includes both th, th0)
def svm_obj_grad(X, y, th, th0, lam):
    dsvm_dth = d_svm_obj_th(X, y, th, th0, lam)
    dsvm_dt0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack((dsvm_dth, dsvm_dt0))

def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    pass


def q7_2():
    X1 = np.array([[1, 2, 3, 9, 10]])
    y1 = np.array([[1, 1, 1, -1, -1]])
    th1, th10 = np.array([[-0.31202807]]), np.array([[1.834]])
    X2 = np.array([[2, 3, 9, 12],
                   [5, 2, 6, 5]])
    y2 = np.array([[1, -1, 1, -1]])
    th2, th20 = np.array([[-3., 15.]]).T, np.array([[2.]])

    d_hinge(np.array([[71.]])).tolist()
    d_hinge(np.array([[-23.]])).tolist()
    d_hinge(np.array([[71, -23.]])).tolist()

    d_hinge_loss_th(X2[:, 0:1], y2[:, 0:1], th2, th20).tolist()
    d_hinge_loss_th(X2, y2, th2, th20).tolist()
    d_hinge_loss_th0(X2[:, 0:1], y2[:, 0:1], th2, th20).tolist()
    d_hinge_loss_th0(X2, y2, th2, th20).tolist()

    d_svm_obj_th(X2[:, 0:1], y2[:, 0:1], th2, th20, 0.01).tolist()
    d_svm_obj_th(X2, y2, th2, th20, 0.01).tolist()
    d_svm_obj_th0(X2[:, 0:1], y2[:, 0:1], th2, th20, 0.01).tolist()
    d_svm_obj_th0(X2, y2, th2, th20, 0.01).tolist()

    ans = svm_obj_grad(X2, y2, th2, th20, 0.01).tolist()
    print(ans)
    ans = svm_obj_grad(X2[:, 0:1], y2[:, 0:1], th2, th20, 0.01).tolist()
    print(ans)

def separable_medium():
    X = np.array([[2, -1, 1, 1],
                  [-2, 2, 2, -1]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def q7_3():
    sep_m_separator = np.array([[ 2.69231855], [ 0.67624906]]), np.array([[-3.02402521]])

    x_1, y_1 = super_simple_separable()
    ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))

    x_1, y_1 = separable_medium()
    ans = package_ans(batch_svm_min(x_1, y_1, 0.0001))

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

print("q6_3()")
q6_3()
print("q1()")
q1()
print("q3()")
q3()
print("q6_2()")
q6_2()
print("q7_1()")
q7_1()
print("q7_2()")
q7_2()

