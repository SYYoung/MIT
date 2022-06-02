import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    C_factor = 1
    print('C_factor = ', C_factor)
    clf = LinearSVC(random_state=0, C=C_factor)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    C_factor = 0.1
    print('In multi_class_svm: C_factor = ', C_factor)
    clf = LinearSVC(random_state=0, C=C_factor, multi_class='ovr')
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

