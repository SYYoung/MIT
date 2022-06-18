import pdb
import numpy as np
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features_std = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]


# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

auto_data_std, auto_labels_std = hw3.auto_data_and_labels(auto_data_all, features_std)
print('auto data std and labels std shape', auto_data_std.shape, auto_labels_std.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data
# added by sheena
k_fold = 10

all_data_set = {'raw': auto_data, 'std':auto_data_std}
alg_list = {'percept': hw3.perceptron, 'avg':hw3.averaged_perceptron}

'''
for data_set in all_data_set.keys():
    print('data set is: ', data_set)
    for method in alg_list.keys():
        acc = hw3.xval_learning_alg(alg_list[method], all_data_set[data_set], auto_labels, k_fold, {'T':10})
        print('algorith is: ', method)
        print('accuracy = ', str(acc))
print('Done')
'''

#acc = hw3.xval_learning_alg(hw3.perceptron, auto_data_std, auto_labels_std, k_fold, {'T':10})
#print('accuracy of percept= ', str(acc))
#acc = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data_std, auto_labels_std, k_fold, {'T':10})
#print('accuracy of avg percept = ', str(acc))

#th, th0 = hw3.perceptron(auto_data_std, auto_labels_std, {'T':10})
#print(th, th0)

#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data
k_fold = 10

alg_list = {'percept': hw3.perceptron, 'avg':hw3.averaged_perceptron}
T_list = [1, 10, 50]
#T_list = [1]

'''
for T in  T_list:
    print('T = ', T)
    for method in alg_list.keys():
        acc = hw3.xval_learning_alg(alg_list[method], review_bow_data, review_labels, k_fold, {'T':T})
        print('algorith is: ', method)
        print('accuracy = ', str(acc))
print('Done')
'''

#acc = hw3.xval_learning_alg(hw3.averaged_perceptron(), review_bow_data, review_labels, k_fold, {'T':10})
#print('accuracy of percept= ', str(acc))
#acc = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data_std, auto_labels_std, k_fold, {'T':10})
#print('accuracy of avg percept = ', str(acc))

'''
th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, {'T':10})
# get the most positive and most negative theta values
theta1 = (th[:, 0]).copy()
theta_order = np.argsort(theta1)
rev_dict = hw3.reverse_dict(dictionary)
most_pos_important = []
most_neg_important = []
for i in range(10):
    most_neg_important.append(rev_dict[theta_order[i]])
    most_pos_important.append(rev_dict[theta_order[-i-1]])

print('the most positive words are: ', most_pos_important)
print('the most negative words are: ', most_neg_important)
'''

#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
print(mnist_data_all.keys())
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples, m, n = x.shape
    x1 = x.reshape(n_samples, m*n)
    return x1.T

def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    n_samples, m, n = x.shape
    row_avg = np.mean(x, axis=1)
    return row_avg.T


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    n_samples, m, n = x.shape
    col_avg = np.mean(x, axis=0)
    return col_avg.T

def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m, n = x.shape
    top_image = x[0:m//2, :]
    bot_image = x[m//2:, :]
    top_avg = np.mean(top_image)
    bot_avg = np.mean(bot_image)

    return np.array([[top_avg], [bot_avg]])


# use this function to evaluate accuracy
# get 0 vs1, 2 vs 4, 6 vs 8, 9 vs 0

# for all 28x28 data
#acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)

#for row average:
method_list = {'row': row_average_features, 'col': col_average_features, 'top_bot': top_bottom_features}
for method in method_list.keys():
    acc = hw3.get_classification_accuracy(method_list[method](data), labels)
    print('method = ', method, ' accuracy = ', str(acc))
#acc = hw3.get_classification_accuracy(row_average_features(data), labels)
#print('accuracy = ', str(acc))

#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

