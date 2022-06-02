import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions

class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        # TODO initialize model layers here
        new_input_dimension = int(input_dimension * 2/3)
        self.model = nn.Sequential(
                        nn.Linear(new_input_dimension, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
        )


    def forward(self, x):
        # print('size of x = ', x.size())
        xf = self.flatten(x)
        img_size = list(xf.size())
        str1 = "xf size = {}, xf.size[0] = {}, xf.size[1] = {}".format(xf.size(), img_size[0], img_size[1])
        # print(str1)

        # TODO use model layers to predict the two digits
        new_img_index = int(img_size[1] * 2/3)
        xf_slice = xf[:, 0:new_img_index]
        # print('xf_slice size = ', xf_slice.size())
        ans = self.model(xf_slice)
        out_first_digit = ans

        new_img_index = int(img_size[1] * 1/3)
        xf_slice = xf[:, new_img_index:]
        # print('second xf_slice size = ', xf_slice.size())
        ans = self.model(xf_slice)
        out_second_digit = ans

        return out_first_digit, out_second_digit

def main():
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = MLP(input_dimension) # TODO add proper layers to MLP class above

    # Train
    train_model(train_batches, dev_batches, model, n_epochs=30)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
