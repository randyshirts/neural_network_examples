import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data_loader.data_loader import flatten_image_data, load_data


def _load_h5_file_with_data(file_name):
    """Method for loading .h5 files

    :returns: dict that contains name of the .h5 file as stored in the .h5 file, as well as a generator of the data
    """
    path = os.path.join("../datasets/", file_name)
    data_file = h5py.File(path)
    x_key = list(data_file.keys())[1]
    data_x = data_file[x_key]
    y_key = list(data_file.keys())[2]
    data_y = data_file[y_key]
    return (data_file, data_x, data_y)


def torch_model():

    # Set layer sizes and other settings
    n_input = 12288
    n_hidden = 7
    n_out = 1
    learning_rate = 0.008
    num_iterations = 2500

    # Get the data
    train_x, train_y, test_x, test_y, classes = load_data(
        train_file="datasets/train_catvnoncat.h5",
        test_file="datasets/test_catvnoncat.h5",
    )
    train_x = flatten_image_data(train_x)
    test_x = flatten_image_data(test_x)

    th_train_x = torch.from_numpy(train_x).float().T
    th_train_y = torch.from_numpy(train_y).float().T
    th_test_x = torch.from_numpy(test_x).float().T
    th_test_y = torch.from_numpy(test_y).float().T

    # Define the model
    model = nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_out),
        nn.Sigmoid(),
    )
    print(model)

    # Set the loss function and optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(num_iterations):
        # Do a forward pass
        pred_y = model(th_train_x)

        # Calc the loss
        loss = loss_function(pred_y, th_train_y)
        losses.append(loss.item())

        # Perform back-prop
        # Todo: Need to initialize grads like the two-layer model - occasionaly get exploding gradients and when we don't we get overfitting
        model.zero_grad()
        loss.backward()

        # Perform optimization and update the parameters
        optimizer.step()

    # Evaluation
    with torch.no_grad():
        model.eval()
        pred_y = model(th_train_x)
        print("\nTraining eval:")
        _ = criterion(pred_y, th_train_y, th_train_x)

        model.eval()
        pred_y = model(th_test_x)
        print("\nTest eval:")
        _ = criterion(pred_y, th_test_y, th_test_x)

    plt.plot(losses)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("Learning rate %f" % (learning_rate))
    plt.show()


def criterion(probas, Y, X):
    m = X.shape[0]
    p = np.zeros((m, 1))

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[0]):
        if probas[i, 0] > 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0

    # print results
    y_np = Y.numpy().astype(np.float64)
    # print ("correct predictions: " + str(np.equal(p, y_np)))
    # print ("true labels: " + str(Y))
    print("Accuracy: " + str(np.sum((np.equal(p, y_np)) / m)))
    return p


if __name__ == "__main__":
    # matplotlib.use('TkAgg')
    torch_model()
