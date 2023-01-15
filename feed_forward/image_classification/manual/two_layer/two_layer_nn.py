import importlib

import matplotlib.pyplot as plt
import numpy as np

import feed_forward.image_classification.manual.dnn_utils as dnn_utils
from feed_forward.image_classification.model import Model


class Two_Layer_NN(Model):
    def __init__(self):
        self._eval = importlib.import_module(
            "feed_forward.image_classification.manual.eval.manual_eval"
        )

    @property
    def eval(self):
        return self._eval

    def predict(self, X: np.array, y: np.array, parameters: dict[str, np.array]):
        return self.eval.predict(X, y, parameters, dnn_utils.n_model_forward)

    def model(
        self,
        X: np.array,
        Y: np.array,
        layers_dims: list[int],
        learning_rate: float = 0.0075,
        num_iterations: int = 3000,
        print_cost: bool = False,
    ) -> dict[str, float]:
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterations

        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """

        np.random.seed(1)
        grads = {}
        costs = []  # to keep track of the cost
        _ = X.shape[1]  # number of examples
        (n_x, n_h, n_y) = layers_dims

        # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
        parameters = dnn_utils.initialize_parameters(n_x, n_h, n_y)

        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Loop (gradient descent)

        for i in range(0, num_iterations):

            # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
            A1, cache1 = dnn_utils.linear_activation_forward(X, W1, b1, "relu")
            A2, cache2 = dnn_utils.linear_activation_forward(A1, W2, b2, "sigmoid")

            # Compute cost
            cost = dnn_utils.compute_cost(A2, Y)

            # Initializing backward propagation
            dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

            # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
            dA1, dW2, db2 = dnn_utils.linear_activation_backward(dA2, cache2, "sigmoid")
            dA0, dW1, db1 = dnn_utils.linear_activation_backward(dA1, cache1, "relu")

            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
            grads["dW1"] = dW1
            grads["db1"] = db1
            grads["dW2"] = dW2
            grads["db2"] = db2

            # Update parameters.
            parameters = dnn_utils.update_parameters(parameters, grads, learning_rate)

            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {np.squeeze(cost)}")
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return parameters
