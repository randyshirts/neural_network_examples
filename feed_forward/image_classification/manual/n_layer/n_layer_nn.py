import importlib

import matplotlib.pyplot as plt
import numpy as np

import feed_forward.image_classification.manual.dnn_utils as dnn_utils
from feed_forward.image_classification.model import Model


class N_Layer_NN(Model):
    def __init__(self):
        self._eval = importlib.import_module(
            "feed_forward.image_classification.manual.eval.manual_eval"
        )

    @property
    def eval(self):
        return self._eval

    def predict(
        self, X: np.array, y: np.array, parameters: dict[str, np.array]
    ) -> np.array:
        """
        This function is used to predict the results of a  N-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        y -- labels for the given data set
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """
        return self.eval.predict(X, y, parameters, dnn_utils.n_model_forward)

    def initialize_parameters_deep(self, layer_dims: list) -> dict:
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(1)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for layer in range(1, L):
            parameters["W" + str(layer)] = np.random.randn(
                layer_dims[layer], layer_dims[layer - 1]
            ) / np.sqrt(
                layer_dims[layer - 1]
            )  # *0.01
            parameters["b" + str(layer)] = np.zeros((layer_dims[layer], 1))

            assert parameters["W" + str(layer)].shape == (
                layer_dims[layer],
                layer_dims[layer - 1],
            )
            assert parameters["b" + str(layer)].shape == (layer_dims[layer], 1)

        return parameters

    def n_model_backward(self, AL, Y, caches) -> dict[str, float]:
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        _ = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        (
            grads["dA" + str(L)],
            grads["dW" + str(L)],
            grads["db" + str(L)],
        ) = dnn_utils.linear_activation_backward(
            dAL, current_cache, activation="sigmoid"
        )

        for layer in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[layer]
            dA_prev_temp, dW_temp, db_temp = dnn_utils.linear_activation_backward(
                grads["dA" + str(layer + 2)], current_cache, activation="relu"
            )
            grads["dA" + str(layer + 1)] = dA_prev_temp
            grads["dW" + str(layer + 1)] = dW_temp
            grads["db" + str(layer + 1)] = db_temp

        return grads

    def model(
        self,
        X: np.array,
        Y: np.array,
        layers_dims: list[int],
        learning_rate: float = 0.0075,
        num_iterations: int = 3000,
        print_cost: bool = False,
    ) -> dict[str, float]:  # lr was 0.009
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(1)
        costs = []  # keep track of cost

        # Parameters initialization.
        parameters = self.initialize_parameters_deep(layers_dims)

        # Loop (gradient descent)
        for i in range(0, num_iterations):

            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = dnn_utils.n_model_forward(X, parameters)

            # Compute cost.
            cost = dnn_utils.compute_cost(AL, Y)

            # Backward propagation.
            grads = self.n_model_backward(AL, Y, caches)

            # Update parameters.
            parameters = dnn_utils.update_parameters(parameters, grads, learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        return parameters
