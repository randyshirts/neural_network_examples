import numpy as np
import torch
import torch.nn as nn

from feed_forward.image_classification.model import Model


class Torch_Two_Layer(Model):
    def __init__(self):
        self._eval = self
        self.th_model = None

    @property
    def eval(self):
        return self._eval

    def model(
        self,
        X: np.array,
        Y: np.array,
        layers_dims: list[int],
        learning_rate: float = 0.0075,
        num_iterations: int = 3000,
        print_cost: bool = False,
    ) -> dict[str, float]:

        n_input = layers_dims[0]
        n_hidden = layers_dims[1]
        n_out = layers_dims[2]
        th_train_x = torch.from_numpy(X).float().T
        th_train_y = torch.from_numpy(Y).float().T

        # Define the model
        self.th_model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
            nn.Sigmoid(),
        )
        print(self.th_model)

        # Set the loss function and optimizer
        loss_function = nn.BCELoss()
        optimizer = torch.optim.SGD(self.th_model.parameters(), lr=learning_rate)

        # Training loop
        losses = []
        for epoch in range(num_iterations):
            # Do a forward pass
            pred_y = self.th_model(th_train_x)

            # Calc the loss
            loss = loss_function(pred_y, th_train_y)
            losses.append(loss.item())

            # Perform back-prop
            # Todo: Need to initialize grads like the two-layer model - occasionaly get exploding gradients
            #    and when we don't we get overfitting
            self.th_model.zero_grad()
            loss.backward()

            # Perform optimization and update the parameters
            optimizer.step()

        return self.th_model.state_dict()

    def predict(
        self, X: np.array, y: np.array, parameters: dict[str, np.array]
    ) -> np.array:
        th_x = torch.from_numpy(X).float().T
        th_y = torch.from_numpy(y).float().T
        with torch.no_grad():
            self.th_model.eval()
            pred_y = self.th_model(th_x)
            _ = self.criterion(pred_y, th_y, th_x)
        return pred_y

    def criterion(self, probas, Y, X):
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

    def sample_external_image(
        self, num_pixels: int, parameters: list, classifications: list, model: object
    ) -> None:
        pass


if __name__ == "__main__":
    model = Torch_Two_Layer()
    print(model)
