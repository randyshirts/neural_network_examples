import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Set layer sizes and other settings
n_input = 10
n_hidden = 15
n_out = 1
batch_size = 100
learning_rate = 0.01
num_iterations = 5000

# Create some random data for input
data_x = torch.randn(batch_size, n_input)
data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()
print(data_x.size())
print(data_y.size())

# Define the model
model = nn.Sequential(
    nn.Linear(n_input, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out), nn.Sigmoid()
)
print(model)

# Set the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_iterations):
    # Do a forward pass
    pred_y = model(data_x)

    # Calc the loss
    loss = loss_function(pred_y, data_y)
    losses.append(loss.item())

    # Set grads to zero each epoch and perform back-prop
    model.zero_grad()
    loss.backward()

    # Perform optimization and update the parameters
    optimizer.step()


plt.plot(losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Learning rate %f" % (learning_rate))
plt.show()
