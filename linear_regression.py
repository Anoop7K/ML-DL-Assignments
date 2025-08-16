# Linear Regression

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(y_numpy, dtype=torch.float32)


# Reshape y to be a 2D tensor with a single column
y = y.view(y.shape[0], 1)

class Model(nn.Module):
  def __init__(self, n_features):
    super(Model, self).__init__()
    self.linear = nn.Linear(n_features, 1)

  def forward(self, x):
    y_pred = self.linear(x)
    return y_pred

n_samples, n_features = X.shape

model = Model(n_features)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass and update
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch) % 1 == 0:
        print(f"Epoch: {epoch + 1}, Loss = {loss.item():.4f}")

# Plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro', label='Original data')
plt.plot(X_numpy, predicted, 'b', label='Fitted line')
plt.xlabel('Age')
plt.ylabel('Outcome')
plt.legend()
plt.show()
