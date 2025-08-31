# Logistic regression

import torch
import torch.nn as nn
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

sugar = datasets.load_breast_cancer()
X, y = sugar.data, sugar.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class Model(nn.Module):
  def __init__(self, n_features):
    super(Model, self).__init__()
    self.linear = nn.Linear(n_features, 1)

  def forward(self, x):
    y_pred = torch.sigmoid(self.linear(x))
    return y_pred

model = Model(n_features)

lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

#Training
num_epochs = 10
for epoch in range(num_epochs):
  y_pred = model(X_train)
  loss = criterion(y_pred, y_train)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  if(epoch+1) % 1 == 0:
    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#Testing
with torch.no_grad():
  y_predicted = model(X_test)
  y_predicteed_cls = y_predicted.round()
  acc = y_predicteed_cls.eq(y_test).sum() / float(y_test.shape[0])
  print(f'accuracy: {acc.item():.4f}')