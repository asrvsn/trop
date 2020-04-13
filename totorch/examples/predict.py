""" Predict & extrapolate Duffing oscillator from observations
"""
import torch
import matplotlib.pyplot as plt

import totorch.systems.duffing as duffing
import totorch.operators as op
from totorch.features import *
from totorch.predict import *
from totorch.utils import rmse

# Sample trajectory for unforced Duffing oscillator
t_max = 80
n_data = 4000
X = duffing.dataset(t_max, n_data, gamma=0.0, x0=2.0, y0=2.0)
print(X.shape)

# Use 5-degree polynomial observable
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Solve Koopman operator
K = op.solve(X, obs=obs)

# Compare predictions
Y = X[:, 1:]
Y_pred = predict(X[:, :-1], K, obs)
Y_extr = extrapolate(X[:,0], K, obs, n_data-1)[:, 1:]

print(Y.shape, Y_pred.shape, Y_extr.shape)

fig, axs = plt.subplots(1,3)
axs[0].plot(Y[0], Y[1])
axs[0].set_title('Ground truth')
axs[1].plot(Y_pred[0], Y_pred[1])
axs[1].set_title(f'1-step prediction\nRMSE:{rmse(Y, Y_pred)}')
axs[2].plot(Y_extr[0], Y_extr[1])
axs[2].set_title(f'N-step extrapolation\nRMSE:{rmse(Y, Y_extr)}')

plt.show()