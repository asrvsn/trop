"""Model-predictive control of a Duffing oscillator
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools

import trop.operators as op
import trop.systems.duffing as duffing
from trop.features import *
from trop.utils import rmse
from trop.control.gradient_mpc import mpc_loop

# Initial conditions for simulation data
t_max = 5
n_data = 8000
n_init = 12
x0s = np.linspace(-2.0, 2.0, n_init)
y0s = np.linspace(-2.0, 2.0, n_init)
ics = list(itertools.product(x0s, y0s))

# Control inputs for simulation data
gamma = 0.5 # control influence parameter
inputs = np.linspace(-1., 1., 10) # 10 in range [-1, 1] (1-dimensional input applied to ydot(t))
inputs = [torch.full((1, n_data), u) for u in inputs] # constant control inputs 

# Generate data
system = lambda ic, u: duffing.dataset(t_max, n_data, x0=ic[0], y0=ic[1], gamma=gamma, u=lambda _: u[0][0]) # constant control input
batch_data, batch_inputs = op.gen_control_data(system, ics, inputs)

# Use 5-degree polynomial observable
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Solve Koopman op & observable-space control matrix
K, B = op.solve_with_control(batch_data, batch_inputs, obs=obs)

# Define reference signal & cost function
def reference(t):
	""" Step-function signal with `nstep` steps bounded in [`lo`, `hi`] """
	lo, hi = -.8, .8
	nstep = 4
	tlen = 25+1
	return torch.floor(t*nstep/tlen)*(hi-lo)/nstep + lo

def cost(t, x, u):
	""" Objective function on 1st dimension of duffing system """ 
	return ((x[0] - reference(t))**2).sum()

# Define plant
alpha, beta, delta = -1.0, 1.0, 0.3
def plant(t, z, u):
	"""Plant definition in scipy.integrate.ode format"""
	[x, y] = z
	xdot = y
	ydot = -delta*y - alpha*x - beta*(x**3) + gamma*u
	return [xdot, ydot]

# Run MPC
x0 = torch.Tensor([0., 0.])
dt = t_max / n_data
n_iter = 200
horizon = 50
hist_t, hist_u, hist_x = mpc_loop(plant, x0, dt, n_iter, K, B, obs, cost, horizon, umin=-1., umax=1., n_apply=5) # Run MPC with bounded control in [-1, 1]

# Plot results
fig, axs = plt.subplots(1, 2)
axs[0].plot(hist_t, hist_x[0], color='blue', label='plant')
axs[0].plot(hist_t, reference(hist_t), color='orange', label='reference')
axs[0].set_title(f'Plant history\nRMSE: {rmse(hist_x[0], reference(hist_t))}')
axs[1].plot(hist_t, hist_u)
axs[1].set_title('Control input')

plt.legend()
plt.show()