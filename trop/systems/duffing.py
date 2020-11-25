import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pdb

def system(alpha, beta, gamma, delta, u):
	def f(t, y):
		dydt = np.zeros_like(y)
		dydt[0] = y[1]
		dydt[1] = -delta*y[1] - alpha*y[0] - beta*(y[0]**3) + gamma*u(t)
		return dydt
	return f

def dataset(t0: float=0., tf: float=400, n=8000, alpha=-1.0, beta=1.0, gamma=0.5, delta=0.3, x0=-1.0, y0=2.0, u=lambda t:0.):
	"""Duffing oscillator 
	
	Args:
		tmax: # seconds 
		n: # data points (dt = tmax / n)
		x0: initial condition
		y0: initial condition
		u: control signal (Callable : time -> float)
	"""
	t = np.linspace(t0, tf, n)
	sol = solve_ivp(system(alpha, beta, gamma, delta, u), [t0, tf], np.array([x0, y0]), t_eval=t)
	return torch.from_numpy(sol.y).float()

if __name__ == '__main__': 
	X = dataset(gamma=0.0)
	plt.figure(figsize=(8,8))
	plt.title('Unforced')
	plt.plot(X[0], X[1])

	# X = dataset(t, n, gamma=0.2)
	# plt.figure(figsize=(8,8))
	# plt.title('Forced with period-1 oscillation')
	# plt.plot(X[0], X[1])

	# X = dataset(t, n, gamma=0.28)
	# plt.figure(figsize=(8,8))
	# plt.title('Forced with period-2 oscillation')
	# plt.plot(X[0], X[1])

	# X = dataset(t, n, gamma=0.29)
	# plt.figure(figsize=(8,8))
	# plt.title('Forced with period-4 oscillation')
	# plt.plot(X[0], X[1])

	# X = dataset(t, n, gamma=0.37)
	# plt.figure(figsize=(8,8))
	# plt.title('Forced with period-5 oscillation')
	# plt.plot(X[0], X[1])

	# X = dataset(t, n, gamma=0.50)
	# plt.figure(figsize=(8,8))
	# plt.title('Forced with chaos')
	# plt.plot(X[0], X[1])

	# X = dataset(t, n, gamma=0.65)
	# plt.figure(figsize=(8,8))
	# plt.title('Forced with period-2 oscillation (2)')
	# plt.plot(X[0], X[1])

	# u = lambda t: np.sign(np.cos(np.pi*t/0.3))
	# X = dataset(t, n, gamma=0.5, u=u)
	# fig, axs = plt.subplots(2, 1, figsize=(12,8))
	# fig.suptitle('Forced with square wave')
	# axs[0].plot(X[0], X[1])
	# axs[1].plot(taxis, [u(t) for t in taxis])

	plt.show()
