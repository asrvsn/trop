import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

def system(sigma, beta, rho):
	def f(t, z):
		u, v, w = z
		du = -sigma*(u-v)
		dv = rho*u - v - u*w
		dw = -beta*w + u*v
		return du, dv, dw
	return f

def dataset(a=0., b=100., n=1000, sigma=10, beta=2.667, rho=28):
	u0, v0, w0 = 0, 1, 1.05
	t = np.linspace(a, b, n)
	sol = solve_ivp(system(sigma, beta, rho), [a, b], (u0, v0, w0), t_eval=t)
	return torch.from_numpy(sol.y).float()

if __name__ == '__main__':
	n = 10000
	X = dataset(n=n)
	x, y, z = X[0].numpy(), X[1].numpy(), X[2].numpy()
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(x, y, z)
	ax.set_axis_off()
	plt.show()
