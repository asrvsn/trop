from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch

def system(mu: float):
	return lambda t, z: [z[1], mu*(1-z[0]**2)*z[1] - z[0]]

def dataset(mu: float, a=0, b=10, n=500):
	t = linspace(a, b, n)
	sol = solve_ivp(system(mu), [a, b], [1, 0], t_eval=t)
	return torch.from_numpy(sol.y).float()

if __name__ == '__main__':
	mu = 3

	sol = dataset(mu, b=20, n=8000, skip=3500)
	X, Y = sol[:, :-1], sol[:, 1:]
	print(X.shape, Y.shape)
	plt.figure(figsize=(8,8))
	plt.plot(X[0], X[1])
	plt.plot([X[0][0]], [X[1][0]], marker='o', color='red')
	plt.show()