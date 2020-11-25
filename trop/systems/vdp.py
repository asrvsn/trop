from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch

def system(mu: float):
	return lambda t, z: np.array([z[1], mu*(1-z[0]**2)*z[1] - z[0]])

def dataset(mu: float, t0=0, tf=10, n=500):
	t = linspace(t0, tf, n)
	y0 = np.array([1, 0])
	sol = solve_ivp(system(mu), [t0, tf], y0, t_eval=t)
	return torch.from_numpy(sol.y).float()

if __name__ == '__main__':
	mu = 3

	sol = dataset(mu, tf=20, n=8000, skip=3500)
	X, Y = sol[:, :-1], sol[:, 1:]
	print(X.shape, Y.shape)
	plt.figure(figsize=(8,8))
	plt.plot(X[0], X[1])
	plt.plot([X[0][0]], [X[1][0]], marker='o', color='red')
	plt.show()