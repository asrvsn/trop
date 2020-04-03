"""Algorithms for computing transfer operators.
"""

import torch

from totorch.features import *

""" Snapshot generation """ 

def prep_snapshots(data: torch.Tensor, obs=None):
	"""Convert simulation data to snapshot matrices
	
	Args:
		data: either snapshot matrix or batch of snapshot matrices
			(a) d (observable dimension) x N (trajectory length)
			(b) b (# simulations) x d (observable dimension) x N (trajectory length)
		obs: Observable function.

	Returns:
		X, Y: snapshot matrices.

	Note: assumes dt is constant.
	"""
	if len(data.shape) == 3:
		if obs is not None:
			data = torch.stack([obs(x) for x in torch.unbind(data)])
		X, Y = data[:, :, :-1], Y[:, :, 1:]
		d, M = data.shape[1], data.shape[0]*data.shape[2]
		# stack snapshots
		X, Y = X.permute(1, 0, 2).reshape((d, M)), Y.permute(1, 0, 2).reshape((d, M)) 
		return X, Y
	else:
		assert len(data.shape) == 2, "Data must be snapshot matrix"
		if obs is not None:
			data = obs(data)
		X, Y = data[:, :-1], data[:, 1:]
		return X, Y


def gen_control_data(sys: Callable, ics: list, inputs: list):
	"""Generate trajectories given dynamical system with control inputs

	Args:
		sys: trajectory generator : (initial condition, input) -> trajectory
		ics: list of initial conditions
		inputs: list of control inputs 

	Returns:
		batch_data: b (# initial conditions * # control inputs) x d (state dimension) x N (trajectory length)
	"""
	batch_data = []
	for i in range(len(ics)):
		for j in range(len(inputs)):
			x = sys(ics[i], inputs[j])
			batch_data.append(x)
	return torch.stack(batch_data)


""" Transfer operators """

def solve(data: torch.Tensor, koopman=True, obs=None):
	"""Pseudoinverse solution for Koopman & Perron-Frobenius operators
	
	Args:
		data: simulation data (see `prep_snapshots`)
		koopman: if True, Koopman operator, else Perron-Frobenius operator.
		obs: Observable function.

	Returns:
		L: Koopman or Perron-Frobenius operator.

	Note: can be used either for vanilla DMD or extended DMD, simply pass desired observable.
	Based on https://arxiv.org/pdf/1712.01572
	"""
	X, Y = prep_snapshots(data, obs=obs)
	if koopman:
		return Y@X.t()@torch.pinverse(X@X.t())
	else:
		return X@Y.t()@torch.pinverse(X@X.t())

def solve_kernel(data: torch.Tensor, kernel: Kernel, koopman=True):
	"""Pseudoinverse solution for Koopman & Perron-Frobenius operators over RKHS
	
	Args:
		data: simulation data (see `prep_snapshots`)
		kernel: positive-definite kernel.
		koopman: if True, Koopman operator, else Perron-Frobenius operator.

	Returns:
		L: Koopman or Perron-Frobenius operator.

	Based on https://arxiv.org/pdf/1712.01572
	"""
	X, Y = prep_snapshots(data)
	G_XX = kernel.gramian(X, X)
	G_YX = kernel.gramian(Y, X)
	if koopman:
		return G_YX@torch.pinverse(G_XX)
	else:
		return G_YX.t()@torch.pinverse(G_XX)

def solve_with_control(batch_data: torch.Tensor, inputs: torch.Tensor, obs=None):
	"""Pseudoinverse solution for Koopman operator w/ control influence matrix.

	Args:
		batch_data: trajectory data of dimension b (# simulations) x d (observable dimension) x N (trajectory length)
		inputs: control input matrix of dimension b (# simulations) x N (control input over trajectory length)
		obs: Observable function. 

	Returns:
		K: Koopman operator for uncontrolled system
		B: Control influence matrix in observable space

	Note: can be used either for vanilla DMD or extended DMD, simply pass desired observable. Cannot be used for kernel DMD.
	Based on https://arxiv.org/abs/1611.03537
	"""
	assert batch_data.shape[0] == inputs.shape[0] and batch_data.shape[2] == inputs.shape[1], "Trajectory and control input dimensions must match"

	X, Y = prep_snapshots(batch_data, obs=obs)
	U = inputs[:1].reshape((U.shape[0]*(U.shape[1]-1),))
	Xu = torch.cat((X, U), axis=1)
	
	d = X.shape[0]
	L = Y@Xu.t()@torch.pinverse(Xu@Xu.t())
	K, B = L[:, :d], L[:, d:]

	return K, B



