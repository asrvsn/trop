"""Algorithms for computing transfer operators.
"""

import torch
from typing import Callable, List

from trop.features import *

class Koopman:
	''' Intended for use in numpy land ''' 
	def __init__(self, K: np.ndarray, obs: Observable, transpose=False):
		self._K = K
		self.K_torch = torch.from_numpy(K)
		self.obs = obs
		self.transpose = transpose
		self.offset0 = np.zeros(K.shape)
		self.offset = self.offset0

	def __matmul__(self, x: np.ndarray):
		if self.transpose:
			return self.obs.call_numpy(self.obs.preimage((self.K.T@x).T)).T
		else:
			return self.obs.call_numpy(self.obs.preimage(self.K@x))

	def __rmatmul__(self, x: np.ndarray):
		print('called!')
		if self.transpose:
			return self.obs.call_numpy(self.obs.preimage(x@self.K.T))
		else:
			return self.obs.call_numpy(self.obs.preimage((x@self.K).T)).T

	def torch_dot(self, x: torch.Tensor):
		return self.obs(self.obs.preimage(self.K_torch@x))

	@property
	def T(self):
		return Koopman(self.K, self.obs, transpose=(not self.transpose))

	@property
	def K(self):
		return self._K + self.offset

	def set_offset(self, offset: np.ndarray):
		self.offset = offset

	def zero_offset(self):
		self.offset = self.offset0

	def re_project(self, x: np.ndarray):
		return self.obs.call_numpy(self.obs.preimage(x))

""" Snapshot generation """ 

def prep_snapshots(data: torch.Tensor, obs: Observable = None):
	"""Convert simulation data to snapshot matrices
	
	Args:
		data: either snapshot matrix or batch of snapshot matrices
			(a) d (observable dimension) x N (trajectory length)
			(b) b (# simulations) x d (observable dimension) x N (trajectory length)
		obs: Observable function. If not provided, assumed as identity.

	Returns:
		X, Y: snapshot matrices.

	Note: assumes dt is constant.
	"""
	if len(data.shape) == 3:
		if obs is not None:
			data = torch.stack([obs(x) for x in torch.unbind(data)])
		X, Y = data[:, :, :-1], data[:, :, 1:]
		d, M = data.shape[1], data.shape[0]*(data.shape[2]-1)
		X, Y = X.permute(1, 0, 2).reshape((d, M)), Y.permute(1, 0, 2).reshape((d, M)) # stack snapshots along second axis
		return X, Y
	else:
		assert len(data.shape) == 2, "Data must be snapshot matrix"
		if obs is not None:
			data = obs(data)
		X, Y = data[:, :-1], data[:, 1:]
		return X, Y


def gen_control_data(system: Callable, ics: List, inputs: List):
	"""Generate trajectories given dynamical system with control inputs

	Args:
		system: trajectory generator : (initial condition, input) -> trajectory
		ics: list of initial conditions
		inputs: list of control inputs 

	Returns:
		batch_data: b (# initial conditions * # control inputs) x d (state dimension) x N (trajectory length)
		batch_inputs b (# initial conditions * # control inputs) x d_u (control input dimension) x N (trajectory length)
	"""
	batch_data = []
	batch_inputs = []
	for u in inputs:
		for ic in ics:
			batch_data.append(system(ic, u))
			batch_inputs.append(u) 
	return torch.stack(batch_data), torch.stack(batch_inputs)


""" Transfer operators """

def solve(data: torch.Tensor, koopman: bool = True, obs: Observable = None):
	"""Pseudoinverse solution for Koopman & Perron-Frobenius operators
	
	Args:
		data: simulation data (see `prep_snapshots`)
		koopman: (optional) if True, Koopman operator, else Perron-Frobenius operator.
		obs: (optional) Observable function. If not provided, identity.

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

def solve_kernel(data: torch.Tensor, kernel: Kernel, koopman: bool = True):
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

def solve_with_control(batch_data: torch.Tensor, batch_inputs: torch.Tensor, obs: Observable = None):
	"""Pseudoinverse solution for Koopman operator w/ control influence matrix.

	Args:
		batch_data: trajectory data of dimension b (# simulations) x d (observable dimension) x N (trajectory length)
		batch_inputs: control input matrix of dimension b (# simulations) x N (control input over trajectory length)
		obs: Observable function. 

	Returns:
		K: Koopman operator for uncontrolled system
		B: Control influence matrix in observable space

	Note: can be used either for vanilla DMD or extended DMD, simply pass desired observable. Cannot be used for kernel DMD.
	Based on https://arxiv.org/abs/1611.03537
	"""
	assert batch_data.shape[0] == batch_inputs.shape[0] and batch_data.shape[2] == batch_inputs.shape[2], "Trajectory and control input dimensions must match"

	X, Y = prep_snapshots(batch_data, obs=obs)
	U, _ = prep_snapshots(batch_inputs)
	Xu = torch.cat((X, U), axis=0)
	
	d = X.shape[0]
	L = Y@Xu.t()@torch.pinverse(Xu@Xu.t())
	K, B = L[:, :d], L[:, d:]

	return K, B



