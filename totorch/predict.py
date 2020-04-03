""" Prediction & extrapolation """

import torch
import numpy as np
import itertools
import traceback
from tqdm import tqdm

from totorch.features import *


def extrapolate(
		X0: torch.Tensor, K: torch.Tensor, obs: Observable, T: int, 
		B=None, u=None, unlift_every=True, differentiable=False
	):
	"""Extrapolate dynamical system from initial conditions using Koopman operator. 

	Args:
		X0: d (state dimension) x m initial conditions
		K: Koopman operator
		obs: Observable function
		T: extrapolation length

	Optional args:
		B: control influence matrix 
		u: control inputs d (input dimension) x T (trajectory length)
 		unlift_every: use slower but more accurate extrapolation method (TODO: should not have a difference)
		differentiable: build extrapolation in autograd-compliant manner
	"""
	assert X0.shape[0] == obs.d, 'IC dimension should match observable parameters'
	assert X0.shape[1] >= obs.m, f'Insufficient initial conditions for observable with memory requirement {obs.m}'
	if not differentiable:
		K, X0 = K.detach(), X0.detach()
	if u is not None:
		assert B is not None, 'Control matrix required'
		assert u.shape[1] >= T, f'Insufficient control inputs provided for extrapolation length {T}'
		if not differentiable:
			B, u = B.detach(), u.detach()

	if unlift_every:
		if differentiable:
			Y = [X0[:, :obs.m]]
			x_cur = X0[:, :obs.m].unsqueeze(1)
			for i in range(obs.m, T):
				z = P@obs(x_cur, differentiable=True)
				if u is not None:
					z = z + B@u[:, i-1]
				x_cur = obs.preimage(z)
				Y.append(x_cur.view(-1))
			return torch.stack(Y, dim=1)
		else:
			Y = torch.full((obs.d, t), np.nan, device=X0.device)
			Y[:, :obs.m] = X0[:, :obs.m]
			for i in range(obs.m, t):
				x = Y[:, i-obs.m:i]
				z = P@obs(x)
				if u is not None:
					z += B@u[:, i-1]
				Y[:, i] = obs.preimage(z).view(-1)
			return Y
	else:
		if differentiable:
			z_cur = obs(X0[:, :obs.m], differentiable=True)
			Z = [z_cur.view(-1)]
			for i in range(obs.m, t):
				z_cur = P@z_cur
				if u is not None:
					z_cur = z_cur + B@u[:, i-1]
				Z.append(z_cur.view(-1))
			return obs.preimage(torch.stack(Z, dim=1))
		else:
			Z = torch.full((obs.k, t), np.nan, device=X0.device)
			Z[:, 0] = obs(X0[:, :obs.m]).view(-1)
			z = Z[:, obs.m-1].unsqueeze(1)
			for i in range(obs.m, t):
				z = P@z
				if u is not None:
					z += B@u[:, i-1]
				Z[:, i] = z.view(-1)
			return obs.preimage(Z)

def _tr_worker(ic: tuple, K: torch.Tensor, obs: Observable, T: int):
	try:
		ic = torch.Tensor(list(ic)).unsqueeze(0)
		tr = extrapolate(K, obs, ic, T)
		return tr
	except:
		print(traceback.format_exc())

def extrapolate_many(K: torch.Tensor, obs: Observable, ic_space: np.ndarray, T: int):
	"""Efficiently extrapolate trajectories from multiple initial conditions in parallel.

	Args:
		K: Koopman operator
		obs: Observable
		ic_space: space of initial conditions, d (state dimension) x 3 (start, end, n_samples)
		T: trajectory length

	Returns:
		list of trajectories in no particular order

	Assumes inefficient but accurate extrapolation method.
	This method is non-differentiable.
	"""
	trajectories = []
	N = np.prod(ic_space[:,2])
	ic_range = [np.linspace(a, b, n) for [a, b, n] in ic_space]
	with tqdm(total=N, desc='Trajectories') as pbar:

		def finish(tr):
			trajectories.append(tr)
			pbar.update(1)

		# https://github.com/pytorch/pytorch/issues/973
		# torch.multiprocessing.set_sharing_strategy('file_system')
		with multiprocessing.Pool() as pool:
			for ic in itertools.product(ic_range):
				pool.apply_async(_tr_worker, args=(ic, K, obs, T), callback=finish)
			pool.close()
			pool.join()

	return trajectories