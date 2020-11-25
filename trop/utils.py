""" Utilities """

import torch
import numpy as np
import random

def set_seed(seed=None):
	"""Set random seed, or reset if None.
	"""
	random.seed(seed)
	np.random.seed(seed)
	if seed is None: 
		torch.manual_seed(random.randint(1,1e6))
	else:
		torch.manual_seed(seed)

def spectral_radius(A: torch.Tensor, eps=None, n_iter=1000):
	"""Differentiable procedure for computing spectral radius (magnitude of largest eigenvalue)

	Args:
		A: square matrix
		eps: (optional) if provided, uses convergence-based power iteration method
		n_iter: (optional) if provided, uses power iterations with this many steps (default; 1000 steps)

	For 2x2 systems, performs direct computation.
	"""
	if A.shape[0] == A.shape[1] == 2: 
		tr, det = A.trace(), A.det()
		disc = tr**2 - 4*det 
		if disc >= 0:
			return torch.max(
				torch.abs(tr + torch.sqrt(disc)) / 2,
				torch.abs(tr - torch.sqrt(disc)) / 2
			)
		else:
			return torch.sqrt((tr/2)**2 - disc/4)
	elif eps is not None:
		return _sp_radius_conv(A, eps)
	else:
		return _sp_radius_niter(A, n_iter)

def _sp_radius_conv(A: torch.Tensor, eps: float):
	v = torch.ones((A.shape[0], 1), device=A.device)
	v_new = v.clone()
	ev = v.t()@A@v
	ev_new = ev.clone()
	while (A@v_new - ev_new*v_new).norm() > eps:
		v = v_new
		ev = ev_new
		v_new = A@v
		v_new = v_new / v_new.norm()
		ev_new = v_new.t()@A@v_new
	return ev_new

def _sp_radius_niter(A: torch.Tensor, n_iter: int):
	v = torch.ones((A.shape[0], 1), device=A.device)
	for _ in range(n_iter):
		v = A@v
		v = v / v.norm()
	ev = v.t()@A@v
	return ev

def is_semistable(P: torch.Tensor, eps=1e-2):
	"""Semistability test for transfer operator using spectral radius """
	return spectral_radius(P).item() <= 1.0 + eps

def rmse(X: torch.Tensor, Y: torch.Tensor):
	assert X.shape == Y.shape
	return torch.sqrt(torch.mean((X - Y)**2)).item()

""" Tests """

if __name__ == '__main__':
	set_seed(9001)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	prec = 1e-2

	# 2d spectral radius test
	for _ in range(100):
		d = 2
		A = torch.randn((d, d), device=device)
		e = np.random.uniform(0.1, 2.00)
		L = torch.linspace(e, 0.01, d, device=device)
		P = torch.mm(torch.mm(A, torch.diag(L)), torch.pinverse(A))

		np_e_max = np.abs(np.linalg.eigvals(P.cpu().numpy())).max()
		pwr_e_max = spectral_radius(P).item()
		print('True:', e, 'numpy:', np_e_max, 'pwr_iter:', pwr_e_max)
		assert np.abs(e - pwr_e_max) < prec

	# Nd Power iteration test
	for _ in range(100):
		d = 100
		A = torch.randn((d, d), device=device)
		e = np.random.uniform(0.1, 1.10)
		L = torch.linspace(e, 0.01, d, device=device)
		P = torch.mm(torch.mm(A, torch.diag(L)), torch.pinverse(A))

		np_e_max = np.abs(np.linalg.eigvals(P.cpu().numpy())).max()
		pwr_e_max = spectral_radius(P).item()
		print('True:', e, 'numpy:', np_e_max, 'pwr_iter:', pwr_e_max)
		assert np.abs(e - pwr_e_max) < prec