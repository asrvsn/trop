"""Gradient descent-based model predictive control solver"""

import torch 
import numpy as np
from tqdm import tqdm
from scipy.integrate import ode
from typing import Callable, Dict

from trop.features import *
from trop.predict import extrapolate

def solve_mpc(
		t0: float, x0: torch.Tensor, dt: float, 
		K: torch.Tensor, B: torch.Tensor, obs: Observable, cost: Callable, h: int,
		umin: float = None, umax: float = None, lr: float = 0.1, loss_eps: float = 1e-4, unlift_every: bool = False,
	):
	"""Solve a single step of the MPC problem via Koopman operator. Uses autograd + SGD to solve minimization problem.

	Args:
		t0: start time
		x0: initial condition (d x 1)
		dt: time discretization delta
		K: Koopman operator (d x d)
		B: Control influence matrix (d x u) (see `trop.operators.solve_with_control`) 
		obs: Observable function
		cost: convex cost function C : t, x, u -> R. (t, x, u are tensors of 2nd-dimension length h, denoting time, predicted system state, and control input, respectively)
		h: horizon 
		umin: (optional) lower bound on control inputs
		umax: (optional) upper bound on control inputs
		lr: (optional) learning rate for optimizer
		loss_eps: (optional) if change in loss is below this, exits
		unlift_every: (optional) extrapolation method. By default uses faster but less accurate method. 
			For reasonable `h` this should not matter. If performance is low, try setting True.

	Returns:
		u_opt: optimal inputs 
		x_pred: predicted system response 
	"""
	x0 = x0.detach().requires_grad_()
	dim_u = B.shape[1] # dimension of control input
	u = torch.zeros((dim_u, h, 1))
	u = torch.nn.Parameter(u)
	opt = torch.optim.SGD([u], lr=lr, momentum=0.98)

	window = torch.Tensor([t0 + dt*i for i in range(h)])
	loss, prev_loss = torch.Tensor([float('inf')]), torch.Tensor([0.])

	def apply_clamp(u_in):
		if umin is not None or umax is not None:
			return u_in.clamp(min=umin, max=umax)
		return u_in

	while torch.abs(loss - prev_loss).item() > loss_eps:
		prev_loss = loss
		x_pred = extrapolate(x0.unsqueeze(1), K, obs, h-1, B=B, u=apply_clamp(u), unlift_every=unlift_every)
		loss = cost(window, x_pred, u) 
		opt.zero_grad()
		loss.backward()
		opt.step()
		u.data = apply_clamp(u.data)

	return u.data[:,:,0], x_pred

def mpc_loop(
		plant: Callable, x0: torch.Tensor, dt: float, n_iter: int,
		K: torch.Tensor, B: torch.Tensor, obs: Observable, cost: Callable, h: int, 
		umin: float = None, umax: float = None, mpc_args: Dict = {}, n_apply: int = 1, integrator: str = 'dop853', 
	):
	"""Run the model-predictive control loop with feedback from a provided plant.

	Args:
		plant: true ODE system function of the form required by `scipy.integrate.ode`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html . Used to provide actual feedback to the controller.
			Note: the only parameter to the plant should be control input. Please see `trop.examples.control` for an example plant definition.
		x0: initial condition (d x 1)
		dt: discretization time delta
		n_iter: number of steps to simulate plant w/ control
		K: Koopman operator (d x d)
		B: Control influence matrix (d x u) (see `trop.operators.solve_with_control`) 
		obs: Observable function
		cost: convex cost function C : t, x, u -> R. (t, x, u are tensors of 2nd-dimension length h, denoting time, predicted plant state, and control input, respectively). If nonconvex, results not guaranteed.
		h: horizon 
		umin: (optional) lower bound on control inputs
		umax: (optional) upper bound on control inputs
		mpc_args: (optional) arguments to MPC solver, see `solve_mpc`
		n_apply: (optional) number of control inputs to apply after each solution. Default 1 (standard MPC).
		integrator: scipy.ode integrator https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html

	Returns:
		hist_t: time history
		hist_u: control input history
		hist_x: plant response history

	"""
	assert n_apply < h, 'horizon should be larger than n_apply'

	dim_u = B.shape[1] # dimension of control input
	hist_t = [0.]
	hist_u = [torch.zeros(dim_u,)]
	hist_x = [x0]
	t = 0.

	r = ode(plant).set_integrator(integrator)
	r.set_initial_value(x0.numpy()).set_f_params(hist_u[0])

	for _ in tqdm(range(n_iter), desc='MPC'):
		u_opt, _ = solve_mpc(t, x0, dt, K, B, obs, cost, h, umin=umin, umax=umax, **mpc_args)
		for j in range(n_apply):
			u_cur = u_opt[:, j] 
			r.set_f_params(u_cur)
			r.integrate(r.t + dt)
			t += dt
			x0 = torch.Tensor(r.y) # update MPC initial condition with plant output
			hist_t.append(t)
			hist_u.append(u_cur)
			hist_x.append(x0)

	hist_t, hist_u, hist_x = torch.Tensor(hist_t), torch.stack(hist_u), torch.stack(hist_x).t()
	return hist_t, hist_u, hist_x



