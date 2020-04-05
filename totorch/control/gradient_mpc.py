"""Gradient descent-based model predictive control solver"""

import torch 
import numpy as np
from tqdm import tqdm
from scipy.integrate import ode
from typing import Callable

from totorch.features import *
from totorch.predict import extrapolate

def solve_mpc(
		t0: float, x0: torch.Tensor, dt: float, 
		K: torch.Tensor, B: torch.Tensor, obs: Observable, cost: Callable, h: int,
		umin=None, umax=None, lr=0.1, loss_eps: 1e-4, unlift_every=False,
	):
	"""Solve a single step of the MPC problem via Koopman operator. Uses autograd + SGD to solve minimization problem.

	Args:
		t0: start time
		x0: initial condition (d x 1)
		dt: time discretization delta
		K: Koopman operator (d x d)
		B: Control influence matrix (d x u) (see `totorch.operators.solve_with_control`) 
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
	u = torch.full((1, h), 0).unsqueeze(2) 
	u = torch.nn.Parameter(u)
	opt = torch.optim.SGD([u], lr=lr, momentum=0.98)

	window = torch.Tensor([t0 + dt*i for i in range(h)])
	loss, prev_loss = torch.Tensor([float('inf')]), torch.Tensor([0.])

	def apply_clamp(u_in):
		if umin is not None or umax is not None:
			return u_in.clamp(min=umin, max=umax)
		return u_in

	while torch.abs(loss - prev_loss).item() > eps:
		prev_loss = loss
		x_pred = extrapolate(x0.unsqueeze(1), K, obs, h, B=B, u=apply_clamp(u), differentiable=True, unlift_every=unlift_every)
		loss = cost(window, x_pred, u) 
		opt.zero_grad()
		loss.backward()
		opt.step()
		u.data = apply_clamp(u.data)

	return u.data, x_pred

def mpc_loop(
		system: Callable, x0: torch.Tensor, dt: float, n_iter: int,
		K: torch.Tensor, B: torch.Tensor, obs: Observsble, cost: Callable, h: int, 
		mpc_args={}, n_apply=1, integrator='dop853', sys_args={},
	):
	"""Run the model-predictive control loop with feedback from a provided reference system.

	Args:
		system: true ODE system function of the form required by `scipy.integrate.ode`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html . Used to provide actual feedback to the controller.
			Note: should take a named parameter `u`, the control input. Otherwise, system will be simulated uncontrolled. 
		x0: initial condition (d x 1)
		dt: discretization time delta
		n_iter: number of steps to simulate system w/ control
		K: Koopman operator (d x d)
		B: Control influence matrix (d x u) (see `totorch.operators.solve_with_control`) 
		obs: Observable function
		cost: convex cost function C : t, x, u -> R. (t, x, u are tensors of 2nd-dimension length h, denoting time, predicted system state, and control input, respectively). If nonconvex, results not guaranteed.
		h: horizon 
		mpc_args: (optional) arguments to MPC solver, see `solve_mpc`
		n_apply: (optional) number of control inputs to apply after each solution. Default 1 (standard MPC).
		integrator: scipy.ode integrator https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
		sys_args: other parameters to system function `system`

	Returns:
		hist_t: time history
		hist_u: control input history
		hist_x: system response history

	"""
	assert n_apply < h, 'horizon should be larger than n_apply'

	dim_u = B.shape[1] # dimension of control input
	hist_t = [0.]
	hist_u = [torch.zeros(dim_u,)]
	hist_x = [x0]
	t = 0.
	sys_args.update({'u': hist_u[0]})

	r = ode(system).set_integrator(integrator)
	r.set_initial_value(x0.numpy()).set_f_params(sys_args)

	for _ in tqdm(range(n_iter), desc='MPC'):
		u_opt = solve_mpc(t, x0, dt, K, B, obs, cost, h, **mpc_args)
		for j in range(n_apply):
			u_cur = u_opt[j].squeeze() # for 1-dimensional inputs, produce float
			sys_args.update({'u': u_cur})
			r.set_f_params(sys_args)
			r.integrate(r.t + dt)
			t += dt
			x0 = torch.Tensor(r.y) # update MPC initial condition with system output
			hist_t.append(t)
			hist_u.append(u_cur)
			hist_x.append(x0)

	hist_t, hist_u, hist_x = np.array(hist_t), np.array(hist_u), np.array(hist_x).T
	return hist_t, hist_u, hist_x



