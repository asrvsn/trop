import torch
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

from trop.features import PolynomialObservable
import trop.operators as op
import trop.systems.vdp as vdp
from trop.predict import extrapolate
from trop.utils import set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)

# Init features
p, d, k = 4, 2, 8
obs = PolynomialObservable(p, d, k)

# Init data
mu = 3.0
n_vdp = 6000
b_vdp = 40
X = vdp.dataset(mu, n=n_vdp, b=b_vdp).to(device) 

# Nominal operator
K = op.solve(X, obs=obs).to(device)
assert not torch.isnan(K).any().item()

hkl.dump(K.cpu().numpy(), 'koopman.hkl')
print('Koopman operator saved.')

pred_X = extrapolate(X[:,0], K, obs, X.shape[1]-1)

err = torch.norm(X - pred_X, dim=0).cpu().numpy()
X = X.cpu().numpy()
pred_X = pred_X.cpu().numpy()

fig, axs = plt.subplots(1, 3)
axs[0].plot(X[0], X[1])
axs[0].set_title('baseline')
axs[1].plot(pred_X[0], pred_X[1])
axs[1].set_title('predicted')
axs[2].plot(np.arange(len(err)), err)
axs[2].set_title('error')

plt.show()