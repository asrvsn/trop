import torch
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
import pdb

from trop.features import PolynomialObservable
import trop.operators as op
import trop.systems.duffing as duffing
from trop.predict import extrapolate
from trop.utils import set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)

# Init features
p, d, k = 5, 2, 15
obs = PolynomialObservable(p, d, k)

# Init data
tf = 400
n = 8000
X = duffing.dataset(n=n, tf=tf).to(device) 

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