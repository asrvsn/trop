import torch
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from totorch.features import *
import totorch.operators as op
import totorch.systems.lorenz as lorenz
from totorch.predict import extrapolate
from totorch.utils import set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(9001)

# Init features
p, d, k = 2, 3, 9
obs = PolynomialObservable(p, d, k)
# obs = Observable(d, d, 1)

# Init data
dt = 4e-3
b = 30
n = int(b / dt)
X = lorenz.dataset(n=n, b=b).to(device) 

# Nominal operator
K = op.solve(X, obs=obs).to(device)
assert not torch.isnan(K).any().item()

hkl.dump(K.cpu().numpy(), 'koopman.hkl')
print('Koopman operator saved.')

pred_X = extrapolate(X[:,0], K, obs, X.shape[1]-1)

err = torch.norm(X - pred_X, dim=0).cpu().numpy()
X = X.cpu().numpy()
pred_X = pred_X.cpu().numpy()

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot(X[0], X[1], X[2])
ax.set_title('baseline')

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot(pred_X[0], pred_X[1], pred_X[2])
ax.set_title('predicted')

ax = fig.add_subplot(1, 3, 3)
ax.plot(np.arange(len(err)), err)
ax.set_title('error')

plt.tight_layout()
plt.show()
