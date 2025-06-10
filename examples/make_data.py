# %%

import numpy as np
import torch

import context


dtype = torch.float32

# %%

def generate_gp(num_samples, seq_len, tau, dtype=torch.float32):
    t = torch.arange(seq_len, dtype=dtype)
    # Zero mean
    mean = torch.zeros_like(t, dtype=dtype)
    # Time correlation function
    cov = torch.exp(-0.5*(t[None,:] - t[:,None])**2 / tau**2)
    cov += 1e-2*torch.eye(len(t))
    # Check for positive definiteness
    assert np.all(np.linalg.eigvals(cov) > 0)
    # Generate
    distrib = torch.distributions.MultivariateNormal(mean, cov)
    return distrib.rsample((num_samples,)).squeeze()

# %%
# Gaussian process

seq = generate_gp(25, 500, 10.0, dtype)
print(seq.shape)
torch.save(seq, context.tmp_dir / 'data-gp.pt')

# %%
