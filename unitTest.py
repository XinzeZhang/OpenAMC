# %%
import torch
a = torch.device('cuda:1')
# %%
a.get_device()
# %%
a = torch.device('cpu')
b = torch.rand(10).to(a)
b.get_device()
# %%
