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
class test:
    def __init__(self) -> None:
        self.a = 10
        self.b = self.a
        
    def opt(self,):
        self.a = 100
    
    def show(self):
        print(self.b)

t = test()
t.show()
t.opt()
t.show()
# %%
print('%')

# %%
import numpy as np

a = np.array([1,2])
a.shape
# %%
a.max()
# %%
10 % 3
# %%
from task.TaskLoader import Opt

class demo(Opt):
    def __init__(self):
        super().__init__()
        self._unit = 0
    
    @property
    def unit(self):
        return self._unit
    
# %%
a = demo()
print(a.unit)

# %%
import os
os.cpu_count()
# %%
1 // 0.5
# %%
1 // 0.3
# %%
