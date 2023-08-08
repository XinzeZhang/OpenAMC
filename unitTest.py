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
