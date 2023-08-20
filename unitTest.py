# %%
from task.base.TaskLoader import Opt

from models.base._baseSetting import hyper

hyper = hyper()

hyper.addopt = 'one'
hyper.bopt ='bopt'

class Base():
    def __init__(self, hyper) -> None:
        self.hyper = hyper
        
    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x
    
    def _xfit(self, a = None, b = None):
        print("this is from base",a, b)
        fit_info = Opt()
        fit_info.a = a
        fit_info.b = b
        return fit_info
    

class son_a(Base):
    def __init__(self, hyper = None):
        super().__init__(hyper)  
    
    def add(self,):
        self.mask = ['1', '2']
        
class son_b(Base):
    def __init__(self, hyper = None):
        super().__init__(hyper)  
        self.bopt = hyper.bopt
        
    def _xfit(self, a = None, b = None):
        print("this is from son b",a, b)
        fit_info = Opt()
        fit_info.a = a
        fit_info.b = b
        return fit_info

class son_c(son_a,  son_b):
    def __init__(self, hyper = None):
        super().__init__(hyper)
        self.add()
    
    def fit(self, a = 1, b = 1):
        # self.add()
        self._xfit(a, b)

son = son_c(hyper)
son.fit()


# %%
# import tensorflow as tf
# from tensorflow.python.keras.layers import MaxPooling1D

# x = tf.constant([x * 0.1 for x in range(128)])
# x = tf.reshape(x, [1, 1, 128])

# max_pool_1d = MaxPooling1D(data_format='channels_first')
# y = max_pool_1d(x)
# print(y)
# %%
# print(x)
# max_pool_1d(x)
# %%
