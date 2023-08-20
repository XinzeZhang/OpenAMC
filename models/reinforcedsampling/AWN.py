from models.nn.AWN import AWN
from models.reinforcedsampling._maskNet import MaskNet

import torch
import torch.nn as nn

class maskAWN(AWN, MaskNet):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)
        
        self.init_inputMask()
        
    

