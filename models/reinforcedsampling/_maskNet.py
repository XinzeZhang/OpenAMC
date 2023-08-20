import torch
import torch.nn as nn
import numpy as np
from models.base._baseNet import BaseNet
# from models.base._baseTrainer import Trainer
from collections import Counter


def mask_select(data, mask_op, data_type='torch'):
    '''
    data: numpy array (sample, dims, steps)\n
    im_op: list lens== steps
    '''
    _data = []
    for id, tag in enumerate(mask_op):
        if int(tag) == 1:
            _data.append(data[:,:,id])
    if data_type == 'numpy':
        select_data = np.stack(_data, axis=-1)
    elif data_type == 'torch':
        select_data = torch.stack(_data, dim=2)
    else:
        raise ValueError("Unsupported data_type: {}".format(type(data)))
    
    return select_data



class MaskNet(BaseNet):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)
        
    def init_inputMask(self,):
        '''the inputMask constrain should be in the tuningcell
        '''
        input_select_tag = False
        for i in range(self.hyper.sig_len):
            if 'inputMask_{}'.format(i) in self.hyper.dict:
                input_select_tag = True
                break
        
        if input_select_tag:
            for i in range(self.hyper.sig_len):
                if 'inputMask_{}'.format(i) not in self.hyper.dict:
                    raise ValueError('Missing hyper config: "inputMask_{}"!'.format(i))
            
            self.inputMask = [self.hyper.dict['inputMask_{}'.format(i)] for i in range(self.hyper.sig_len)]
            
        else:
            inputMask = np.ones(self.hyper.sig_len)
            self.inputMask = inputMask.tolist()
            
        inputStep_count = Counter(self.inputMask)
        self.inputMask_Pos = inputStep_count[1]
        
        if self.inputMask_Pos < 2: # add min constrain to input mask
            if self.inputMask[-1] == 1:
                self.inputMask[-2] = 1
            else:
                self.inputMask[-1] = 1
                
            inputStep_count = Counter(self.inputMask)
            self.inputMask_Pos = inputStep_count[1]
                
        
        