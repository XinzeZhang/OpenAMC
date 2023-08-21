from collections.abc import Mapping
import copy
import numpy as np
import torch
import torch.utils.data as Data
import os
from task.util import set_fitset
from task.base.TaskLoader import TaskDataset
from collections import Counter

def mask_cutout(data_set, mask_op):
    '''
    data_set:(data, label)\n
    data:  torch.tensor (sample, dims, steps)\n
    im_op: list lens== steps
    '''
    data, label = data_set
    _data = []
    
    # samples, dims = _data.size(0), _data.size(1)
    
    for id, tag in enumerate(mask_op):
        if int(tag) == 1:
            _data.append(data[:,:,id])
            # _data.append(data[:,:,id])
    select_data = torch.stack(_data, dim=2)
    
    new_data_set = (select_data, label)
    
    return new_data_set

def mask_count(inputMask):
    inputStep_count = Counter(inputMask) # inputStep_count is a dict. 0, 1 are keys.
    inputMask_Neg = inputStep_count[0]
    
    return inputMask_Neg

class maskcut_Dataset(TaskDataset):
    def __init__(self, args=None):
        super().__init__(args)
        
        
    def load_fitset(self, fit_batch_size = None, mask_opt = None):
        _fit_batch_size = fit_batch_size if fit_batch_size is not None else self.batch_size
        
        _mask_opt = [1 for i in range(self.sig_len)] if mask_opt is None else mask_opt
        
        self.train_set = mask_cutout(self.train_set, _mask_opt)
        self.val_set = mask_cutout(self.val_set, _mask_opt)
                
        train_loader, val_loader = set_fitset(batch_size= _fit_batch_size, num_workers=self.num_workers, train_set=self.train_set, val_set=self.val_set)

        return train_loader, val_loader
    
    def load_testset(self, test_batch_size = 64, mask_opt = None):
        _mask_opt = [1 for i in range(self.sig_len)] if mask_opt is None else mask_opt
        self.test_set = mask_cutout(self.test_set, _mask_opt)
        Signals_test, Labels_test = self.test_set

        Sample_list = []
        Label_list = []
        
        if 'num_snrs' not in self.dict:
            self.num_snrs = list(np.unique(self.snrs))
        
        
        for snr in self.num_snrs:
            test_SNRs = map(lambda x: self.SNRs[x], self.test_idx)
            test_SNRs = list(test_SNRs)
            test_SNRs = np.array(test_SNRs).squeeze()
            test_sig_i = Signals_test[np.where(np.array(test_SNRs) == snr)]
            test_lab_i = Labels_test[np.where(np.array(test_SNRs) == snr)]
            Sample = torch.chunk(test_sig_i, test_batch_size, dim=0)
            Label = torch.chunk(test_lab_i, test_batch_size, dim=0)
            
            Sample_list.append(Sample)
            Label_list.append(Label)
        
        return Sample_list, Label_list