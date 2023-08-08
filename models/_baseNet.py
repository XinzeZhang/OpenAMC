import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

from models._comTrainer import Trainer


class BaseNet(nn.Module):
    def __init__(self, hyper = None, logger = None):
        '''
        self.hyper = hyper\n
        self.logger = logger
        '''
        super(BaseNet, self).__init__()
        
        self.hyper = hyper
        self.logger = logger

        for (arg, value) in hyper.dict.items():
            self.logger.info("Argument %s: %r", arg, value)

    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x

    def xfit(self, train_loader, val_loader):
        """
        If self.hyper has pretraining_file, then directly loading the pretraining states;\n
        else, training the model with the Traniner in models._comTrainer.\n
        
        Return: fit_info
        """
        pretraining_tag = False
        fit_info = None
        if 'pretraining_file' in self.hyper.dict and self.hyper.pretraining_file is not None and os.path.exists(self.hyper.pretraining_file):
            try:
                self.logger.info(f'Finding pretraining file in the location {self.hyper.pretraining_file}')
                
                device_id = next(self.parameters()).get_device()
                
                
                model_state = torch.load(self.hyper.pretraining_file, map_location = f'cuda:{device_id}') if device_id > -1 else torch.load(self.hyper.pretraining_file)
                
                self.load_state_dict(model_state)

                self.logger.info('Successfully loading the pretraining file!')
                pretraining_tag = True
            except:
                self.logger.exception(
                    '{}\nGot an error on loading pretraining_file in the location: {}.\n{}'.format('!'*50, self.hyper.pretraining_file, '!'*50))
                raise SystemExit()
        
        if pretraining_tag is False:
            fit_info = self._xfit(train_loader, val_loader)
            
        
        return fit_info

    def _xfit(self, train_loader, val_loader):
        net_trainer = Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info

    def predict(self, sample):
        """
        Return: prediction label of each sample as torch.tensor.
        """        
        sample = sample.to(self.hyper.device)
        logit = self.forward(sample)
        pre_lab = torch.argmax(logit, 1).cpu()
        return pre_lab
    
# if __name__ == '__main__':
#     model = AMC_Net(11, 128, 3)
#     x = torch.rand((4, 2, 128))
#     y = model(x)
