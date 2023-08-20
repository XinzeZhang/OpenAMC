import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models.base._baseNet import BaseNet
import os
from models.base._baseTrainer import Trainer, EarlyStopping
import time
import pandas as pd
from torch import optim, nn
from tqdm import tqdm as real_tqdm
from ray.experimental.tqdm_ray import tqdm as ray_tqdm
class CausalConv1d(nn.Module):
    '''Refer to https://discuss.pytorch.org/t/causal-convolution/3456/11'''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)
    
    def forward(self, x):
        x = self.conv(x)
        return x[:,:,:-self.padding]

class MCLDNN(BaseNet):
    '''Refer to https://github.com/wzjialang/MCLDNN/blob/master/MCLDNN.py
    '''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
                    
        output_dim = hyper.num_classes

        # input(batch, 1, 2, 128)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels = 1, out_channels = 50, kernel_size = (2, 8), padding='same'),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(1),
            CausalConv1d(in_channels = 1, out_channels = 50, kernel_size=8),
            nn.ReLU(),
        )        
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(1),
            CausalConv1d(in_channels = 1, out_channels = 50, kernel_size=8),
            nn.ReLU(),
        )
        # # afer conv3(batch, 80, 1, 14)
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(50),
            nn.Conv2d(in_channels= 50, out_channels= 50, kernel_size=(1,8), padding='same'),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(100),
            nn.Conv2d(in_channels= 100, out_channels= 100, kernel_size=(2,5), padding='valid'),
            nn.ReLU(),
        )

        # afer conv4(batch, 80, 1, 6)
        # reshape(batch, 80, 6)
        # self.lstm1 = nn.Sequential(
        #     nn.LSTM(input_size= 100, hidden_size= 128, num_layers= 1, batch_first= True)
        # )
        # self.lstm2 = nn.Sequential(
        #     nn.LSTM(input_size= 128, hidden_size= 128, num_layers= 1, batch_first= True)
        # )
        self.lstm1 = nn.LSTM(input_size= 100, hidden_size= 128, num_layers= 1, batch_first= True)
        self.lstm2 = nn.LSTM(input_size= 128, hidden_size= 128, num_layers= 1, batch_first= True)
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= 128),
            nn.SELU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= 128),
            nn.SELU(),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= output_dim)
        )
        
        self.initialize_weight()
        self.to(self.hyper.device)
        

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)
        x_iq = self.conv1(x)
        x_i = self.conv2(x[:,:,0,:])
        x_q = self.conv3(x[:,:,1,:])
        x_iq2 = torch.stack([x_i,x_q],dim=-2)
        
        x_iq2 = self.conv4(x_iq2)
        _x_all = torch.cat([x_iq,x_iq2], dim=1)
        x_all = self.conv5(_x_all)
        x_all = torch.transpose(x_all[:,:,0,:],1,2)

        x_all, (h,c) = self.lstm1(x_all)
        x_all, (h,c) = self.lstm2(x_all)
        x_all = x_all[:,-1,:]
        x_all = self.fc1(x_all)
        x_all = self.fc2(x_all)
        out = self.fc3(x_all)

        return out 
    
    def _xfit(self, train_loader, val_loader):
        net_trainer = MCLDNN_Trainer2(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info    
    
class MCLDNN_Trainer(Trainer):
    def __init__(self, model,train_loader,val_loader, cfg, logger):
        super().__init__(model,train_loader,val_loader,cfg,logger)

    def adjust_lr(self):
        if self.early_stopping.counter >= self.cfg.milestone_step:
            history_lr = self.optimizer.param_groups[0]['lr']
            self.adjust_learning_rate(self.optimizer, self.cfg.gamma)                  
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f'Learning rate decreased ({history_lr:.3E} --> {current_lr:.3E}).')
            
class MCLDNN_Trainer2(Trainer):
    def __init__(self, model,train_loader,val_loader, cfg, logger):
        super().__init__(model,train_loader,val_loader,cfg,logger)
        
    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.cfg.device)
        self.early_stopping = EarlyStopping(
            self.logger, patience=self.cfg.patience)
        
        T_0 = 1 if 'T_0' not in self.cfg.dict else self.cfg.T_0
        T_mult = 2 if 'T_mult' not in self.cfg.dict else self.cfg.T_mult
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0= T_0, T_mult=T_mult)

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
    
    def run_optim_step(self, i, sig_batch, lab_batch):
        sig_batch = sig_batch.to(self.cfg.device)
        lab_batch = lab_batch.to(self.cfg.device)
        loss, acc = self.cal_loss_acc(sig_batch, lab_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(self.iter-1 + i / len(self.train_loader))

        self.train_loss.update(loss.item())
        self.train_acc.update(acc)
    
    def run_train_step(self, ray = False):
        
        if ray:
            for step, (sig_batch, lab_batch) in ray_tqdm(enumerate(self.train_loader),  total=len(self.train_loader)):
                self.run_optim_step(step, sig_batch, lab_batch)
        else:
            # pass
            with real_tqdm(total=len(self.train_loader),
                    desc=f'Epoch{self.iter}/{self.cfg.epochs}',
                    postfix=dict,
                    mininterval=0.3) as pbar:
                for step, (sig_batch, lab_batch) in enumerate(self.train_loader):
                    self.run_optim_step(step, sig_batch, lab_batch)

                    pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                        'train_acc': self.train_acc.avg})
                    pbar.update(1)
                
        return self.train_loss.avg, self.train_acc.avg
        
    def adjust_lr(self):
        # history_lr = self.optimizer.param_groups[0]['lr']
        # self.scheduler.step()      
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            f'Learning rate: ({current_lr:.3E}).')

if __name__ == '__main__':
    model = MCLDNN(11)
    x = torch.rand((4, 1,2, 128))
    y = model(x)
    print(y.shape)