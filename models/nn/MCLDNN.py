import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models.nn._baseNet import BaseNet, BaseNetConfig
import os
from models.nn._baseTrainer import Trainer, EarlyStopping, AnnealingTrainer
import time
import pandas as pd
from torch import optim, nn
from tqdm.auto import tqdm as real_tqdm
class MCLDNN_config(BaseNetConfig):
    '''Refer to the paper:  'A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition.
    '''
    def base_modify(self):
        self.import_path = 'models/nn/MCLDNN.py'
        self.class_name = 'MCLDNN'
        # self.trainer_module = (self.import_path, 'MCLDNN_Trainer')
        self.arch = 'crnn'

        self.hyper.milestone_step = 2
        self.hyper.gamma = 0.8


class MCLDNN(BaseNet):
    '''Refer to https://github.com/wzjialang/MCLDNN/blob/master/MCLDNN.py
    '''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)

    def initialize_arch(self):
        output_dim = self.hyper.num_classes

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
        self.has_rnn = True

    def rnn_reactivate(self,):
        self.lstm1.train()
        self.lstm2.train()


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

    def get_logits_and_intermediate_features(self, x):
        features = []

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
        features.append(x_all)
        x_all = self.fc1(x_all)
        features.append(x_all)
        x_all = self.fc2(x_all)
        features.append(x_all)
        out = self.fc3(x_all)

        return out, features


    def feature_extract(self, x):
        x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)
        x_i = self.conv2(x[:,:,0,:])
        x_q = self.conv3(x[:,:,1,:])
        x_iq2 = torch.stack([x_i,x_q],dim=-2)

        return x_iq2


class CausalConv1d(nn.Module):
    '''Refer to https://discuss.pytorch.org/t/causal-convolution/3456/11'''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:,:,:-self.padding]



if __name__ == '__main__':
    model = MCLDNN(11)
    x = torch.rand((4, 1,2, 128))
    y = model(x)
    print(y.shape)