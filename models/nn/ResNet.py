import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base._baseNet import BaseNet


class Subsampling_ResNet(BaseNet):
    '''
    The model arch. is same with code in https://github.com/dl4amc/dds, refered to the paper "Ensemble Wrapper Subsampling for Deep Modulation Classificatio, IEEE TCCN 2023", which is inspired by the ResNet arch. in "Over-the-air deep learning based radio signal classiﬁcation,” IEEE J. Sel. Topics Signal Process., vol. 12, no. 1, pp. 168–179, Feb. 2018."\n
    https://github.com/dl4amc/dds
    '''

    def __init__(self, hyper=None, logger=None):
        super().__init__(hyper, logger)
        output_dim = hyper.num_classes

        # input (batch, 2, 128)
        self.res_stack1 = Res_Stack(input_dim=2, output_dim=32)
        # after res_stack1 (batch, 32, 64)
        self.res_stack2 = Res_Stack(input_dim=32, output_dim=32)
        # after res_stack2 (batch, 32, 32)
        self.res_stack3 = Res_Stack(input_dim=32, output_dim=32)
        # after res_stack3 (batch, 32, 16)
        # self.res_stack4 = Res_Stack(input_dim=32, output_dim=32)
        # after res_stack4 (batch, 32, 8)

        if self.hyper.sig_len <= 8:
            raise ValueError('Input sig_len is <= 8, make the representation dim. of the residual net too small.')
        else:
            res_feature_dim = int(self.hyper.sig_len / 2 / 2 / 2)
        
        

        self.fc1 = nn.Sequential(
            nn.Linear(32*res_feature_dim, 128),
            nn.SELU(),
            nn.AlphaDropout(0.1),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.SELU(),
            nn.AlphaDropout(0.1),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, output_dim),
        )

        self.initialize_weight()
        self.to(self.hyper.device)

    def forward(self, x):
        # x = x.view(-1, 2, 128)
        x = self.res_stack1(x)
        x = self.res_stack2(x)
        x = self.res_stack3(x)
        # x = self.res_stack4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class Res_Stack(nn.Module):
    def __init__(self, input_dim, output_dim = 32):
        super(Res_Stack, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d(output_dim)
        )

        self.res_unit1 = Res_Unit(hidden_dim=32)
        self.res_unit2 = Res_Unit(hidden_dim=32)
        self.max_pooling = nn.MaxPool1d(kernel_size=2) #	
        

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.max_pooling(x)
        return x

class Res_Unit(nn.Module):
    def __init__(self, hidden_dim = 32):
        super(Res_Unit, self).__init__()
        input_dim = hidden_dim
        output_dim = hidden_dim
        if input_dim != output_dim:
            raise ValueError(f'Different input_dim and output_dim in Res_Unit!!!!!\nThe input_dim: {input_dim} \tThe output_dim: {output_dim}')
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim,
                      kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=output_dim, out_channels=output_dim,
                      kernel_size=5, padding='same',),
            nn.BatchNorm1d(output_dim),
        )
        # self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output + x
        # output = self.relu(output)
        return output