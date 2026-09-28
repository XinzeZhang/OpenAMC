import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models.nn._baseNet import BaseNet, BaseNetConfig

class VTCNN_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/VT_CNN.py'
        self.class_name = 'VTCNN'

        self.hyper.epochs = 100
        self.hyper.gamma = 0.5

class VTCNN(BaseNet):
    '''Refer to https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb'''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)

    def initialize_arch(self):
        output_dim = self.hyper.num_classes
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            # nn.ZeroPad2d(padding = (2, 2)),
            nn.Conv2d(in_channels = 1, out_channels = 256, padding=(0,2), kernel_size = (1, 3)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        if self.hyper.sig_len < 1024:
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(256),
                # nn.ZeroPad2d(padding = (0, 2)),
                nn.Conv2d(in_channels = 256, out_channels = 80, padding=(0,2), kernel_size=(2, 3)),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            fc_input_dim = (self.hyper.sig_len + 4 - 2 + 4 - 2 ) * 80
        else:
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(256),
                # nn.ZeroPad2d(padding = (0, 2)),
                nn.Conv2d(in_channels = 256, out_channels = 80, padding=(0,2), kernel_size=(2, 3)),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.BatchNorm2d(80),
                # nn.ZeroPad2d(padding = (0, 2)),
                nn.Conv2d(in_channels = 80, out_channels = 8, padding=(0,2), kernel_size=(1, 3)),
                nn.ReLU(),
                nn.Dropout(0.5),
            )
            fc_input_dim = (self.hyper.sig_len + 4 - 2 + 4 - 2 + 4 - 2) * 8


        # assert fc_input_dim == 10560


        self.fc1 = nn.Sequential(
                nn.Linear(fc_input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
            )

        self.fc2 = nn.Sequential(
            nn.Linear(256, output_dim)
        )

        self.initialize_weight()
        self.to(self.hyper.device)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_uniform_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.kaiming_normal_(m.bias)
                # nn.init.constant_(m.bias, 0)
