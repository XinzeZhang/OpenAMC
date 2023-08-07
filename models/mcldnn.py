import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models._comTrainer import Trainer

class MCLDNN(nn.Module):
    def __init__(self, hyper = None, logger = None):
        super(MCLDNN, self).__init__()
        self.hyper = hyper
        self.logger = logger

        for (arg, value) in hyper.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
                    
        output_dim = hyper.num_classes

        # input(batch, 1, 2, 128)
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.ZeroPad2d(padding = (3,4,0,1)),
            nn.Conv2d(in_channels = 1, out_channels = 50, kernel_size = (2, 8)),
            nn.ReLU(),
            # nn.Dropout(0.6)
        )
        self.conv2 = nn.Sequential(
            # nn.BatchNorm1d(1),
            nn.ZeroPad2d(padding = (7, 0,0,0)),
            nn.Conv1d(in_channels = 1, out_channels = 50, kernel_size=(8)),
            nn.ReLU(),
            # nn.Dropout(0.6)
        )        
        self.conv3 = nn.Sequential(
            # nn.BatchNorm1d(1),
            nn.ZeroPad2d(padding = (7, 0,0,0)),
            nn.Conv1d(in_channels = 1, out_channels = 50, kernel_size=(8)),
            nn.ReLU(),
            # nn.Dropout(0.6)
        )
        # # afer conv3(batch, 80, 1, 14)
        self.conv4 = nn.Sequential(
            # nn.BatchNorm2d(50),
            nn.ZeroPad2d(padding = (3, 4,0,0)),
            nn.Conv2d(in_channels= 50, out_channels= 50, kernel_size=(1,8)),
            nn.ReLU(),
            # nn.Dropout(0.6),
        )

        self.conv5 = nn.Sequential(
            # nn.BatchNorm2d(100),
            nn.Conv2d(in_channels= 100, out_channels= 100, kernel_size=(2,5)),
            nn.ReLU(),
            # nn.Dropout(0.6),
        )
        # afer conv4(batch, 80, 1, 6)
        # reshape(batch, 80, 6)
        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size= 100, hidden_size= 128, num_layers= 1, batch_first= True)
        )
        self.lstm2 = nn.Sequential(
            nn.LSTM(input_size= 128, hidden_size= 128, num_layers= 1, batch_first= True)
        )
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

    def forward(self, x, y=None, is_train=False):
        x = x.view(x.shape[0],1, 2, 128)
        x_iq = self.conv1(x)
        x_i = self.conv2(x[:,:,0,:])
        x_q = self.conv3(x[:,:,1,:])
        x_iq2 = torch.stack([x_i,x_q],dim=-2)
        
        x_iq2 = self.conv4(x_iq2)
        x_all = torch.cat([x_iq,x_iq2], dim=1)
        x_all = self.conv5(x_all)
        x_all = torch.transpose(x_all[:,:,0,:],1,2)

        x_all, (h,c) = self.lstm1(x_all)
        x_all, (h,c) = self.lstm2(x_all)
        x_all = x_all[:,-1,:]
        x_all = self.fc1(x_all)
        x_all = self.fc2(x_all)
        out = self.fc3(x_all)
        if is_train:
            loss = F.cross_entropy(out,y)
            return loss, out

        return out 
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = MCLDNN(11)
    x = torch.rand((4, 1,2, 128))
    y = model(x)
    print(y.shape)