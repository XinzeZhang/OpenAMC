import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models._baseNet import BaseNet

class CLDNN(BaseNet):
    '''Refer to X. Liu, D. Yang and A. E. Gamal, "Deep neural network architectures for modulation classification," 2017 51st Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, 2017, pp. 915-919, doi: 10.1109/ACSSC.2017.8335483. \n
    https://github.com/Richardzhangxx/AMR-Benchmark/blob/main/RML201610a/CLDNN2/rmlmodels/CLDNNLikeModel.py
    '''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
        
        output_dim = hyper.num_classes
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels= 1, out_channels = 256, kernel_size= (1, 3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels= 256, out_channels= 80, kernel_size=(2,3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Conv2d(in_channels= 80, out_channels= 80, kernel_size=(1,3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.5)
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Conv2d(in_channels= 80, out_channels= 80, kernel_size=(1,3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.5)
        )
        
        self.lstm1 = nn.Sequential(
            nn.LSTM(input_size= 80, hidden_size= 50, num_layers= 1, batch_first= True)
        )
        
        self.fc1 =  nn.Sequential(
            nn.Linear(in_features= 50, out_features= 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= output_dim)
        )
        
        self.initialize_weight()
        self.to(self.hyper.device)
                    
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = torch.transpose(x[:,:,0,:],1,2)
        x, (h,c) = self.lstm1(x)
        x = x[:,-1,:]
        x = self.fc1(x)
        out = self.fc2(x)
        
        return out 
        
class PCNN(BaseNet):
    '''Refer to the paper "Ensemble Wrapper Subsampling for Deep Modulation Classificatio, IEEE TCCN 2023", which is inspired by VT-CNN2. \n
    https://codeocean.com/capsule/8397297/tree/v1/code/ranker.py
    '''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)  
        
        output_dim = hyper.num_classes
        
        output_dim = hyper.num_classes
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels= 1, out_channels = 256, padding=(0,2), kernel_size= (1, 3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.6)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels= 256, out_channels= 80, padding=(0,2), kernel_size=(2,3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.6)
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Conv2d(in_channels= 80, out_channels= 80,  padding=(0,2),kernel_size=(1,3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.6)
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.Conv2d(in_channels= 80, out_channels= 80, padding=(0,2), kernel_size=(1,3)),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size= (1,2))
            nn.Dropout(0.6),
            nn.ZeroPad2d(padding=(2,2))
        )
        
        fc_input_dim = 80 * (self.hyper.sig_len + 4 - 2 + 4 - 2 + 4 - 2 + 4 - 2 + 4 )

        self.fc1 =  nn.Sequential(
            nn.Linear(in_features= fc_input_dim, out_features= 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features= 128, out_features= output_dim)
        )
        
        self.initialize_weight()
        self.to(self.hyper.device)
    
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                                    
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # x = x.view(x.shape[0],1, 2, 128)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.shape[0],-1)
        # x = x.view(x.shape[0],-1) (batch_size, 11200) when sig_len = 128
        # x, (h,c) = self.lstm1(x)
        # x = x[:,-1,:]
        x = self.fc1(x)
        out = self.fc2(x)
        
        return out 
        
        