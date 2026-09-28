import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.base_model import BaseModel
from models.nn._baseNet import BaseNet, BaseNetConfig

class VGG16_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/VGG.py'
        self.class_name = 'VGG16'

        self.hyper.epochs = 100
        self.hyper.gamma = 0.5

class VGG16(BaseNet):
    '''Refer to K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.'''
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)

    def initialize_arch(self):
        output_dim = self.hyper.num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
            nn.BatchNorm2d(512),
            nn.ReLU())

        reduce_dim = self.hyper.sig_len // 2 // 2 // 2 // 2 // 2
        if self.hyper.sig_len < 1024:
            self.layer13 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)))
        else:
            self.target_len = 128
            k = self.hyper.sig_len // self.target_len
            self.layer13 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=(1,3), stride=1, padding = (0,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (1,2), stride = (1,2)),
                nn.Conv2d(512, 512, kernel_size=(1,k), stride=(1,k), groups=512, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(512))
            reduce_dim = reduce_dim // k

        fc_input_dim = reduce_dim * 2 * 512


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_dim, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc3= nn.Sequential(
            nn.Linear(4096, output_dim))



        self.initialize_weight()
        self.to(self.hyper.device)

    def forward(self, x):
        # print('Source shape:', x.shape)  # Debug: print input shape
        x = torch.unsqueeze(x, 1)
        # print('Input shape:', x.shape)  # Debug: print input shape
        x = self.layer1(x)
        # print('layer1 shape:', x.shape)  # Debug: print input shape
        x = self.layer2(x)
        # print('layer2 shape:', x.shape)  # Debug: print input shape
        x = self.layer3(x)
        # print('layer3 shape:', x.shape)  # Debug: print input shape
        x = self.layer4(x)
        # print('layer4 shape:', x.shape)  # Debug: print input shape
        x = self.layer5(x)
        # print('layer5 shape:', x.shape)  # Debug: print input shape
        x = self.layer6(x)
        # print('layer6 shape:', x.shape)  # Debug: print input shape
        x = self.layer7(x)
        # print('layer7 shape:', x.shape)  # Debug: print input shape
        x = self.layer8(x)
        # print('layer8 shape:', x.shape)  # Debug: print input shape
        x = self.layer9(x)
        # print('layer9 shape:', x.shape)  # Debug: print input shape
        x = self.layer10(x)
        # print('layer10 shape:', x.shape)  # Debug: print input shape
        x = self.layer11(x)
        # print('layer11 shape:', x.shape)  # Debug: print input shape
        x = self.layer12(x)
        # print('layer12 shape:', x.shape)  # Debug: print input shape
        x = self.layer13(x)
        # print('layer13 shape:', x.shape)  # Debug: print input shape

        x = x.view(x.shape[0], -1)
        # print('layer13->14 shape:', x.shape)  # Debug: print input shape
        x = self.fc1(x)
        # print('layer14 shape:', x.shape)  # Debug: print input shape
        x = self.fc2(x)
        # print('layer15 shape:', x.shape)  # Debug: print input shape
        x = self.fc3(x)
        # print('layer16 shape:', x.shape)  # Debug: print input shape

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

if __name__ == "__main__":
    from taskRecog.Wrapper import Task
    from taskRecog.Parser import get_parser


    args = get_parser()
    args.model_init_path = 'models/nn/VGG.py'

    args.model = 'VGG16_config'
    args.test = True
    task = Task(args)
    task.conduct()