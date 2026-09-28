import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

import torch
import torch.nn as nn
from torch import optim

from models.nn._baseNet import BaseNet
# from models.nn._baseTrainer import Trainer, AnnealingTrainer, EarlyStopping
import math
# from models.nn._augTrainer import AnnealingTrainer as augAnnealingTrainer
# from models.nn._augTrainer import Trainer as augTrainer

from models.nn._baseNet import BaseNetConfig
class AWN_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/AWN.py'
        self.class_name = 'AWN'
        # self.trainer_module = (self.import_path, 'AWN_Trainer')

        self.hyper.batch_size = 128
        self.hyper.gamma = 0.5
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320

# class AWN_Trainer(AnnealingTrainer):
#     def cal_loss_acc(self, sig_batch, lab_batch):
#         logit, regu_sum = self.model(sig_batch)
#         loss = self.criterion(logit, lab_batch)
#         loss += sum(regu_sum)

#         pre_lab = torch.argmax(logit, 1)
#         acc = torch.sum(pre_lab == lab_batch.data).double(
#         ).item() / lab_batch.size(0)

#         return loss, acc

# class Aug_AWN_Trainer(augAnnealingTrainer):
#     def cal_loss_acc(self, sig_batch, lab_batch, lab_weight):
#         logit, regu_sum = self.model(sig_batch)
#         loss = self.criterion(logit, lab_batch)
#         loss = loss * lab_weight
#         loss = torch.mean(loss)
#         loss += sum(regu_sum)
#         pre_lab = torch.argmax(logit, 1)
#         acc = torch.sum(pre_lab == lab_batch.data).double(
#         ).item() / lab_batch.size(0)
#         return loss, acc

# class NormAug_AWN_Trainer(augAnnealingTrainer):
#     def cal_loss_acc(self, sig_batch, lab_batch, lab_weight):
#         logit, regu_sum = self.model(sig_batch)
#         loss = self.criterion(logit, lab_batch)
#         loss = (loss * lab_weight / lab_weight.sum()).sum()
#         loss = torch.mean(loss)
#         loss += sum(regu_sum)
#         pre_lab = torch.argmax(logit, 1)
#         acc = torch.sum(pre_lab == lab_batch.data).double(
#         ).item() / lab_batch.size(0)
#         return loss, acc


class AWN(BaseNet):
    '''
    J. Zhang, T. Wang, Z. Feng, and S. Yang, “Toward the automatic modulation classification with adaptive wavelet network,” IEEE Transactions on Cognitive Communications and Networking, vol. 9, no. 3, pp. 549–563, June 2023, doi: 10.1109/TCCN.2023.3252580.
    '''

    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)

    def initialize_arch(self):
        self.num_classes = self.hyper.num_classes

        max_level = 0
        while( self.hyper.sig_len / math.pow(2, 1+max_level) > 2):
            max_level +=1

        self.num_levels = self.hyper.num_level if self.hyper.num_level <= max_level else max_level

        self.in_channels = self.hyper.in_channels
        self.out_channels = self.in_channels * (self.num_levels + 1)
        self.kernel_size = 3 if 'kernel_size' not in self.hyper.dict else self.hyper.kernel_size # only be 3 can run.
        self.latent_dim = self.hyper.latent_dim
        self.regu_details = self.hyper.regu_details
        self.regu_approx = self.hyper.regu_approx

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            # Call a 2d Conv to integrate I, Q channels
            nn.Conv2d(1, self.in_channels,
                      kernel_size=(2, 7), stride=(1,), bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels,
                      kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.levels = nn.ModuleList()

        for i in range(self.num_levels):
            self.levels.add_module(
                'level_' + str(i),
                LevelTWaveNet(self.in_channels,
                              self.kernel_size,
                              self.regu_details,
                              self.regu_approx)
            )

        self.SE_attention_score = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 4, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 4, self.out_channels, bias=False),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, self.num_classes)
        )

        self.to(self.hyper.device)

    def forward(self, x):
        x = x.unsqueeze(1)  # x:[N, 2, T] -> [N, 1, 2, T]
        x = self.conv1(x)
        x = x.squeeze(2)  # x:[N, C, 1, T] -> [N, C, T]
        x = self.conv2(x)
        regu_sum = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        x = torch.mul(self.SE_attention_score(x), x)

        logit = self.fc(x)

        return logit, regu_sum

    def feature_extract(self,x):
        x = x.unsqueeze(1)  # x:[N, 2, T] -> [N, 1, 2, T]
        x = self.conv1(x)

        x = x.squeeze(2)  # x:[N, C, 1, T] -> [N, C, T]
        x = self.conv2(x)
        regu_sum = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        x = torch.mul(self.SE_attention_score(x), x)

        return x

    def get_logits_and_intermediate_features(self, x):
        features = []

        x = x.unsqueeze(1)
        x = self.conv1(x)
        features.append(x)

        x = self.conv2(x.squeeze(2))
        features.append(x)

        regu_sum = []
        det = []

        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        features.append(x)

        x = torch.mul(self.SE_attention_score(x), x)
        logit = self.fc(x)

        return logit, features

    def logits(self, sample):
        """
        Return: prediction logits of each sample as torch.tensor.
        """
        sample = sample.to(self.hyper.device)
        logits, _ = self.forward(sample)
        return logits

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

        self.conv_even = lambda x: x[:, :, ::2]
        self.conv_odd = lambda x: x[:, :, 1::2]

    def forward(self, x):
        """
        returns the odd and even part
        :param x:
        :return: x_even, x_odd
        """
        return self.conv_even(x), self.conv_odd(x)

class Operator(nn.Module):
    def __init__(self, in_planes, kernel_size=3, dropout=0.):
        super(Operator, self).__init__()

        pad = (kernel_size - 1) // 2 + 1

        self.operator = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Operator as Predictor() or Updator()
        :param x:
        :return: P(x) or U(x)
        """
        x = self.operator(x)
        return x

class LiftingScheme(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(LiftingScheme, self).__init__()

        self.split = Splitting()

        self.P = Operator(in_planes, kernel_size)
        self.U = Operator(in_planes, kernel_size)

    def forward(self, x):
        """
        Implement Lifting Scheme
        :param x:
        :return: c: approximation coefficient
                 d: details coefficient
        """
        (x_even, x_odd) = self.split(x)
        c = x_even + self.U(x_odd)
        d = x_odd - self.P(c)
        return c, d

class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        """
        Conduct decomposition and calculate regularization terms
        :param x:
        :return: approx component, details component, regularization terms
        """
        global regu_d, regu_c
        (L, H) = self.wavelet(x)  # 10 9 128
        approx = L
        details = H
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                regu_d = self.regu_details * H.abs().mean()
            # Constrain on the approximation
            if self.regu_approx:
                regu_c = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                # Only the details
                regu = regu_d
            elif self.regu_details == 0.0:
                # Only the approximation
                regu = regu_c
            else:
                # Both
                regu = regu_d + regu_c

            return approx, details, regu

if __name__ == "__main__":
    """ 测试网络结构构建是否构建正确，并打印每层参数 """
    from torchinfo import summary

    hyper = AWN_config().hyper
    hyper.num_classes = 11
    hyper.sig_len = 1024
    model = AWN(hyper=hyper)

    x = torch.randn(2, 2, hyper.sig_len, device=hyper.device)
    features = model.get_intermediate_features(x)
    for i, f in enumerate(features):
        print(f"Feature {i} shape: {f.shape}")

    # model.cuda()
    # print(model)
    # # 统计网络参数及输出大小
    # summary(model, (2, hyper.sig_len), batch_dim=0)
