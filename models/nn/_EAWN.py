import torch
import torch.nn as nn

from models.nn._baseNet import BaseNet
from models.nn._baseTrainer import AnnealingTrainer
from models.nn.AWN import AWN, AWN_Trainer
import math

import importlib

class EAWN(AWN):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)

    def initialize_arch(self):
        self.aux_import()

        self.fc = nn.Sequential(
            nn.Linear(self.out_channels+self.aux_dim, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, self.num_classes)
            )

        # self.aux_mlp = nn.Sequential(
        #     nn.Linear(self.aux_dim, self.aux_dim*2),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Linear(self.aux_dim*2, self.aux_dim),
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # )
        # self.readout = nn.Linear(self.latent_dim +self.aux_dim, self.num_classes)


        self.to(self.hyper.device)

    def aux_import(self,):
        with torch.no_grad():
            self.auxs= nn.ModuleList()
            self.aux_dim = 0
            for model_config in self.hyper.emodels:
                self.aux_dim = self.aux_dim + model_config.hyper.num_classes + 2
                model = importlib.import_module(model_config.import_path)
                model = getattr(model, model_config.class_name)
                # model_config.hyper.num_classes = self.hyper.emodel_logits
                model_config.hyper.merge(self.hyper, ['sig_len', 'device'])
                model = model(model_config.hyper, self.logger)
                model.load_pretraing_file(file_path =  model_config.hyper.pretraining_allDatafile)
                self.auxs.add_module(model_config.aux_name, model)

    def aux_infeature(self,x):
        with torch.no_grad():
            aux_out= []
            for aux in self.auxs:
                aux_logit = aux.logits(x)
                logit_mean_std = torch.stack(torch.std_mean(aux_logit, dim=1), dim=1)
                aux_info = torch.cat((aux_logit, logit_mean_std), dim=1)
                aux_out.append(aux_info)
            # if len(aux_out) > 0:
            #     aux_out = torch.cat(aux_out, dim=1)

        return aux_out

    def forward(self, x):
        signal = x.detach().clone()
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

        aux_out = self.aux_infeature(signal)
        if len(aux_out) > 0:
            aux_out = torch.cat(aux_out, dim=1)
            x = torch.cat((x, aux_out), dim=1)


        logit = self.fc(x)
        # aux_feature = self.aux_mlp(aux_out)
        # logit = self.readout(x)

        return logit, regu_sum

class EAWN2(EAWN):
    def _xfit(self, train_loader, val_loader):
        net_trainer = EAWN_Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info


class EAWN_Trainer(AnnealingTrainer):
    def __init__(self, model,train_loader,val_loader, cfg, logger):
        super().__init__(model,train_loader,val_loader, cfg,logger)

    def cal_loss_acc(self, sig_batch, lab_batch):
        logit, regu_sum = self.model(sig_batch)
        loss = self.criterion(logit, lab_batch)
        loss += sum(regu_sum)

        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ).item() / lab_batch.size(0)

        return loss, acc

class EAWN3(EAWN):
    def __init__(self, hyper = None, logger = None):
        super().__init__(hyper, logger)

    def initialize_arch(self):
        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.aux_import()

        # self.aux_attention = nn.Sequential(
        #     nn.Linear(self.aux_dim, self.aux_dim // 4, bias=False),
        #     nn.Dropout(0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.aux_dim // 4, self.aux_dim, bias=False),
        #     nn.Sigmoid()
        # )
        ffm_dim = self.latent_dim +self.aux_dim
        self.norm_layer1=nn.LayerNorm(ffm_dim)
        self.norm_layer2 = nn.LayerNorm(ffm_dim)

        self.ff_attention = nn.Sequential(
            nn.Linear(ffm_dim, ffm_dim // 4, bias=False),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(ffm_dim // 4, ffm_dim, bias=False),
            nn.Tanh(),
            nn.Dropout(0.5),
        )

        self.readout =nn.Sequential(
            nn.Linear(self.latent_dim + self.aux_dim, self.latent_dim),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(self.latent_dim, self.num_classes)
        )

        self.to(self.hyper.device)

    def forward(self, x):
        signal = x.detach().clone()
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

        x = self.fc(x)

        aux_out = self.aux_infeature(signal)
        aux_out = torch.cat(aux_out, dim=1)
        # aux_feature = torch.mul(self.aux_attention(aux_out), aux_out)

        x = torch.cat((x, aux_out), dim=1)
        x = self.norm_layer1(x)
        x = torch.mul(self.ff_attention(x), x)
        x = self.norm_layer2(x)
        logit = self.readout(x)

        return logit, regu_sum

    def _xfit(self, train_loader, val_loader):
        net_trainer = EAWN_Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info

class EAWN4(EAWN):
    def _xfit(self, train_loader, val_loader):
        net_trainer = AWN_Trainer(self, train_loader, val_loader, self.hyper, self.logger)
        net_trainer.loop()
        fit_info = net_trainer.epochs_stats
        return fit_info

class EAWN5(EAWN3):
    def aux_import(self,):
        with torch.no_grad():
            self.auxs= nn.ModuleList()
            self.aux_dim = 0
            for model_config in self.hyper.emodels:
                self.aux_dim += model_config.hyper.num_classes + 2
                self.aux_dim += model_config.hyper.in_channels * (model_config.hyper.num_level + 1) # the out_channels in AWN
                model = importlib.import_module(model_config.import_path)
                model = getattr(model, model_config.class_name)
                # model_config.hyper.num_classes = self.hyper.emodel_logits
                model_config.hyper.merge(self.hyper, ['sig_len', 'device'])
                model = model(model_config.hyper, self.logger)
                model.load_pretraing_file(file_path =  model_config.hyper.pretraining_allDatafile)
                self.auxs.add_module(model_config.aux_name, model)

    def aux_infeature(self,x):
        with torch.no_grad():
            aux_out= []
            for aux in self.auxs:
                aux_logit = aux.logits(x)
                logit_mean_std = torch.stack(torch.std_mean(aux_logit, dim=1), dim=1)
                aux_hidden = aux.feature_extract(x)
                aux_info = torch.cat([aux_hidden, aux_logit, logit_mean_std], dim=1)
                aux_out.append(aux_info)
            aux_out = torch.cat(aux_out, dim=1)

        return aux_out

class EAWN6(EAWN3):
    def aux_import(self,):
        with torch.no_grad():
            self.auxs= nn.ModuleList()
            self.aux_dim = 0
            self.aux_num = 0
            for model_config in self.hyper.emodels:
                self.aux_dim = model_config.hyper.num_classes
                self.aux_num += 1
                # self.aux_dim += model_config.hyper.in_channels * (model_config.hyper.num_level + 1) # the out_channels in AWN
                model = importlib.import_module(model_config.import_path)
                model = getattr(model, model_config.class_name)
                # model_config.hyper.num_classes = self.hyper.emodel_logits
                model_config.hyper.merge(self.hyper, ['sig_len', 'device'])
                model = model(model_config.hyper, self.logger)
                model.load_pretraing_file(file_path =  model_config.hyper.pretraining_allDatafile)
                self.auxs.add_module(model_config.aux_name, model)

    def aux_infeature(self,x):
        with torch.no_grad():
            aux_out= []
            for aux in self.auxs:
                aux_logit = aux.logits(x)
                # logit_mean_std = torch.stack(torch.std_mean(aux_logit, dim=1), dim=1)
                # aux_hidden = aux.feature_extract(x)
                # aux_info = torch.cat([aux_hidden, aux_logit, logit_mean_std], dim=1)
                aux_out.append(aux_logit)

            aux_out = torch.stack(aux_out, dim=1)

        return aux_out
