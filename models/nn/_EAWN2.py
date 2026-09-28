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
        self.num_heads = 2

        self.aux_import()

        self.fc = nn.Sequential(
            nn.Linear(self.out_channels+self.aux_num, self.latent_dim),
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
            self.aux_num = 0
            for model_config in self.hyper.emodels:
                self.aux_dim = model_config.hyper.num_classes + 2
                self.aux_num += 1
                # self.aux_dim += model_config.hyper.in_channels * (model_config.hyper.num_level + 1) # the out_channels in AWN
                model = importlib.import_module(model_config.import_path)
                model = getattr(model, model_config.class_name)
                # model_config.hyper.num_classes = self.hyper.emodel_logits
                model_config.hyper.merge(self.hyper, ['sig_len', 'device'])
                model = model(model_config.hyper, self.logger)
                model.load_pretraing_file(file_path =  model_config.hyper.pretraining_allDatafile)
                self.auxs.add_module(model_config.aux_name, model)

        self.block = Block(
            dim=self.aux_dim,
            num_heads = 2 if 'num_heads' not in self.hyper.dict else self.hyper.num_heads,
            mlp_ratio= 1 if 'mlp_ratio' not in self.hyper.dict else self.hyper.mlp_ratio,
            qkv_bias=True,
            qk_scale=None if 'qk_scale' not in self.hyper.dict else self.hyper.qk_scale,
            drop=0.5 if'drop' not in self.hyper.dict else self.hyper.drop,
            attn_drop=0.5 if 'attn_drop' not in self.hyper.dict else self.hyper.attn_drop,
            drop_path=0.5 if 'drop_path' not in self.hyper.dict else self.hyper.drop_path
        )

    def aux_infeature(self,x):
        with torch.no_grad():
            aux_out= []
            for aux in self.auxs:
                aux_logit = aux.logits(x)
                logit_mean_std = torch.stack(torch.std_mean(aux_logit, dim=1), dim=1)
                # aux_hidden = aux.feature_extract(x)
                aux_info = torch.cat([aux_logit, logit_mean_std], dim=1)
                aux_out.append(aux_info)

            aux_out = torch.stack(aux_out, dim=1)

        return aux_out

    def forward(self, x):
        # signal = x.detach().clone()
        aux_out = self.aux_infeature(x)
        aux_out = self.block(aux_out)
        aux_out = self.avgpool(aux_out)
        aux_out = aux_out.squeeze(2)
        # aux_out = aux_out[:,:, -1]


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

        x = torch.cat((x, aux_out), dim=1)


        logit = self.fc(x)
        # aux_feature = self.aux_mlp(aux_out)
        # logit = self.readout(x)

        return logit, regu_sum

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



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # if register_hook:
        #     self.save_attention_map(attn)
        #     attn.register_hook(self.save_attn_gradients)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        # x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # d_ff = dim * 4
        # self.conv1 = nn.Conv1d(in_channels=dim, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=dim, kernel_size=1)
        self.dropout = nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        self.activation = act_layer()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        x = x + self.attn(self.norm1(x), register_hook=register_hook)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        # x = self.norm2(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x