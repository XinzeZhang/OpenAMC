import os.path
import time

import pandas as pd
import torch
from torch import optim, nn

from tqdm.auto import trange
from tqdm.auto import tqdm as real_tqdm
# from torch.optim import lr_scheduler

import numpy as np
from taskRecog.util import os_makedirs
from models.nn._baseTrainer import AverageMeter, EarlyStopping
from models.nn._baseTrainer import Trainer as baseTrainer



class Trainer(baseTrainer):

    def before_train(self):
        super().before_train()
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.hyper.device)
        # renew loss with none reduction

    def cal_loss_acc(self, sig_batch, lab_batch, lab_weight):
        sig_batch = sig_batch.to(self.hyper.device)
        lab_batch = lab_batch.to(self.hyper.device)
        lab_weight = lab_weight.to(self.hyper.device)
        if 'AWN' in self.hyper.class_name:
            logit, regu_sum = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss = loss * lab_weight
            loss = torch.mean(loss)
            loss += sum(regu_sum)
        else:
            logit = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss = loss * lab_weight
            loss = torch.mean(loss)

        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ).item() / lab_batch.size(0)

        return loss, acc

    def run_train_step(self,):

        with real_tqdm(total=len(self.train_loader),
                    desc=f'Epoch {self.iter}/{self.hyper.epochs}',
                    postfix=dict,
                    mininterval=0.3) as pbar:
                for step, data_batch in enumerate(self.train_loader):
                    self.run_optim_step(step, data_batch)
                    pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                        'train_acc': self.train_acc.avg})
                    pbar.update(1)

        return self.train_loss.avg, self.train_acc.avg

    def run_optim_step(self, i, data_batch):
        sig_batch, lab_batch, lab_weight = data_batch
        loss, acc = self.cal_loss_acc(sig_batch, lab_batch, lab_weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss.update(loss.item())
        self.train_acc.update(acc)

    def run_val_step(self,):

        with real_tqdm(total=len(self.val_loader),
                desc=f'Epoch {self.iter}/{self.hyper.epochs}',
                postfix=dict,
                mininterval=0.3,
                colour='blue') as pbar:
            for step, data_batch in enumerate(self.val_loader):
                with torch.no_grad():
                    sig_batch, lab_batch, lab_weight = data_batch
                    loss, acc = self.cal_loss_acc(sig_batch, lab_batch, lab_weight)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)

                    pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                        'val_acc': self.val_acc.avg})
                    pbar.update(1)

        return self.val_loss.avg, self.val_acc.avg


class AnnealingTrainer(Trainer):
    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.hyper.device)
        self.early_stopping = EarlyStopping(
            self.logger, patience=self.hyper.patience)

        T_0 = 1 if 'T_0' not in self.hyper.dict else self.hyper.T_0
        T_mult = 2 if 'T_mult' not in self.hyper.dict else self.hyper.T_mult

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0= T_0, T_mult=T_mult)

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def run_optim_step(self, i, data_batch):
        sig_batch, lab_batch, lab_weight = data_batch
        loss, acc = self.cal_loss_acc(sig_batch, lab_batch, lab_weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(self.iter-1 + i / len(self.train_loader))

        self.train_loss.update(loss.item())
        self.train_acc.update(acc)

    def adjust_lr(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(
            f'Learning rate: ({current_lr:.3E}).')

class NormAnnealingTrainer(AnnealingTrainer):
    def cal_loss_acc(self, sig_batch, lab_batch, lab_weight):
        if 'AWN' in self.hyper.class_name:
            logit, regu_sum = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss = (loss * lab_weight / lab_weight.sum()).sum()
            loss = torch.mean(loss)
            loss += sum(regu_sum)
        else:
            logit = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss = (loss * lab_weight / lab_weight.sum()).sum()
            loss = torch.mean(loss)

        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        )/ lab_batch.size(0)
        acc = acc.item()

        return loss, acc