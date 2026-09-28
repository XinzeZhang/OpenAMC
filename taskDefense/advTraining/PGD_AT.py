import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm as real_tqdm
from models.nn._baseTrainer import AnnealingTrainer, EarlyStopping, time, AverageMeter
import logging
import random
from tqdm.auto import trange
from taskAttack.attackmethods.gradient.fgsm import PGD

from models.nn._baseNet import Opt
from taskAttack.util import signal_energy, signal_db2pow
import pandas as pd
from taskRecog.util import logit_acc

class attacker_opts(Opt):
    def __init__(self, init=None):
        super().__init__(init)
        self.PSR = -20
        self.epoch = 5
        self.norm = 'l2'
        self.warmup_epoch = 10
        # self.epsilon = 0.06
        # self.alpha = 0.03

    def update_psr_epsilon(self, signals):
        if self.norm != 'linfty':
            if len(signals.size()) == 2:
                signals = signals.unsqueeze(0)
            sig_energy = signal_energy(signals).mean()
            gain = signal_db2pow(self.PSR)
            epsilon = torch.sqrt(sig_energy * gain)
            alpha = epsilon / self.epoch

            self.epsilon = epsilon
            self.alpha = alpha

        return self.epsilon, self.alpha

class PGD_AT(AnnealingTrainer):
    def __init__(self, hyper, logger, **kwargs):
        super().__init__(hyper,logger)
        self.attacker_opts = attacker_opts()
        self.best_metric = 'adv' # must be 'nat' or 'adv'
        self.attacker_opts.update(kwargs)
        for key, value in self.attacker_opts.dict.items():
            logger.info(f"Attacker option {key}: {value}")

        self.warmup_epoch = self.attacker_opts.warmup_epoch


    def update_psr_epsilon(self, X_detached):
        if self.attacker_opts.norm != 'linfty':
            self.attacker.epsilon, self.attacker.alpha = self.attacker_opts.update_psr_epsilon(X_detached)

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.hyper.device)
        self.early_stopping = EarlyStopping(
            self.logger, patience=self.hyper.patience)

        T_0 = 1 if 'T_0' not in self.hyper.dict else self.hyper.T_0
        T_mult = 2 if 'T_mult' not in self.hyper.dict else self.hyper.T_mult

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0= T_0, T_mult=T_mult)

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0

        self.train_loss_list = []
        self.train_adv_acc_list = []
        self.train_nat_acc_list = []

        self.val_loss_list = []
        self.val_adv_acc_list = []
        self.val_nat_acc_list = []

        # self.criterion = nn.CrossEntropyLoss(
        #     reduction='none').to(self.hyper.device)  #return loss for each sample

    def before_train_step(self):
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()   #reconstruct at each epoch
        self.train_nat_acc = AverageMeter()    #reconstruct at each epoch
        self.train_adv_acc = AverageMeter()

        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_batch(self, data_batch, warmup = False):
        sig_batch, lab_batch = data_batch[0].to(self.hyper.device, non_blocking=True), data_batch[1].to(self.hyper.device, non_blocking=True)

        nat_logits, nat_loss = self.sig_logits_loss(sig_batch, lab_batch)
        nat_acc = logit_acc(nat_logits, lab_batch)
        self.train_nat_acc.update(nat_acc)
        if warmup:
            self.optimizer.zero_grad()
            nat_loss.backward()
            self.train_loss.update(0.0)
            self.train_adv_acc.update(0.0)
        else:
            self.attacker=PGD(model=self.model,logger=self.logger, **self.attacker_opts.dict)

            self.optimizer.zero_grad()
            adv_loss, adv_acc = self.cal_loss_acc(sig_batch, lab_batch)
            self.train_loss.update(adv_loss.item())
            self.train_adv_acc.update(adv_acc)

            adv_loss.backward()

        self.optimizer.step()

        metric_dict = {'train_loss_adv': self.train_loss.avg,
                            'train_acc_adv': self.train_adv_acc.avg,
                            'train_acc_nat': self.train_nat_acc.avg
                            }

        return metric_dict

    def warmup_train_step(self, data_batch):
        metric_dict = self.run_train_batch(data_batch, warmup=True)
        return metric_dict

    def run_train_step(self):
        is_warmup = self.iter <= self.warmup_epoch
        train_step = self.warmup_train_step if is_warmup else self.run_train_batch
        desc = (
            f'Warmup Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}'
            if is_warmup
            else f'Adversarial Training Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}'
        )
        metric_dict = {}

        with real_tqdm(total=len(self.train_loader),
                desc=desc,
                postfix=dict,
                mininterval=0.3,
                dynamic_ncols=True, colour='red', leave=False) as pbar:
            for i, data_batch in enumerate(self.train_loader):
                metric_dict = train_step(data_batch)
                self.scheduler.step(self.iter - 1 + i / len(self.train_loader))
                pbar.set_postfix(**metric_dict)
                pbar.update(1)

        return metric_dict

    def after_train_step(self):
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info('\033[31m' + '====> Epoch: {} Time: {:.2f}\tlr: {:.3E}\tTrain Loss: {:.3E}\tTrain Adv Acc: {:.3f}% \tTrain Nat Acc: {:.3f}%'.format(
            self.iter, time.time() - self.t_s,  self.lr_list[-1], self.train_loss.avg, self.train_adv_acc.avg * 100, self.train_nat_acc.avg * 100)+
    '\033[0m')

        self.train_loss_list.append(self.train_loss.avg)
        self.train_adv_acc_list.append(self.train_adv_acc.avg)
        self.train_nat_acc_list.append(self.train_nat_acc.avg)

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter() #reconstruct at each epoch
        self.val_adv_acc = AverageMeter() #reconstruct at each epoch
        self.val_nat_acc = AverageMeter() #reconstruct at each epoch
        # if logging:
        self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self):
        '''
        run validation step, return average adversarial loss and accuracy of adversarial validation examples at this epoch
        '''
        self.attacker=PGD(model=self.model,logger=self.logger, **self.attacker_opts.dict)
        with real_tqdm(total=len(self.val_loader),
                desc=f'Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}',
                postfix=dict,
                mininterval=0.3,
                colour='blue', leave=False) as pbar:
            for step, data_batch in enumerate(self.val_loader):
                sig_batch, lab_batch = data_batch[0].to(self.hyper.device, non_blocking=True), data_batch[1].to(self.hyper.device, non_blocking=True)

                self.update_psr_epsilon(sig_batch)
                delta = self.attacker(sig_batch, lab_batch)
                if torch.isnan(delta).any():
                    self.logger.error("NaN detected in delta! Setting NaNs to 0")
                    delta = delta.nan_to_num()
                adv_X = sig_batch + delta.to(self.hyper.device)

                adv_logits, adv_loss = self.sig_logits_loss(adv_X, lab_batch)
                adv_acc = logit_acc(adv_logits, lab_batch)

                self.val_loss.update(adv_loss.item())
                self.val_adv_acc.update(adv_acc)

                nat_acc = self.sig_acc(sig_batch, lab_batch)
                self.val_nat_acc.update(nat_acc)

                metric_dict = {'val_loss_adv': self.val_loss.avg,
                                    'val_acc_adv': self.val_adv_acc.avg,
                                    'val_acc_nat': self.val_nat_acc.avg}

                pbar.set_postfix(**metric_dict)
                pbar.update(1)

        return metric_dict

    def after_val_step(self):
        '''
        update and save best checkpoint, do EarlyStopping and adjust learning rate
        '''
        if self.best_metric == 'nat':
            self.early_stopping(self.val_nat_acc.avg)
            if self.val_nat_acc.avg >= self.best_monitor:
                self.best_monitor = self.val_nat_acc.avg
                self.best_epoch = self.iter
                torch.save(self.model.state_dict(), self.best_tmp_file)
        elif self.best_metric == 'adv':
            self.early_stopping(self.val_adv_acc.avg)
            if self.val_adv_acc.avg >= self.best_monitor:
                self.best_monitor = self.val_adv_acc.avg
                self.best_epoch = self.iter
                torch.save(self.model.state_dict(), self.best_tmp_file)
        else:
            raise ValueError(f"Unknown best metric: {self.best_metric}, must be 'nat' or 'adv'.")

        self.logger.info( '\033[34m' +
            '====> Epoch: {} Time: {:.2f}\tVal Loss: {:.3E}\tVal Adv Acc: {:.3f}%\tVal Nat Acc {:.3f}%'.format(self.iter, time.time() - self.t_s, self.val_loss.avg, self.val_adv_acc.avg * 100, self.val_nat_acc.avg * 100)
            + '\033[0m'
            )


        self.logger.info('Best Epoch: {} \t Best Val {} Acc: {:.3f}%'.format(self.best_epoch, self.best_metric.capitalize(),self.best_monitor * 100 ))

        self.adjust_lr()

        self.val_loss_list.append(self.val_loss.avg)
        self.val_adv_acc_list.append(self.val_adv_acc.avg)
        self.val_nat_acc_list.append(self.val_nat_acc.avg)

        self.epochs_stats = pd.DataFrame(
            data={"epoch": range(self.iter),
                  "lr_list": self.lr_list,
                  "train_loss": self.train_loss_list,
                  "val_loss": self.val_loss_list,
                  "train_nat_acc": self.train_nat_acc_list,
                  "val_nat_acc": self.val_nat_acc_list,
                  "train_adv_acc": self.train_adv_acc_list,
                  "val_adv_acc": self.val_adv_acc_list}
        )

    def cal_loss_acc(self, sig_batch, lab_batch):
        device = self.hyper.device
        X_detached = sig_batch.detach().clone().requires_grad_(True)
        y_detached = lab_batch.detach().clone()

        X_detached = X_detached.to(device)
        y_detached = y_detached.to(device)

        # 计算对抗扰动
        # Due to the X_detached are shuffled, the corr. snr values are also shuffled. So Need for each X, re-calculate the gain -> epsilon.
        # Strictly, we should re-calculate the epsilon for each example. However, this implementation would be extremely time consuming. And the final performance is not sensitive to this epsilon per example.
        # Therefore, we re-calculate the epsilon for each batch.
        self.update_psr_epsilon(X_detached)
            # print(self.attacker.epsilon, self.attacker.alpha)
        delta = self.attacker(X_detached,y_detached)
        # After generating the perturbation, the model must be set at training mode before calcuating the loss.
        self.model.train()
            ## for each example
            # for i in range(X_detached.shape[0]):
            #     self.attacker_opts.update_psr_epsilon(X_detached[i])
            #     self.attacker.epsilon = self.attacker_opts.epsilon
            #     self.attacker.alpha = self.attacker_opts.alpha
            #     i_X, i_Y = X_detached[i].unsqueeze(0), y_detached[i].unsqueeze(0)
            #     i_delta = self.attacker(i_X, i_Y)
            #     delta[i] = i_delta[0]
        if torch.isnan(delta).any():
            self.logger.error("NaN detected in delta! Setting NaNs to 0")
            delta = delta.nan_to_num()

        # 生成对抗样本
        adv_X = X_detached + delta.to(device)

        # 计算对抗损失
        adv_logits, adv_loss = self.sig_logits_loss(adv_X, lab_batch)
        adv_acc = logit_acc(adv_logits, lab_batch)

        return adv_loss, adv_acc



# uncomment the following lines to make unit test for PGD adversarial training, and set the model and dataset in the main function below.
# from models.nn.AWN import AWN_config as awn
if __name__ == "__main__":
    from taskDefense.Parser import get_parser
    from taskDefense.Wrapper import Task

    import os
    from pathlib import Path

    args, parser = get_parser(parsing=True)

    args.data = 'dr4'
    args.model = 'awn'
    args.test = True
    args.clean = True
    project_root = os.getcwd()
    args.model_init_path = str(Path(__file__).resolve().relative_to(project_root))
    # args.gid = 1
    args.defense_method = 'pgdat'
    # args.exp_name = 'defense_{method}_unitTest'.format(method=args.defense_method)
    # args.model = 'awn'
    args.snr = [0,10]
    args.algo = 'mi'
    args.psr = -20
    args.cuda = True
    # uncomment the following lines to set pretraining file, if not set, the model will be trained with the given defense method
    args.gid = 1
    # args.hyper = Opt()

    task = Task(args, parser)
    task.conduct(eval=True,ave_confMax=False, show_variance=False)
