from taskRecog.util import Opt, chunk_list_nsub
import torch
from models.nn._baseTrainer import AnnealingTrainer,EarlyStopping,AverageMeter
from tqdm.auto import trange
import numpy as np
from tqdm.auto import tqdm
from torch import optim, nn
import time
import pandas as pd

def loss_calweight(tag, step_idxs):
    mask_ratio = 1 - np.sum(np.array(step_idxs.cpu())) / len(np.array(step_idxs.cpu()).flatten().tolist())

    if tag == 'mratio':
        r = mask_ratio
    elif tag == 'plus':
        r = 1 + mask_ratio
    elif tag == 'minus':
        r = 1 - mask_ratio
    elif tag == 'inverse':
        assert step_idxs is not None
        unitM = torch.ones_like(step_idxs)
        change = unitM - step_idxs
        r = unitM.abs().sum() / (unitM.abs().sum() + change.abs().sum())
        r = r.item()
    else:
        r = 1
    return mask_ratio, r

def eval_samples(model, sample_list, label_list):
    pre_lab_all = []
    label_all = []
    # loop of SNRs in test_sample_list
    for (Sample, Label) in tqdm(zip(sample_list, label_list), total=len(sample_list)):
        pred_i = []
        label_i = []
        for (sample, label) in zip(Sample, Label):
            pre_lab = model.predict(sample)
            pred_i.append(pre_lab)
            label_i.append(label)
        pred_i = np.concatenate(pred_i)
        label_i = np.concatenate(label_i)

        pre_lab_all.append(pred_i)
        label_all.append(label_i)

    return pre_lab_all, label_all

class Sampler(Opt):
    def __init__(self, init=None):
        super().__init__(init)
        self.local_dir = ''
        self.name = ''
        self.num_env = init.num_env

    def env_config(self, snrs, logger):
        num_snrs = len(snrs)
        if not  num_snrs % self.num_env == 0:
            logger.warning(f'Unequal class with num_env: {self.num_env} for total {num_snrs} SNR conditions.')

        snr_dict = {}

        env_list = chunk_list_nsub(snrs, self.num_env)

        for i, period in enumerate(env_list):
            logger.info(f'{i}-th snr env class with snr conditions: {period}')
            for snr in period:
                snr_dict[snr] = i

        self.snr_dict = snr_dict
        self.env_list = env_list


class AddMask(torch.nn.Module):
    def __init__(self, mask_size = (128,), mask_type = 'soft'):
        super().__init__()
        # self.a = torch.nn.Parameter(torch.ones(mask_size))
        self.mask_size = mask_size
        self.mask_type = mask_type

        if mask_type == 'hard':
            self.act = torch.nn.Sigmoid()
            self.a = torch.nn.Parameter(torch.ones(mask_size))
        elif mask_type == 'clip':
            self.a = torch.nn.Parameter(torch.randn(mask_size))
        else:
            self.a = torch.nn.Parameter(torch.rand(mask_size)*2)

    def hard_mask(self, a):
        # soft-> 0,1 with prob.
        b = torch.rand_like(a.data).to(a.device)
        maskcode = torch.zeros_like(b).to(a.device)
        activate_a = torch.empty_like(b).to(a.device)
        activate_a[:,] = self.act(a)

        maskcode[activate_a >= b] = 1
        return maskcode.float()

    def forward(self, x):
        if self.mask_type == 'hard':
            maskcode = self.hard_mask(self.a)
        else:
            if self.mask_type == 'clip':
                self.a.data = torch.clamp(self.a.data, 0, 1)

            maskcode = self.a
        m_x = x.mul(maskcode)
        return m_x

    def get_mask(self,):
        if self.mask_type == 'hard':
            maskcode = self.hard_mask(self.a)
        else:
            if self.mask_type == 'clip':
                maskcode = torch.clamp(self.a.data, 0, 1)
            else:
                maskcode = self.a.data

        return maskcode.detach().clone()

# def hard_mask(x, code):
#     b = torch.rand_like(x).to(code.device)
#     maskcode = torch.zeros_like(b).to(code.device)
#     activate_a = torch.empty_like(b).to(code.device)
#     activate_a[:,] = torch.sigmoid(code)
#     maskcode[activate_a >= b] = 1

#     return maskcode.float()

class MaskTrainer(AnnealingTrainer):
    def __init__(self, hyper, logger, model, mask):
        super().__init__(hyper,logger)
        self.model = model.to(self.hyper.device)
        self.mask = mask.to(self.hyper.device)

    def loop(self,
                 train_loader,
                 val_loader, ):
        self.logger.info(f'Using Trainer: {self.__class__.__name__}')
        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.model = model.to(self.hyper.device)
        # self.mask = mask.to(self.hyper.device)

        self.before_train()

        for self.iter in trange(1, self.hyper.epochs + 1):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info('Early stopping')
                break

        # last_model_name = self.hyper.data_name + '_' + \
        #     f'{self.hyper.model_name}' + '.early_stop.pt'
        self.early_stopping.early_stop = True
        torch.save(self.early_stopping.early_stop, self.finishTag_file)

    def before_train(self):
        self.optimizer = optim.Adam(self.mask.parameters(), lr=self.hyper.lr)
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
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def before_train_step(self):
        self.mask.train()
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()   #reconstruct at each epoch
        self.train_acc = AverageMeter()    #reconstruct at each epoch
        self.logger.info(f"Starting training epoch {self.iter}:")

    def cal_loss_acc(self, sig_batch, lab_batch):
        '''
        forward, calculate loss and accuracy
        '''
        if 'AWN' in self.hyper.class_name:
            logit, regu_sum = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss += sum(regu_sum)
        else:
            logit = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)

        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ) / lab_batch.size(0)
        acc = acc.item()

        return loss, acc

    def run_train_step(self,):
        '''
        run training step, return average loss and accuracy at this epoch
        '''
        with tqdm(total=len(self.train_loader),
                desc=f'Epoch {self.iter}/{self.hyper.epochs}',
                postfix=dict,
                mininterval=0.3) as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.train_loader):
                sig_batch = sig_batch.to(self.hyper.device)
                lab_batch = lab_batch.to(self.hyper.device)
                sig_batch = self.mask(sig_batch)
                loss, acc = self.cal_loss_acc(sig_batch, lab_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(self.iter-1 + step / len(self.train_loader))

                self.train_loss.update(loss.item())
                self.train_acc.update(acc)

                pbar.set_postfix(**{'train_loss': self.train_loss.avg,
                                    'train_acc': self.train_acc.avg})
                pbar.update(1)

        return self.train_loss.avg, self.train_acc.avg

    def after_train_step(self):
        '''
        print log at this epoch, save average loss and accuracy to self.train_loss_list and self.train_acc_list
        '''
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info('====> Epoch: {} Time: {:.2f}\tlr: {:.4E}\tTrain Loss: {:.6E}\tTrain Acc: {:.3f}% '.format(
            self.iter, time.time() - self.t_s,  self.lr_list[-1], self.train_loss.avg, self.train_acc.avg * 100))
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)

    def before_val_step(self, logging =True):
        self.mask.eval()
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter() #reconstruct at each epoch
        self.val_acc = AverageMeter() #reconstruct at each epoch
        if logging:
            self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self,):
        '''
        run validation step, return average loss and accuracy at this epoch
        '''
        with tqdm(total=len(self.val_loader),
                desc=f'Epoch {self.iter}/{self.hyper.epochs}',
                postfix=dict,
                mininterval=0.3,
                colour='blue') as pbar:
            for step, (sig_batch, lab_batch) in enumerate(self.val_loader):
                with torch.no_grad():
                    sig_batch = sig_batch.to(self.hyper.device)
                    lab_batch = lab_batch.to(self.hyper.device)
                    sig_batch = self.mask(sig_batch)
                    loss, acc = self.cal_loss_acc(sig_batch, lab_batch)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)

                    pbar.set_postfix(**{'val_loss': self.val_loss.avg,
                                        'val_acc': self.val_acc.avg})
                    pbar.update(1)

        return self.val_loss.avg, self.val_acc.avg

    def after_val_step(self, checkpoint = True):
        '''
        update and save best checkpoint, do EarlyStopping and adjust learning rate
        '''
        if self.val_acc.avg >= self.best_monitor:
            self.best_monitor = self.val_acc.avg
            self.best_epoch = self.iter
            # toDo: change to annother location.
            if checkpoint:
                torch.save(self.mask.state_dict(), self.best_file)

        self.logger.info(
            '====> Epoch: {} Time: {:.2f}\tVal Loss: {:.6E}\tVal Acc: {:.3f}%'.format(self.iter, time.time() - self.t_s, self.val_loss.avg, self.val_acc.avg * 100))

        self.early_stopping(self.val_acc.avg)
        self.logger.info('Best Epoch: {} \t Best Val Acc: {:.3f}%'.format(self.best_epoch, self.best_monitor * 100 ))

        self.adjust_lr()

        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)

        self.epochs_stats = pd.DataFrame(
            data={"epoch": range(self.iter),
                  "lr_list": self.lr_list,
                  "train_loss": self.train_loss_list,
                  "val_loss": self.val_loss_list,
                  "train_acc": self.train_acc_list,
                  "val_acc": self.val_acc_list}
        )