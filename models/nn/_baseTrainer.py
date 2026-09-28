import os.path
import time

import pandas as pd
import torch
from torch import optim, nn

from tqdm.auto import trange
from tqdm.auto import tqdm as real_tqdm
# from torch.optim import lr_scheduler

import numpy as np
from taskRecog.util import os_makedirs, logit_acc

Red = "\033[31m"
Blue = "\033[34m"
Reset = "\033[0m"


class Trainer:
    def __init__(self, hyper, logger, **kwargs):
        super(Trainer, self).__init__()

        self.epochs_stats = None
        self.val_acc_list = None
        self.val_loss_list = None
        self.train_acc_list = None
        self.train_loss_list = None
        self.val_acc = None
        self.val_loss = None
        self.train_acc = None
        self.best_monitor = None
        self.lr_list = None
        self.train_loss = None
        self.t_s = None
        self.early_stopping = None
        self.criterion = None
        self.optimizer = None

        self.hyper = hyper
        self.logger = logger

        self.iter = 1
        self.checkpoint_folder = os.path.join(self.hyper.model_fit_dir, "checkpoint")
        os_makedirs(self.checkpoint_folder)

        self.best_file = os.path.join(
            self.checkpoint_folder,
            f"{self.hyper.data_name}_{self.hyper.model_name}.best.pt",
        )
        self.best_tmp_file = self.best_file.replace(".best.pt", ".best_tmp.pt")
        self.finish = True if os.path.exists(self.best_file) else False

    def loop(
        self,
        model,
        train_loader,
        val_loader,
    ):
        self.logger.info(f"Using Trainer: {self.__class__.__name__}")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(self.hyper.device)

        self.before_train()

        for self.iter in trange(1, self.hyper.epochs + 1):
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            self.before_val_step()
            self.run_val_step()
            self.after_val_step()
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        # move the self.best_tmp_file to self.best_file
        os.replace(self.best_tmp_file, self.best_file)
        self.logger.info(f"Best model saved to {self.best_file} at epoch {self.best_epoch}")


    @staticmethod
    def adjust_learning_rate(optimizer, gamma):
        """Sets the learning rate when we have to, using learing rate decay"""
        lr = optimizer.param_groups[0]["lr"] * gamma
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def before_train(self):
        """
        initialize optimizer, loss function, EarlyStopping, some training process lists
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.hyper.device)
        self.early_stopping = EarlyStopping(self.logger, patience=self.hyper.patience)

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def cal_ori_loss(self, sig_batch, lab_batch):
        """
        forward, calculate loss
        """
        _, loss = self.sig_logits_loss(sig_batch, lab_batch)
        return loss

    def sig_acc(self, sig_batch, lab_batch):
        """
        forward, calculate acc
        """
        logit, _ = self.sig_logits_loss(sig_batch, lab_batch)

        acc = logit_acc(logit, lab_batch)
        return acc

    def sig_logits_loss(self, sig_batch, lab_batch):
        """
        forward, calculate loss and logits
        """
        # sig_batch = sig_batch.to(self.hyper.device)
        # lab_batch = lab_batch.to(self.hyper.device)
        if "AWN" in self.hyper.class_name:
            logit, regu_sum = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss += sum(regu_sum)
        else:
            logit = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)

        return logit, loss

    def cal_loss_acc(self, data_batch):
        """
        forward, calculate loss and accuracy
        """
        sig_batch, lab_batch = data_batch[0], data_batch[1]
        sig_batch = sig_batch.to(self.hyper.device, non_blocking=True)
        lab_batch = lab_batch.to(self.hyper.device, non_blocking=True)
        logit, loss = self.sig_logits_loss(sig_batch, lab_batch)

        acc = logit_acc(logit, lab_batch)

        return loss, acc

    def before_train_step(self):
        self.model.train()
        self.t_s = time.time()
        self.train_loss = AverageMeter()  # reconstruct at each epoch
        self.train_acc = AverageMeter()  # reconstruct at each epoch
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_batch(self, i, data_batch):
        # sig_batch, lab_batch = data_batch[0], data_batch[1]
        self.optimizer.zero_grad()
        loss, acc = self.cal_loss_acc(data_batch)
        loss.backward()
        self.optimizer.step()

        self.train_loss.update(loss.item())
        self.train_acc.update(acc)

    def run_train_step(self):
        """
        run training step, return average loss and accuracy at this epoch
        """

        desc = (
            f'Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}'
        )

        with real_tqdm(
            total=len(self.train_loader),
            desc=desc,
            postfix=dict,
            mininterval=0.3,
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for step, data_batch in enumerate(self.train_loader):

                self.run_train_batch(step, data_batch)

                pbar.set_postfix(
                    **{
                        "train_loss": self.train_loss.avg,
                        "train_acc": self.train_acc.avg,
                    }
                )
                pbar.update(1)

        return self.train_loss.avg, self.train_acc.avg

    def after_train_step(self):
        """
        print log at this epoch, save average loss and accuracy to self.train_loss_list and self.train_acc_list
        """
        self.lr_list.append(self.optimizer.param_groups[0]["lr"])
        self.logger.info(
            "{}====> Epoch: {} Time: {:.2f}\tlr: {:.3E}\tTrain Loss: {:.3E}\tTrain Acc: {:.3f}% {}".format(
                Red,
                self.iter,
                time.time() - self.t_s,
                self.lr_list[-1],
                self.train_loss.avg,
                self.train_acc.avg * 100,
                Reset,
            )
        )
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)

    def before_val_step(self, logging=True):
        self.model.eval()
        self.t_s = time.time()
        self.val_loss = AverageMeter()  # reconstruct at each epoch
        self.val_acc = AverageMeter()  # reconstruct at each epoch
        if logging:
            self.logger.info(f"Starting validation epoch {self.iter}:")

    def run_val_step(self):
        """
        run validation step, return average loss and accuracy at this epoch
        """

        with real_tqdm(
            total=len(self.val_loader),
            desc=f"Epoch {self.iter}/{self.hyper.epochs}",
            postfix=dict,
            mininterval=0.3,
            colour="blue",
            dynamic_ncols=True,
            leave=False,
        ) as pbar:
            for step, data_batch in enumerate(self.val_loader):
                with torch.no_grad():
                    loss, acc = self.cal_loss_acc(data_batch)

                    self.val_loss.update(loss.item())
                    self.val_acc.update(acc)

                    pbar.set_postfix(
                        **{"val_loss": self.val_loss.avg, "val_acc": self.val_acc.avg}
                    )
                    pbar.update(1)

        return self.val_loss.avg, self.val_acc.avg

    def adjust_lr(self):
        """
        decrease learing rate when early_stopping.counter reach milestone_step
        """
        if (
            self.early_stopping.counter != 0
            and self.early_stopping.counter % self.hyper.milestone_step == 0
        ):
            history_lr = self.optimizer.param_groups[0]["lr"]
            self.adjust_learning_rate(self.optimizer, self.hyper.gamma)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Learning rate decreased ({history_lr:.3E} --> {current_lr:.3E})."
            )

    def after_val_step(self, checkpoint=True):
        """
        update and save best checkpoint, do EarlyStopping and adjust learning rate
        """
        if self.val_acc.avg >= self.best_monitor:
            self.best_monitor = self.val_acc.avg
            self.best_epoch = self.iter
            # toDo: change to annother location.
            if checkpoint:
                torch.save(self.model.state_dict(), self.best_tmp_file)

        self.logger.info(
            f"{Blue}====> Epoch: {self.iter} Time: {time.time() - self.t_s:.2f}\tVal Loss: {self.val_loss.avg:.3E}\tVal Acc: {self.val_acc.avg * 100:.3f}%{Reset}"
        )

        self.early_stopping(self.val_acc.avg)
        self.logger.info(
            "Best Epoch: {} \t Best Val Acc: {:.3f}%".format(
                self.best_epoch, self.best_monitor * 100
            )
        )

        self.adjust_lr()

        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)

        self.epochs_stats = pd.DataFrame(
            data={
                "epoch": range(self.iter),
                "lr_list": self.lr_list,
                "train_loss": self.train_loss_list,
                "val_loss": self.val_loss_list,
                "train_acc": self.train_acc_list,
                "val_acc": self.val_acc_list,
            }
        )


class AnnealingTrainer(Trainer):
    """
    Extend from the Trainer class and use the CosineAnnealingWarmRestarts learning rate scheduling strategy
    """

    def __init__(self, hyper, logger, **kwargs):
        super().__init__(hyper, logger, **kwargs)

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
        self.criterion = nn.CrossEntropyLoss().to(self.hyper.device)
        self.early_stopping = EarlyStopping(self.logger, patience=self.hyper.patience)

        T_0 = 1 if "T_0" not in self.hyper.dict else self.hyper.T_0
        T_mult = 2 if "T_mult" not in self.hyper.dict else self.hyper.T_mult

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult
        )

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def run_train_batch(self, i, data_batch):

        self.optimizer.zero_grad()
        loss, acc = self.cal_loss_acc(data_batch)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(self.iter - 1 + i / len(self.train_loader))

        self.train_loss.update(loss.item())
        self.train_acc.update(acc)

    def run_train_step(
        self,
    ):

        desc = (
            f'Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}'
        )

        with real_tqdm(
            total=len(self.train_loader),
            desc=desc,
            postfix=dict,
            mininterval=0.3,
            dynamic_ncols=True,
        ) as pbar:
            for step, data_batch in enumerate(self.train_loader):
                self.run_train_batch(step, data_batch)

                pbar.set_postfix(
                    **{
                        "train_loss": self.train_loss.avg,
                        "train_acc": self.train_acc.avg,
                    }
                )
                pbar.update(1)

        return self.train_loss.avg, self.train_acc.avg

    def adjust_lr(self):
        # history_lr = self.optimizer.param_groups[0]['lr']
        # self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(f"Learning rate: ({current_lr:.3E}).")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, logger, patience=7, delta=0):
        """
        Args:
            logger: log the info to a .txt
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            counter(int): How many epochs has the validation score not improved
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_acc):

        score = val_acc * 100

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f"Validation accuracy increased ({self.val_acc_max:.3f}% --> {score:.3f})%."
            )
            self.val_acc_max = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f"Validation accuracy increased ({self.val_acc_max:.3f}% --> {score:.3f})%."
            )
            self.val_acc_max = score
            self.counter = 0
