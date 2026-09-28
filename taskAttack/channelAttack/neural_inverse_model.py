"""Neural Inverse Model used by FIM."""

import importlib

import pandas as pd
import torch
import torch.nn as nn

from models.nn._baseNet import BaseNet
from models.nn._baseTrainer import AnnealingTrainer


MODEL_ZOO = {
    "nim": ("taskAttack.channelAttack.neural_inverse_model", "NeuralInverseModel"),
}


def load_nim_model_class(name):
    if name not in MODEL_ZOO:
        raise ValueError(f"Unsupported NIM architecture: {name}")
    module_path, class_name = MODEL_ZOO[name]
    return getattr(importlib.import_module(module_path), class_name)


class NeuralInverseModel(BaseNet):
    def initialize_arch(self):
        self.nim = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 2, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.nim(x.to(self.hyper.device)).nan_to_num()

    def loader_predict(self, data_loader):
        predictions, labels = [], []
        with torch.no_grad():
            for signals, batch_labels in data_loader:
                predictions.append(self.forward(signals).cpu())
                labels.append(batch_labels)
        return torch.cat(predictions), torch.cat(labels)


class ChannelTrainer(AnnealingTrainer):
    def cal_loss_acc(self, data_batch):
        sig_batch, lab_batch = data_batch[:2]
        prediction = self.model(sig_batch.to(self.hyper.device))
        loss = self.criterion(prediction, lab_batch.to(self.hyper.device))
        return loss, loss.item()

    def before_train(self):
        from torch import optim

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
        self.criterion = nn.MSELoss().to(self.hyper.device)
        self.early_stopping = EarlyStopping(self.logger, self.hyper.patience)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.hyper.dict.get("T_0", 1),
            T_mult=self.hyper.dict.get("T_mult", 2),
        )
        self.lr_list = []
        self.best_monitor = float("inf")
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def after_train_step(self):
        self.lr_list.append(self.optimizer.param_groups[0]["lr"])
        self.train_loss_list.append(self.train_loss.avg)
        self.train_acc_list.append(self.train_acc.avg)

    def after_val_step(self, checkpoint=True):
        if self.val_acc.avg <= self.best_monitor:
            self.best_monitor = self.val_acc.avg
            self.best_epoch = self.iter
            if checkpoint:
                torch.save(self.model.state_dict(), self.best_tmp_file)
        self.early_stopping(self.val_acc.avg)
        self.adjust_lr()
        self.val_loss_list.append(self.val_loss.avg)
        self.val_acc_list.append(self.val_acc.avg)
        self.epochs_stats = pd.DataFrame(
            {
                "epoch": range(self.iter),
                "lr_list": self.lr_list,
                "train_loss": self.train_loss_list,
                "val_loss": self.val_loss_list,
                "train_acc": self.train_acc_list,
                "val_acc": self.val_acc_list,
            }
        )


class EarlyStopping:
    def __init__(self, logger, patience=7, delta=0):
        self.logger = logger
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, value):
        if self.best_score is None or value <= self.best_score + self.delta:
            self.best_score = value
            self.counter = 0
            return
        self.counter += 1
        self.logger.info(
            "EarlyStopping counter: %s out of %s", self.counter, self.patience
        )
        if self.counter >= self.patience:
            self.early_stop = True
