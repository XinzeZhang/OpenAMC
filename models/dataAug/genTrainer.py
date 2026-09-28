import torch
import torch.nn as nn
from torch import optim

from models.nn._baseTrainer import AnnealingTrainer, EarlyStopping


import random
from taskRecog.util import chunk_list_nsub

'''
 Refer to G. Dong and H. Liu, “Signal Augmentations Oriented to Modulation Recognition in the Realistic Scenarios,” IEEE Transactions on Communications, vol. 71, no. 3, pp. 1665–1677, Mar. 2023, doi: 10.1109/TCOMM.2023.3236379.
'''
class common_Trainer_SigGEN(AnnealingTrainer):
    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
        self.criterion = nn.CrossEntropyLoss(
            reduction='none').to(self.hyper.device)
        self.early_stopping = EarlyStopping(
            self.logger, patience=self.hyper.patience)

        T_0 = 1 if 'T_0' not in self.hyper.dict else self.hyper.T_0
        T_mult = 2 if 'T_mult' not in self.hyper.dict else self.hyper.T_mult

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult)

        self.lr_list = []
        self.best_monitor = 0.0
        self.best_epoch = 0
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []

    def cal_ori_loss(self, sig_batch, lab_batch):
        sig_batch = sig_batch.to(self.hyper.device)
        lab_batch = lab_batch.to(self.hyper.device)
        if 'AWN' in self.hyper.class_name:
            logit, regu_sum = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
            loss += sum(regu_sum)
        else:
            logit = self.model(sig_batch)
            loss = self.criterion(logit, lab_batch)
        # regu_tensor = regu_sum[0].repeat(loss.size(0))
        # loss += regu_tensor
        return loss

    def SegCS(self, X, y, n=3):
        batch_size = X.size(0)
        sig_length = X.size(2)

        new_batch = []
        for i_b in range(batch_size):
            example = X[i_b, :, :]

            indices = list(range(sig_length))

            # seg = random.sample(indices, n-1)
            # seg.sort()
            # seg.append(sig_length)

            # idx = []
            # left = 0
            # for i in seg:
            #     idx.append(indices[left:i])
            #     left = i
            # indices = idx

            # 取索引切片
            indices = chunk_list_nsub(indices, n)

            # 论文中原文是可能是要进行如下flip操作，参考 L. Huang et al., “Data Augmentation for Deep Learning-Based Radio Modulation Classification,” IEEE Access, vol. 8, pp. 1498–1506, 2020. 具体如何flip并未讲清楚，也没有给出代码或参考论文。另外根据SigGEN论文的Fig 13所示，所有示例均没有进行flip操作，因此，在代码实现中有待商榷。


            # 随机取第n个位置，循环移位
            id_shift = torch.randint(0, n, (1,)).item()
            dir_shift = random.choice([-1, 1])  # dir_shift:移动方向（1：右移，-1:左移）
            move_id = dir_shift*id_shift
            indices_seg = list(range(n))
            indices_seg = indices_seg[move_id:] + indices_seg[:move_id]

            # 拼接索引
            indices_new = []
            for i in indices_seg:
                indices_new += indices[i]

            example = example[:, indices_new]
            operator = torch.ones_like(example)
            id_flip = torch.randint(0, n, (1,)).item()
            # id_flip = id_shift
            operator[:, indices[id_flip]] = -1
            example = example.mul(operator)

            new_batch.append(example)

        # 在索引位置操作
        X_SegCS = torch.stack(new_batch).to(self.hyper.device)

        y_SegCS = y

        # 计算X_SegCS的loss
        loss_SegCS = self.cal_ori_loss(X_SegCS, y_SegCS)

        return loss_SegCS

    def SegRE(self, X, y):
        sig_length = X.size()[2]

        def addnoise(X, std_dev_tensor=1):
            length = random.choice(list(range(32, 40)))  # 长度
            indices = random.choice(list(range(sig_length-length)))  # 起点
            X1, X2, X3 = torch.split(X, split_size_or_sections=[
                                     indices, length, sig_length-length-indices], dim=-1)
            X2_mean = torch.mean(X2, dim=-1, keepdim=True)
            X2_mean = X2_mean.repeat(1, 1, length)
            noise = torch.normal(X2_mean, std_dev_tensor)
            X2_new = X2_mean*noise
            X_new = torch.cat((X1, X2_new, X3), dim=-1)
            return X_new

        n = random.choice([2, 3, 5])
        X_new = X
        for i in range(n):
            X_new = addnoise(X_new)
        X_SegRE = X_new
        y_SegRE = y

        # 计算X_SegRE的loss
        loss_SegRE = self.cal_ori_loss(X_SegRE, y_SegRE)

        return loss_SegRE

    def SigPC(self, X, y, loss, alpha=0.8, beta=0.8):
        batch_size = X.size()[0]
        rand_indices = torch.randperm(batch_size)
        loss_new = loss[rand_indices]
        # the loss of aug. example is calculated by eq.8 in the original paper.
        # Note!!!!, The filp operation is not neccesary due the eq.8 : ls
        # X_flip = torch.flip(X, dims=[2])
        # y_flip=y
        # logit_flip, regu_sum_flip = self.model(X_flip)
        # loss_flip = self.criterion(logit_flip, y_flip)
        # loss_flip += sum(regu_sum_flip)
        # beta_dist = torch.distributions.Beta(alpha,beta)
        # lamb = beta_dist.sample((batch_size,))
        # loss_SigPC=lamb*loss_flip+(1-lamb)*loss_new
        # return loss_SigPC
        beta_dist = torch.distributions.Beta(alpha, beta)
        lamb = beta_dist.sample((batch_size,))
        lamb = lamb.to(self.hyper.device)
        loss_SigPC = lamb*loss+(1-lamb)*loss_new
        return loss_SigPC

    def SigMC(self, X, loss, n=4):
        batch_size = X.size(0)
        sig_length = X.size(2)

        index_sample = []
        index_length = []

        # 取索引切片
        indices = list(range(sig_length))
        indices = chunk_list_nsub(indices, n)

        len_indices = [len(i_seg) for i_seg in indices]

        for i in range(batch_size):
            index = random.sample(range(batch_size), n)
            index_sample.append(index)
            random.shuffle(len_indices)
            index_length.append(len_indices)

        index_sample = torch.tensor(index_sample)
        index_sample = index_sample.to(self.hyper.device)

        index_length = torch.tensor(index_length)
        index_length = index_length.to(self.hyper.device)
        index_length = index_length*1.0/sig_length

        loss = loss[index_sample]*index_length
        loss = loss.sum(dim=-1)

        return loss

    def cal_acc(self, sig_batch, lab_batch):
        sig_batch = sig_batch.to(self.hyper.device)
        lab_batch = lab_batch.to(self.hyper.device)
        if 'AWN' in self.hyper.class_name:
            logit, _ = self.model(sig_batch)
        else:
            logit = self.model(sig_batch)
        pre_lab = torch.argmax(logit, 1)
        acc = torch.sum(pre_lab == lab_batch.data).double(
        ).item() / lab_batch.size(0)
        return acc

    def cal_loss(self, ori_loss, aug_loss):
        # loss=(ori_loss+aug_loss)/2
        loss = torch.cat([ori_loss, aug_loss])
        loss = torch.mean(loss)
        return loss

    def cal_loss_acc(self, sig_batch, lab_batch):
        def sigGEN(X, y, ori_loss):
            loss1 = self.SegCS(X, y)
            loss2 = self.SegRE(X, y)
            loss3 = self.SigPC(X, y, ori_loss)
            loss4 = self.SigMC(X, ori_loss)

            loss0 = torch.stack((loss1, loss2, loss3, loss4), dim=0)

            batch_size = batch_size = X.size()[0]
            mask_tensor = torch.zeros(4, batch_size)
            for j in range(batch_size):
                i = random.randint(0, 3)
                mask_tensor[i][j] = 1

            mask_tensor = mask_tensor.to(self.hyper.device)

            loss0 = loss0*mask_tensor
            loss_sigGEN = torch.sum(loss0, dim=0)
            return loss_sigGEN

        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss_sigGEN = sigGEN(sig_batch, lab_batch, ori_loss)
        loss = (ori_loss+loss_sigGEN)/2
        loss = torch.mean(loss)
        """"""

        acc = self.cal_acc(sig_batch, lab_batch)

        return loss, acc


class common_Trainer_SegCS(common_Trainer_SigGEN):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.SegCS(sig_batch, lab_batch, n=2))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc


class common_Trainer_SegRE(common_Trainer_SigGEN):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.SegRE(sig_batch, lab_batch))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc


class common_Trainer_SigPC(common_Trainer_SigGEN):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.SigPC(
            sig_batch, lab_batch, ori_loss))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc


class common_Trainer_SigMC(common_Trainer_SigGEN):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.SigMC(sig_batch, ori_loss))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc


# class AWN_Trainer_SigGEN(AnnealingTrainer):
#     def before_train(self):
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper.lr)
#         self.criterion = nn.CrossEntropyLoss(
#             reduction='none').to(self.hyper.device)
#         self.early_stopping = EarlyStopping(
#             self.logger, patience=self.hyper.patience)

#         T_0 = 1 if 'T_0' not in self.hyper.dict else self.hyper.T_0
#         T_mult = 2 if 'T_mult' not in self.hyper.dict else self.hyper.T_mult

#         self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
#             self.optimizer, T_0=T_0, T_mult=T_mult)

#         self.lr_list = []
#         self.best_monitor = 0.0
#         self.best_epoch = 0
#         self.train_loss_list = []
#         self.train_acc_list = []
#         self.val_loss_list = []
#         self.val_acc_list = []

#     def cal_ori_loss(self, sig_batch, lab_batch):
#         logit, regu_sum = self.model(sig_batch)
#         loss = self.criterion(logit, lab_batch)
#         regu_tensor = regu_sum[0].repeat(loss.size(0))
#         loss += regu_tensor
#         return loss

#     def SegCS(self, X, y, n=4):
#         batch_size = X.size(0)
#         sig_length = X.size(2)

#         new_batch = []
#         for i_b in range(batch_size):
#             example = X[i_b, :, :]

#             # 取索引切片
#             indices = list(range(sig_length))
#             indices = chunk_list_nsub(indices, n)

#             # 随机取第n个位置，循环移位
#             id_shift = torch.randint(0, n, (1,)).item()
#             dir_shift = random.choice([-1, 1])  # dir_shift:移动方向（1：右移，-1:左移）
#             move_id = dir_shift*id_shift
#             indices_seg = list(range(n))
#             indices_seg = indices_seg[move_id:] + indices_seg[:move_id]

#             # 随机取第n段翻转, flip操作应该为值乘以-1，参考 L. Huang et al., “Data Augmentation for Deep Learning-Based Radio Modulation Classification,” IEEE Access, vol. 8, pp. 1498–1506, 2020.
#             operator = torch.ones_like(example)

#             id_flip = torch.randint(0, n, (1,)).item()
#             operator[:, indices[id_flip]] = -1
#             # 拼接索引
#             indices_new = []
#             for i in indices_seg:
#                 indices_new += indices[i]

#             example = example[:, indices_new]
#             example = example.mul(operator)
#             new_batch.append(example)

#         # 在索引位置操作
#         X_SegCS = torch.stack(new_batch).to(self.hyper.device)

#         y_SegCS = y

#         # 计算X_SegCS的loss
#         loss_SegCS = self.cal_ori_loss(X_SegCS, y_SegCS)

#         return loss_SegCS

#     def SegRE(self, X, y):
#         sig_length = X.size()[2]

#         def addnoise(X, std_dev_tensor=1):
#             length = random.choice(list(range(32, 40)))  # 长度
#             indices = random.choice(list(range(sig_length-length)))  # 起点
#             X1, X2, X3 = torch.split(X, split_size_or_sections=[
#                                      indices, length, sig_length-length-indices], dim=-1)
#             X2_mean = torch.mean(X2, dim=-1, keepdim=True)
#             X2_mean = X2_mean.repeat(1, 1, length)
#             noise = torch.normal(X2_mean, std_dev_tensor)
#             X2_new = X2_mean*noise
#             X_new = torch.cat((X1, X2_new, X3), dim=-1)
#             return X_new

#         n = random.choice([2, 3, 5])
#         X_new = X
#         for i in range(n):
#             X_new = addnoise(X_new)
#         X_SegRE = X_new
#         y_SegRE = y

#         # 计算X_SegRE的loss
#         loss_SegRE = self.cal_ori_loss(X_SegRE, y_SegRE)

#         return loss_SegRE

#     def SigPC(self, X, y, loss, alpha=0.8, beta=0.8):
#         batch_size = X.size()[0]
#         rand_indices = torch.randperm(batch_size)
#         loss_new = loss[rand_indices]
#         # the loss of aug. example is calculated by eq.8 in the original paper.
#         # Note!!!!, The filp operation is not neccesary due the eq.8 : ls
#         # X_flip = torch.flip(X, dims=[2])
#         # y_flip=y
#         # logit_flip, regu_sum_flip = self.model(X_flip)
#         # loss_flip = self.criterion(logit_flip, y_flip)
#         # loss_flip += sum(regu_sum_flip)
#         # beta_dist = torch.distributions.Beta(alpha,beta)
#         # lamb = beta_dist.sample((batch_size,))
#         # loss_SigPC=lamb*loss_flip+(1-lamb)*loss_new
#         # return loss_SigPC
#         beta_dist = torch.distributions.Beta(alpha, beta)
#         lamb = beta_dist.sample((batch_size,))
#         lamb = lamb.to(self.hyper.device)
#         loss_SigPC = lamb*loss+(1-lamb)*loss_new
#         return loss_SigPC

#     def SigMC(self, X, loss, n=4):
#         batch_size = X.size(0)
#         sig_length = X.size(2)

#         index_sample = []
#         index_length = []

#         # 取索引切片
#         indices = list(range(sig_length))
#         indices = chunk_list_nsub(indices, n)

#         len_indices = [len(i_seg) for i_seg in indices]

#         for i in range(batch_size):
#             index = random.sample(range(batch_size), n)
#             index_sample.append(index)
#             random.shuffle(len_indices)
#             index_length.append(len_indices)

#         index_sample = torch.tensor(index_sample)
#         index_sample = index_sample.to(self.hyper.device)

#         index_length = torch.tensor(index_length)
#         index_length = index_length.to(self.hyper.device)
#         index_length = index_length*1.0/sig_length

#         loss = loss[index_sample]*index_length
#         loss = loss.sum(dim=-1)

#         return loss

#     def cal_acc(self, sig_batch, lab_batch):
#         logit, _ = self.model(sig_batch)
#         pre_lab = torch.argmax(logit, 1)
#         acc = torch.sum(pre_lab == lab_batch.data).double(
#         ).item() / lab_batch.size(0)
#         return acc

#     def cal_loss(self, ori_loss, aug_loss):
#         loss = (ori_loss+aug_loss)/2
#         loss = torch.mean(loss)
#         return loss

#     def cal_loss_acc(self, sig_batch, lab_batch):
#         def sigGEN(X, y, ori_loss):
#             loss1 = self.SegCS(X, y)
#             loss2 = self.SegRE(X, y)
#             loss3 = self.SigPC(X, y, ori_loss)
#             loss4 = self.SigMC(X, ori_loss)

#             loss0 = torch.stack((loss1, loss2, loss3, loss4), dim=0)

#             batch_size = batch_size = X.size()[0]
#             mask_tensor = torch.zeros(4, batch_size)
#             for j in range(batch_size):
#                 i = random.randint(0, 3)
#                 mask_tensor[i][j] = 1

#             mask_tensor = mask_tensor.to(self.hyper.device)

#             loss0 = loss0*mask_tensor
#             loss_sigGEN = torch.sum(loss0, dim=0)
#             return loss_sigGEN

#         ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
#         loss_sigGEN = sigGEN(sig_batch, lab_batch, ori_loss)
#         loss = (ori_loss+loss_sigGEN)/2
#         loss = torch.mean(loss)
#         """"""

#         acc = self.cal_acc(sig_batch, lab_batch)

#         return loss, acc


# class AWN_Trainer_SegCS(AWN_Trainer_SigGEN):
#     def cal_loss_acc(self, sig_batch, lab_batch):
#         ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
#         loss = self.cal_loss(ori_loss, self.SegCS(sig_batch, lab_batch))
#         acc = self.cal_acc(sig_batch, lab_batch)
#         return loss, acc


# class AWN_Trainer_SegRE(AWN_Trainer_SigGEN):
#     def cal_loss_acc(self, sig_batch, lab_batch):
#         ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
#         loss = self.cal_loss(ori_loss, self.SegRE(sig_batch, lab_batch))
#         acc = self.cal_acc(sig_batch, lab_batch)
#         return loss, acc


# class AWN_Trainer_SigPC(AWN_Trainer_SigGEN):
#     def cal_loss_acc(self, sig_batch, lab_batch):
#         ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
#         loss = self.cal_loss(ori_loss, self.SigPC(
#             sig_batch, lab_batch, ori_loss))
#         acc = self.cal_acc(sig_batch, lab_batch)
#         return loss, acc


# class AWN_Trainer_SigMC(AWN_Trainer_SigGEN):
#     def cal_loss_acc(self, sig_batch, lab_batch):
#         ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
#         loss = self.cal_loss(ori_loss, self.SigMC(sig_batch, ori_loss))
#         acc = self.cal_acc(sig_batch, lab_batch)
#         return loss, acc
