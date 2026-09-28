import torch
import torch.nn as nn
from torch import optim

from models.nn._baseTrainer import AnnealingTrainer, EarlyStopping


import random

'''
 Refer to Liang Huang and Weijian Pan, 'Data Augmentation for Deep Learning-Based Radio Modulation Classification', IEEE Access, DOI:10.1109/ACCESS.2019.2960775
'''
class common_Trainer_SigRFG(AnnealingTrainer):
    def before_train(self):
        super().before_train()
        self.criterion = nn.CrossEntropyLoss(
            reduction='none').to(self.hyper.device)  #return loss for each sample

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

    def Rotation(self, X, y):
        rotation_matrices = torch.stack([
            torch.tensor([[0, -1], [1, 0]]),
            torch.tensor([[-1, 0], [0, -1]]),
            torch.tensor([[0, 1], [-1, 0]])
        ]).float().to(X.device)

        batch_size = X.size(0)
        random_indices = torch.randint(0, 3, (batch_size,)).to(X.device)

        selected_matrices = rotation_matrices[random_indices]

        _x = torch.permute(X, (0, 2, 1))

        # (batch_size, height, 2) x (batch_size, 2, 2)  -->   (batch, height, 2)
        X_rotation = torch.bmm(_x, selected_matrices)

        X_rotation = torch.permute(X_rotation, (0, 2, 1)) #new signals (batch, 2, height)

        y_rotation = y.detach().clone().to(y.device)

        loss_rotation = self.cal_ori_loss(X_rotation, y_rotation)

        return loss_rotation

    def Flip(self, X, y):
        batch_size = X.size(0)

        random_indices = torch.randint(0, 3, (batch_size,)).to(X.device)

        X_flip = X.detach().clone()

        h_flip_mask = random_indices == 0
        X_flip[h_flip_mask, 0, :] *= -1

        v_flip_mask = random_indices == 1
        X_flip[v_flip_mask, 1, :] *= -1

        both_flip_mask = random_indices == 2
        X_flip[both_flip_mask] *= -1

        y_flip=y.detach().clone().to(y.device)

        loss_flip = self.cal_ori_loss(X_flip, y_flip)

        return loss_flip

    def Gaussian_noise(self, X, y):
        std_list = [0.0005, 0.001, 0.002]
        batch_size = X.size(0)

        random_indices = torch.randint(0, len(std_list), (batch_size,)).to(X.device)
        selected_stds = torch.tensor(std_list).to(X.device)[random_indices]

        noise = torch.normal(0, selected_stds[:, None, None])
        noise = noise.to(X.device)

        X_noise = X + noise

        loss_noise = self.cal_ori_loss(X_noise, y)

        return loss_noise

    def Rotation_Flip(self, X, y):
        rotation_matrices = torch.stack([
            torch.tensor([[0, -1], [1, 0]]),
            torch.tensor([[-1, 0], [0, -1]]),
            torch.tensor([[0, 1], [-1, 0]])
        ]).float().to(X.device)

        batch_size = X.size(0)

        random_indices = torch.randint(0, 5, (batch_size,)).to(X.device)

        X_augmented = X.detach().clone()

        _x = torch.permute(X, (0, 2, 1))
        for i in range(3):
            rotation_mask = (random_indices == i)
            if rotation_mask.any():
                r_x = torch.matmul(_x[rotation_mask], rotation_matrices[i])
                X_augmented[rotation_mask] = torch.permute(r_x, (0, 2, 1))

        h_flip_mask = (random_indices == 3)
        X_augmented[h_flip_mask, 0, :] *= -1

        v_flip_mask = (random_indices == 4)
        X_augmented[v_flip_mask, 1, :] *= -1

        y_augmented=y.detach().clone().to(y.device)

        loss_rf = self.cal_ori_loss(X_augmented, y_augmented)

        return loss_rf

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

class common_Trainer_SigR(common_Trainer_SigRFG):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.Rotation(sig_batch, lab_batch))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

class common_Trainer_SigF(common_Trainer_SigRFG):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.Flip(sig_batch, lab_batch))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

class common_Trainer_SigG(common_Trainer_SigRFG):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.Gaussian_noise(sig_batch, lab_batch))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

class common_Trainer_SigRF(common_Trainer_SigRFG):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.Rotation_Flip(sig_batch, lab_batch))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

# class AWN_Trainer_SigRFG(common_Trainer_SigRFG):
#     def cal_ori_loss(self, sig_batch, lab_batch):
#         logit, regu_sum = self.model(sig_batch)
#         loss = self.criterion(logit, lab_batch)
#         regu_tensor = regu_sum[0].repeat(loss.size(0))
#         loss += regu_tensor
#         return loss

#     def cal_acc(self, sig_batch, lab_batch):
#         logit, _ = self.model(sig_batch)
#         pre_lab = torch.argmax(logit, 1)
#         acc = torch.sum(pre_lab == lab_batch.data).double(
#         ).item() / lab_batch.size(0)
#         return acc

# class AWN_Trainer_SigR(AWN_Trainer_SigRFG, common_Trainer_SigR): pass

# class AWN_Trainer_SigF(AWN_Trainer_SigRFG, common_Trainer_SigF): pass

# class AWN_Trainer_SigG(AWN_Trainer_SigRFG, common_Trainer_SigG): pass

# class AWN_Trainer_SigRF(AWN_Trainer_SigRFG, common_Trainer_SigRF): pass
