import torch
import torch.nn as nn
from torch import optim

from models.nn._baseTrainer import AnnealingTrainer, EarlyStopping
import numpy as np
from scipy.spatial.distance import euclidean
import random
from fastdtw import fastdtw
from joblib import Parallel, delayed  # 并行计算
'''
 Refer to Xinjie Xu, 'Mixing Signals: Data Augmentation Approach for Deep Learning Based Modulation Recognition' DOI :10.48550/arXiv.2204.03737
'''
class common_Trainer_SigMix(AnnealingTrainer):
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

    def RM(self, X, y, proportion=0.5, N=2):

        batch_size, channels, signal_length = X.shape

        segment_length = int(proportion * signal_length)

        # label dict :  {label : indices list of this label}
        label_dict = {}
        for i in range(batch_size):
            label = y[i].item()
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(i)

        for label in label_dict:
            label_dict[label] = torch.tensor(label_dict[label], device=X.device)

        target_size = batch_size * (N - 1)
        current_size = 0

        new_data = torch.empty((target_size, channels, signal_length), device=X.device)
        new_labels = torch.empty(target_size, dtype=y.dtype, device=y.device)

        while current_size < target_size:
            for label, indices in label_dict.items():
                if len(indices) < 2:
                    continue

                # len(indices)*(N-1)//2  <-- here we adjust the size to (N-1)//2 to get the fastest speed
                batch_indices = torch.randint(0, len(indices), (min(len(indices)*(N-1)//2, (target_size - current_size) // 2), 2), device=X.device)
                idx1, idx2 = batch_indices[:, 0], batch_indices[:, 1]
                signal1, signal2 = X[indices[idx1]], X[indices[idx2]]

                # get random start idx
                start_idx1 = torch.randint(0, signal_length - segment_length, (idx1.size(0),), device=X.device)
                # start_idx2 = torch.ones_like(start_idx1).to(X.device) * (signal_length - segment_length - 1) - start_idx1

                new_signal1 = signal1.detach().clone()
                new_signal2 = signal2.detach().clone()

                batch_indices = torch.arange(idx1.size(0), device=X.device) # [bs,]
                arange_tensor = torch.arange(segment_length, device=X.device).unsqueeze(0)  # [1, segment_length]

                start_idx1_expanded = start_idx1.unsqueeze(1) + arange_tensor  # 形状 [bs, segment_length]
                start_idx2_expanded = start_idx1.unsqueeze(1) + arange_tensor  # 形状 [bs, segment_length]

                # advanced indexing to mix signals in batch
                new_signal1[batch_indices.unsqueeze(1), :, start_idx1_expanded] = \
                    signal2[batch_indices.unsqueeze(1), :, start_idx2_expanded]

                new_signal2[batch_indices.unsqueeze(1), :, start_idx2_expanded] = \
                    signal1[batch_indices.unsqueeze(1), :, start_idx1_expanded]

                new_data[current_size:current_size + 2 * idx1.size(0)] = torch.cat([new_signal1, new_signal2], dim=0)
                new_labels[current_size:current_size + 2 * idx1.size(0)] = torch.cat([y[indices[idx1]], y[indices[idx2]]], dim=0)

                current_size += 2 * idx1.size(0)

                if current_size >= target_size:
                    break

        loss_rm = self.cal_ori_loss(new_data[:current_size], new_labels[:current_size])
        return loss_rm


    def n_RM(self, X, y, proportion=0.5, N=13, n=4):
        batch_size, channels, signal_length = X.shape

        segment_length = int(proportion * signal_length)

        # label dict : {label : indices list of this label}
        label_dict = {}
        for i in range(batch_size):
            label = y[i].item()
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(i)

        for label in label_dict:
            label_dict[label] = torch.tensor(label_dict[label], device=X.device)

        target_size = batch_size * (N - 1)
        current_size = 0

        new_data = torch.empty((target_size, channels, signal_length), device=X.device)
        new_labels = torch.empty(target_size, dtype=y.dtype, device=y.device)

        while current_size < target_size:
            for label, indices in label_dict.items():
                if len(indices) < 2:
                    continue

                # Step 1: Select target signals
                target_batch_indices = torch.randint(0, len(indices), (min(len(indices)*(N-1), (target_size - current_size)),), device=X.device)
                target_signals = X[indices[target_batch_indices]].detach().clone()
                target_labels = y[indices[target_batch_indices]]

                # Step 2: Perform n swaps
                for _ in range(n):
                    # Select swap signals
                    swap_batch_indices = torch.randint(0, len(indices), (target_batch_indices.size(0),), device=X.device)
                    swap_signals = X[indices[swap_batch_indices]].detach().clone()

                    # Get random start indices for both target and swap signals
                    start_idx_target = torch.randint(0, signal_length - segment_length, (target_batch_indices.size(0),), device=X.device)
                    # start_idx_swap = torch.ones_like(start_idx_target).to(X.device) * (signal_length - segment_length - 1) - start_idx_target
                    # start_idx_swap = torch.randint(0, signal_length - segment_length, (target_batch_indices.size(0),), device=X.device)

                    # Prepare for advanced indexing
                    batch_indices = torch.arange(target_batch_indices.size(0), device=X.device)
                    arange_tensor = torch.arange(segment_length, device=X.device).unsqueeze(0)  # [1, segment_length]

                    start_idx_target_expanded = start_idx_target.unsqueeze(1) + arange_tensor  # Shape: [bs, segment_length]
                    start_idx_swap_expanded = start_idx_target.unsqueeze(1) + arange_tensor  # Shape: [bs, segment_length]

                    # Perform the swap: replace a segment of target_signals with a segment of swap_signals
                    target_signals[batch_indices.unsqueeze(1), :, start_idx_target_expanded] = \
                        swap_signals[batch_indices.unsqueeze(1), :, start_idx_swap_expanded]


                new_data[current_size:current_size + target_batch_indices.size(0)] = target_signals
                new_labels[current_size:current_size + target_batch_indices.size(0)] = target_labels

                current_size += target_batch_indices.size(0)

                if current_size >= target_size:
                    break

        loss_rm = self.cal_ori_loss(new_data[:current_size], new_labels[:current_size])
        return loss_rm


    def batch_dtw_subsequences(self,X1, X2, segment_length):
        batch_size, channels, signal_length = X1.shape

        num_windows = signal_length - segment_length + 1

        dtw_distances = torch.empty((batch_size, num_windows, num_windows), device=X1.device)

        for i in range(num_windows):
            for j in range(num_windows):
                print(f'i , j  :{i} ,{j}')
                # get subsequence
                subseq1 = X1[:, :, i:i+segment_length]  # [batch_size, channels, segment_length]
                subseq2 = X2[:, :, j:j+segment_length]  # [batch_size, channels, segment_length]

                # calculate distance
                cost = torch.cdist(subseq1.permute(0, 2, 1), subseq2.permute(0, 2, 1), p=2)  # [batch_size, segment_length, segment_length]

                dtw_matrix = torch.full((batch_size, segment_length + 1, segment_length + 1), float('inf'), device=X1.device)
                dtw_matrix[:, 0, 0] = 0

                # calculate dtw matrix using dp
                for m in range(1, segment_length + 1):
                    for n in range(1, segment_length + 1):
                        cost_mn = cost[:, m-1, n-1]
                        dtw_matrix[:, m, n] = cost_mn + torch.min(torch.stack([
                            dtw_matrix[:, m-1, n],    # 上
                            dtw_matrix[:, m, n-1],    # 左
                            dtw_matrix[:, m-1, n-1]   # 左上
                        ], dim=-1), dim=-1).values
                # the meaning of dtw_distances[i,j] is when start_idx1=i and start_idx2=j the dtw distance of these two subsequences
                dtw_distances[:, i, j] = dtw_matrix[:, -1, -1]

        return dtw_distances

    def MSM(self, X, y, proportion=0.5, N=13):
        batch_size, channels, signal_length = X.shape

        segment_length = int(proportion * signal_length)

        #  {label : indices list of this label}
        label_dict = {}
        for i in range(batch_size):
            label = y[i].item()
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(i)

        for label in label_dict:
            label_dict[label] = torch.tensor(label_dict[label], device=X.device)

        target_size = batch_size * (N - 1)
        current_size = 0

        new_data = torch.empty((target_size, channels, signal_length), device=X.device)
        new_labels = torch.empty(target_size, dtype=y.dtype, device=y.device)

        while current_size < target_size:
            for label, indices in label_dict.items():
                if len(indices) < 2:
                    continue
                print(f'label : {label}')
                batch_indices = torch.randint(0, len(indices), (min(len(indices)*(N-1)//2, (target_size - current_size) // 2), 2), device=X.device)
                idx1, idx2 = batch_indices[:, 0], batch_indices[:, 1]
                signal1, signal2 = X[indices[idx1]], X[indices[idx2]]

                # calculate the dtw distance of all start_idx choices
                dtw_matrix = self.batch_dtw_subsequences(signal1, signal2, segment_length)

                # get the start_idx1 and start_idx2 to minimize the dtw distance
                min_dtw_indices = torch.argmin(dtw_matrix.reshape(batch_indices.size(0), -1), dim=-1)
                start_idx1 = min_dtw_indices // (signal_length - segment_length + 1)
                start_idx2 = min_dtw_indices % (signal_length - segment_length + 1)

                new_signal1 = signal1.detach().clone()
                new_signal2 = signal2.detach().clone()

                batch_indices = torch.arange(idx1.size(0), device=X.device)
                arange_tensor = torch.arange(segment_length, device=X.device).unsqueeze(0)  # [1, segment_length]

                start_idx1_expanded = start_idx1.unsqueeze(1) + arange_tensor  # 形状 [bs, segment_length]
                start_idx2_expanded = start_idx2.unsqueeze(1) + arange_tensor  # 形状 [bs, segment_length]

                new_signal1[batch_indices.unsqueeze(1), :, start_idx1_expanded] = \
                    signal2[batch_indices.unsqueeze(1), :, start_idx2_expanded]

                new_signal2[batch_indices.unsqueeze(1), :, start_idx2_expanded] = \
                    signal1[batch_indices.unsqueeze(1), :, start_idx1_expanded]

                new_data[current_size:current_size + 2 * idx1.size(0)] = torch.cat([new_signal1, new_signal2], dim=0)
                new_labels[current_size:current_size + 2 * idx1.size(0)] = torch.cat([y[indices[idx1]], y[indices[idx2]]], dim=0)

                current_size += 2 * idx1.size(0)

                if current_size >= target_size:
                    break

        loss_msm = self.cal_ori_loss(new_data[:current_size], new_labels[:current_size])
        return loss_msm

    def find_best_subsequence_with_theta_batch(self, signals1, signals2, segment_length, theta=0.2, patience=5):
        signal_length = signals1.shape[2]
        batch_size = signals1.shape[0]

        min_dtw_distances = torch.full((batch_size,), float('inf'))
        best_starts1 = torch.zeros(batch_size, dtype=torch.long)
        best_starts2 = torch.zeros(batch_size, dtype=torch.long)
        found = torch.zeros(batch_size, dtype=torch.bool)

        for _ in range(patience):
            starts1 = torch.randint(0, signal_length - segment_length + 1, (batch_size,))
            starts2 = torch.randint(0, signal_length - segment_length + 1, (batch_size,))

            subseqs1 = [signals1[i, :, starts1[i]:starts1[i] + segment_length].cpu().numpy() for i in range(batch_size)]
            subseqs2 = [signals2[i, :, starts2[i]:starts2[i] + segment_length].cpu().numpy() for i in range(batch_size)]

            distances = Parallel(n_jobs=-1)(delayed(lambda x, y: fastdtw(x.T, y.T)[0])(subseqs1[i], subseqs2[i]) for i in range(batch_size))


            for i, distance in enumerate(distances):
                if found[i]:
                    continue
                if distance < theta:
                    found[i] = True
                    best_starts1[i] = starts1[i]
                    best_starts2[i] = starts2[i]
                elif distance < min_dtw_distances[i]:
                    min_dtw_distances[i] = distance
                    best_starts1[i] = starts1[i]
                    best_starts2[i] = starts2[i]

        return best_starts1, best_starts2

    def Theta_SM(self, X, y, proportion=0.5, N=13, theta=0.2, patience=5):
        batch_size, channels, signal_length = X.shape
        device = X.device

        new_data = torch.zeros((batch_size * (N - 1), channels, signal_length), device=device)
        new_labels = torch.zeros(batch_size * (N - 1), dtype=y.dtype, device=device)

        segment_length = int(proportion * signal_length)


        label_dict = {}
        for i in range(batch_size):
            label = y[i].item()
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append(i)


        target_size = batch_size * (N - 1)
        current_size = 0

        while current_size < target_size:
            for label, indices in label_dict.items():
                if len(indices) < 2:
                    continue

                num_pairs = min(len(indices)*(N-1) // 2, (target_size - current_size) // 2)
                idx_pairs = torch.randint(0, len(indices), (num_pairs, 2), device=device)
                signals1 = X[[indices[idx_pairs[i, 0]] for i in range(num_pairs)]]
                signals2 = X[[indices[idx_pairs[i, 1]] for i in range(num_pairs)]]

                best_starts1, best_starts2 = self.find_best_subsequence_with_theta_batch(signals1, signals2, segment_length, theta, patience)


                best_starts1 = best_starts1.to(device)
                best_starts2 = best_starts2.to(device)

                batch_indices = torch.arange(num_pairs, device=device)

                arange_tensor = torch.arange(segment_length, device=device).unsqueeze(0)  # 形状 [1, segment_length]

                best_starts1_expanded = best_starts1.unsqueeze(1) + arange_tensor  # 形状 [batch_size, segment_length]
                best_starts2_expanded = best_starts2.unsqueeze(1) + arange_tensor  # 形状 [batch_size, segment_length]

                new_signal1 = signals1.detach().clone()
                new_signal2 = signals2.detach().clone()

                new_signal1[batch_indices.unsqueeze(1), :, best_starts1_expanded] = \
                    signals2[batch_indices.unsqueeze(1), :, best_starts2_expanded]
                new_signal2[batch_indices.unsqueeze(1), :, best_starts2_expanded] = \
                    signals1[batch_indices.unsqueeze(1), :, best_starts1_expanded]

                new_data[current_size:current_size + 2 * num_pairs] = torch.cat([new_signal1, new_signal2], dim=0)

                new_labels[current_size:current_size + 2 * num_pairs] = torch.cat([y[torch.tensor(indices, device=device)[idx_pairs[:, 0]]],
                                                                                y[torch.tensor(indices, device=device)[idx_pairs[:, 1]]]], dim=0)


                current_size += 2 * num_pairs

                if current_size >= target_size:
                    break

        loss_theta_sm = self.cal_ori_loss(new_data, new_labels)

        return loss_theta_sm


    def RM2(self,X,y,proportion=0.5,N=13):
        '''
        只对-6db以上mix
        '''
        pass


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

class common_Trainer_SigRM(common_Trainer_SigMix):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.RM(sig_batch, lab_batch,N=2))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

class common_Trainer_SigMSM(common_Trainer_SigMix):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.MSM(sig_batch, lab_batch,N=2))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

class common_Trainer_SigTheta_SM(common_Trainer_SigMix):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.Theta_SM(sig_batch, lab_batch,N=2))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc

class common_Trainer_SignRM(common_Trainer_SigMix):
    def cal_loss_acc(self, sig_batch, lab_batch):
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.n_RM(sig_batch, lab_batch,N=2))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc


# class AWN_Trainer_SigMix(common_Trainer_SigMix):
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

# class AWN_Trainer_SigRM(AWN_Trainer_SigMix, common_Trainer_SigRM): pass

# class AWN_Trainer_SigMSM(AWN_Trainer_SigMix, common_Trainer_SigMSM): pass

# class AWN_Trainer_SigTheta_SM(AWN_Trainer_SigMix, common_Trainer_SigTheta_SM): pass

# class AWN_Trainer_SignRM(AWN_Trainer_SigMix, common_Trainer_SignRM): pass
