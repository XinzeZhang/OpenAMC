import torch
import torch.nn as nn
from torch import optim
import numpy as np
from models.nn._baseTrainer import AnnealingTrainer, EarlyStopping


import random

'''
Refer to Q. Zheng, P. Zhao, Y. Li, H. Wang, and Y. Yang, “Spectrum interference-based two-level data augmentation method in deep learning for automatic modulation classification,” Neural Comput & Applic, vol. 33, no. 13, pp. 7723–7745, Jul. 2021, doi: 10.1007/s00521-020-05514-1.

'''
class common_Trainer_SigSIB(AnnealingTrainer):
    def before_train(self):
        super().before_train()
        self.criterion = nn.CrossEntropyLoss(
            reduction='none').to(self.hyper.device)  #return loss for each sample


    def spectrum_interference(self, stft_result, F_max, T_max):
        """
        对STFT频谱进行干扰

        参数:
        stft_result: STFT变换结果，shape为(batch_size, 2, freq_bins, time_bins)
        F_max: 频率轴最大值freq_bins
        T_max: 时间轴最大值time_bins
        随机选取起始点，按照截断正态分布选择掩码的宽度和高度，最后处理一下边界条件。
        只对幅度谱做掩码，复数谱用于获取相位信息

        返回:
        干扰后的频谱
        """
        batch_size = stft_result.shape[0]
        device = stft_result.device

        # 计算参数，从论文中来
        mu_v = F_max/2
        mu_h = T_max/2
        sigma_v = F_max/4
        sigma_h = T_max/4
        a_v = a_h = 1
        b_v = F_max/2
        b_h = T_max/2

        # 为每个样本生成随机起始点
        v0 = torch.rand(batch_size, device=device) * F_max
        h0 = torch.rand(batch_size, device=device) * T_max

        # 生成截断正态分布的宽度和高度
        def truncated_normal(mu, sigma, a, b):
            # 标准正态分布的CDF
            Phi = lambda x: 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

            # 生成均匀分布的随机数
            u = torch.rand(batch_size, device=device)

            # 转换为截断正态分布
            x = mu + sigma * torch.erfinv(
                2 * ((Phi((a - mu)/sigma) + u * (Phi((b - mu)/sigma) - Phi((a - mu)/sigma))) - 0.5)
            )
            return x

        m_v = truncated_normal(mu_v, sigma_v, a_v, b_v)
        m_h = truncated_normal(mu_h, sigma_h, a_h, b_h)

        # 处理边界溢出
        mask_v = v0 + m_v > F_max
        mask_h = h0 + m_h > T_max
        m_v[mask_v] = F_max - v0[mask_v]
        m_h[mask_h] = T_max - h0[mask_h]

        # 创建干扰掩码
        mask = torch.ones_like(stft_result)
        for i in range(batch_size):
            v_start = int(v0[i].item())
            h_start = int(h0[i].item())
            v_width = int(m_v[i].item())
            h_width = int(m_h[i].item())

            mask[i, :, v_start:v_start+v_width, h_start:h_start+h_width] = 0

        # 应用掩码
        interfered_spectrum = stft_result * mask

        return interfered_spectrum

    def batch_stft_transform(self, batch_data, window_length=32, overlap=16, nfft=32):
    #原论文def batch_stft_transform(self, batch_data, window_length=16, overlap=8, nfft=16):
        """
        使用STFT对批量数据进行变换
        """
        # 创建Hamming窗
        window = 0.53 - 0.47 * torch.cos(2 * np.pi * torch.arange(window_length) / (window_length - 1))
        window = window.to(batch_data.device)

        hop_length = window_length - overlap

        # 对每个通道分别进行STFT
        batch_size = batch_data.shape[0]
        stft_result = []

        for i in range(2):  # 对两个通道分别处理
            channel_stft = torch.stft(
                batch_data[:, i, :],
                n_fft=nfft,
                hop_length=hop_length,
                win_length=window_length,
                window=window,
                return_complex=True
            )
            stft_result.append(channel_stft) #返回结果为复数 torch.complex64 或 torch.complex128

        # 将结果堆叠成所需的形状 (batch_size, 2, freq_bins, time_bins)
        stft_result = torch.stack(stft_result, dim=1)

        # 计算幅度谱
        stft_magnitude = torch.abs(stft_result)  #这个abs就是对复数求模

        return stft_magnitude, stft_result  # 返回幅度谱和复数谱
    def inverse_stft_transform(self, interfered_spectrum, original_complex_spectrum, window_length=32, overlap=16, nfft=32):
    #原论文def inverse_stft_transform(self, interfered_spectrum, original_complex_spectrum, window_length=16, overlap=8, nfft=16):
        """
        执行逆STFT变换

        参数:
        interfered_spectrum: 干扰后的幅度谱
        original_complex_spectrum: 原始复数谱，用于获取相位信息
        """
        hop_length = window_length - overlap

        # 创建Hamming窗
        window = 0.53 - 0.47 * torch.cos(2 * np.pi * torch.arange(window_length) / (window_length - 1))
        window = window.to(interfered_spectrum.device)

        # 从原始复数谱中获取相位信息
        phase = torch.angle(original_complex_spectrum)

        # 重建复数谱
        complex_spectrum = interfered_spectrum * torch.exp(1j * phase)

        # 对每个通道分别进行逆STFT
        batch_size = interfered_spectrum.shape[0]
        reconstructed_signal = torch.empty((batch_size, 2, 128), device=interfered_spectrum.device)

        for i in range(2):
            reconstructed_signal[:, i, :] = torch.istft(
                complex_spectrum[:, i],
                n_fft=nfft,
                hop_length=hop_length,
                win_length=window_length,
                window=window,
                length=128
            )

        return reconstructed_signal

    def aug(self, X, y):
        try:
            X_aug = X.detach().clone()
            y_aug = y.detach().clone().to(y.device)

            # STFT变换，获取幅度谱和复数谱
            stft_magnitude, complex_spectrum = self.batch_stft_transform(X_aug)

            # 频谱干扰
            interfered_spectrum = self.spectrum_interference(
                stft_magnitude,
                F_max=stft_magnitude.shape[2],
                T_max=stft_magnitude.shape[3]
            )

            # 逆STFT变换，使用原始相位信息，幅度谱采用遮掩后的，同时利用原频谱的相位信息。
            X_aug = self.inverse_stft_transform(interfered_spectrum, complex_spectrum)

            loss_aug = self.cal_ori_loss(X_aug, y_aug)

            return loss_aug
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return None

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
        ori_loss = self.cal_ori_loss(sig_batch, lab_batch)
        loss = self.cal_loss(ori_loss, self.aug(sig_batch, lab_batch))
        acc = self.cal_acc(sig_batch, lab_batch)
        return loss, acc
