from scipy.fft import fft
import torch
from scipy import signal
import os
import numpy as np
from taskAttack.util import bound_pert
from taskAttack.attackmethods.gradient.fgsm import IFGSM


class FCIAA(IFGSM):
    """
    Chen, Y., Qiao, X., Zhang, J., Zhang, T., Du, Y., 2024. Frequency-constrained iterative adversarial attacks for automatic modulation classification. IEEE Commun. Lett. 28, 2734–2738. https://doi.org/10.1109/LCOMM.2024.3482552
    """

    def set_specific_params(self,):
        self.sampling_freq = 1.0  # 采样频率
        self.filter_order = 15    # 滤波器阶数
        self.epoch = 30            # 迭代次数
        self.cutoff_freq = None             # 截止频率，将在频域分析后确定
        self.filter_b = None                # 滤波器分子系数
        self.filter_a = None                # 滤波器分母系数

    def frequency_domain_analysis(self, x):
        """
        通过FFT分析信号的频谱成分，确定截止频率

        参数:
            x: 输入信号 (batch_size, 2, signal_length)

        返回:
            cutoff_freq: 确定的截止频率
        """
        # 计算信号的平均频谱
        batch_size, channels, signal_length = x.shape
        x_np = x.cpu().numpy()

        # 对每个样本和通道执行FFT
        spectra = np.zeros((batch_size, channels, signal_length))
        for i in range(batch_size):
            for c in range(channels):
                # 执行FFT
                spectrum = np.abs(fft(x_np[i, c, :]))
                spectra[i, c, :] = spectrum

        # 计算平均频谱
        avg_spectrum = np.mean(spectra, axis=(0, 1))

        # 计算累积能量
        cum_energy = np.cumsum(avg_spectrum)
        total_energy = cum_energy[-1]

        # 确定包含95%能量的频率索引（这是一个常见做法）
        energy_threshold = 0.95 * total_energy
        cutoff_idx = np.where(cum_energy >= energy_threshold)[0][0]

        # 将索引转换为频率
        cutoff_freq = cutoff_idx * self.sampling_freq / signal_length

        # 为了安全起见，限制截止频率在Nyquist频率以下
        nyquist = self.sampling_freq / 2.0
        self.cutoff_freq = min(cutoff_freq, nyquist * 0.9)
        # self.cutoff_freq = min(cutoff_freq, nyquist * 0.156) # In the original paper, the cutoff frequency is set to 0.156 of Nyquist frequency, which is a more conservative choice to ensure the perturbation is well within the signal bandwidth.

        return self.cutoff_freq

    def get_filter_coefficients(self):
        """
        根据确定的截止频率获取Butterworth低通滤波器系数

        返回:
            b, a: 滤波器系数，分别为分子和分母系数
        """
        if self.cutoff_freq is None:
            raise ValueError("Need set cutoff_freq before getting filter coefficients")

        nyquist = self.sampling_freq / 2.0
        normalized_cutoff = self.cutoff_freq / nyquist

        # 创建Butterworth低通滤波器，确保归一化截止频率在0到1之间
        b, a = signal.butter(self.filter_order, min(normalized_cutoff, 0.99), 'low')

        self.filter_b = b
        self.filter_a = a

        return b, a

    def apply_filter(self, x):
        """
        应用低通滤波器到输入信号

        参数:
            x: 输入信号 (batch_size, 2, signal_length)

        返回:
            filtered_x: 滤波后的信号
        """
        if self.filter_b is None or self.filter_a is None:
            raise ValueError("Need set filter coefficients before applying filter")

        # 将张量转换为numpy数组进行滤波
        x_np = x.detach().cpu().numpy()
        batch_size, channels, signal_length = x_np.shape

        filtered_x = np.zeros_like(x_np)
        for i in range(batch_size):
            for c in range(channels):
                # 应用滤波器
                filtered_x[i, c, :] = signal.filtfilt(self.filter_b, self.filter_a, x_np[i, c, :])

        # 转回张量
        return torch.tensor(filtered_x, dtype=x.dtype, device=x.device)

    def get_grad(self, loss, delta, **kwargs):
        """
        计算梯度并应用低通滤波器以确保扰动在期望的频率范围内

        参数:
            loss: 损失值
            delta: 扰动

        返回：
            filtered_grad: 滤波后的梯度
        """
        # 首先获取原始梯度
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        # 将梯度转换为numpy数组
        grad_np = grad.detach().cpu().numpy()
        batch_size, channels, signal_length = grad_np.shape

        # 应用滤波器到每个梯度分量
        filtered_grad_np = np.zeros_like(grad_np)
        for i in range(batch_size):
            for c in range(channels):
                # 应用滤波器
                filtered_grad_np[i, c, :] = signal.filtfilt(self.filter_b, self.filter_a, grad_np[i, c, :])

        # 将滤波后的梯度转回张量
        filtered_grad = torch.tensor(filtered_grad_np, dtype=grad.dtype, device=grad.device)

        return filtered_grad

    def forward(self, data, label, **kwargs):
        """
        FCIAA攻击的主要流程

        参数:
            data: 输入信号
            label: 目标标签

        返回:
            delta: 最终的对抗扰动
        """
        # 首先进行频域分析以确定截止频率
        self.cutoff_freq = self.frequency_domain_analysis(data)
        # self.logger.info(f"Cutoff_freq: {self.cutoff_freq:.4f} Hz")

        # 获取滤波器系数
        self.filter_b, self.filter_a = self.get_filter_coefficients()

        # 调用父类的forward方法获取扰动
        delta = super().forward(data, label, **kwargs)

        # 对最终扰动再次应用滤波器
        final_filtered_delta = self.apply_filter(delta)

        return final_filtered_delta


# uncomment the following lines to make unit test
if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser

    from data import data_zoo

    args, parser = get_parser()

    args.algo = 'fci'
    args.gid = 0
    model ='awn'
    args.surrogate_model = model
    args.target_model = model

    args.psr = -20
    args.data = 'dr2'
    data_name = data_zoo[args.data]['data_name']
    model_ckp = f'checkpoints/{data_name}/nature/{data_name}_{model}.best.pt'
    args.surrogate_ckp = model_ckp
    args.target_ckp = model_ckp

    args.cuda = True
    args.test = True
    args.clean = True
    args.batch_size = 200 # 12000 per snr

    # args.snr = [0,10]
    task = Attack(args, parser)
    task.conduct()