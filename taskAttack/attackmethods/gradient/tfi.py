import os

from matplotlib import pyplot as plt
import torch
from taskAttack.attackmethods._baseAttackAlgo import BaseAttackAlgo
from taskAttack.util import bound_pert, os_makedirs
import torch.nn.functional as F
# from taskAttack.attackmethods.gradient.fgsm import IFGSM

def _downsample_1d(x, scale: int):
    """
    x: (B, C, T)
    scale=1 -> unchanged
    """
    if scale == 1:
        return x
    return F.avg_pool1d(x, kernel_size=scale, stride=1, ceil_mode=False)


def _upsample_1d(x, target_len: int):
    """
    x: (B, C, T_small)
    upsample back to target_len
    """
    return F.interpolate(x, size=target_len, mode='linear', align_corners=False)


def _normalize_amplitude(amp, eps=1e-8):
    """
    amp: (..., Freq)
    normalize along frequency axis
    """
    return amp / (amp.sum(dim=-1, keepdim=True) + eps)


def _sanitize_tensor(tensor):
    finite_mask = torch.isfinite(tensor)
    return torch.where(finite_mask, tensor, torch.zeros_like(tensor))


def multiscale_frequency_consistency_gradient(
    grad,
    scales=(1, 2, 4),
    alpha=1.0,
    eps=1e-8,
    domain = 'freq',
    reduce = 'mean',
    sig_len = 128,
    debug = False
):
    """
    grad: (B, C, T) 输入梯度
    scales: 多尺度集合
    alpha: 权重锐化系数，>1 更强调共同高响应频带
    return: stabilized_grad, freq_weight

    思路：
    1) 对不同时间尺度下的梯度做 RFFT
    2) 提取归一化幅度谱
    3) 统一到原始频率长度
    4) 求平均作为频率一致性权重
    5) 用这个权重调制原始尺度频谱，再 IRFFT 回时域
    """
    B, C, T = grad.shape

    # 原始尺度频谱
    if domain == 'time':
        grad_fft = torch.fft.rfft(grad, dim=-1)     # (B, C, F)
    else:
        grad_fft = grad
    Freq = grad_fft.shape[-1]

    amp_list = []

    gs_list = []

    time_grad = torch.fft.irfft(grad, n=sig_len, dim=-1)  # (B, C, T)
    for s in scales:
        if domain == 'time':
            g_s = _downsample_1d(grad, s)           # (B, C, T/s)
        else:
            g_s = _downsample_1d(time_grad, s)     
        
        gs_list.append(g_s)
        
        fft_s = torch.fft.rfft(g_s, dim=-1)     # (B, C, F_s)
        amp_s = torch.abs(fft_s)                # 幅度谱
        amp_s = _normalize_amplitude(amp_s, eps=eps)

        # 插值到统一频率长度
        amp_s = F.interpolate(
            amp_s, size=Freq, mode='linear', align_corners=False
        )
        
        
        amp_list.append(amp_s)

    # 多尺度一致性权重：平均后可做锐化
    if reduce == 'mean':
        freq_weight = torch.stack(amp_list, dim=0).mean(dim=0)   # (B, C, F)
        freq_weight = freq_weight / (freq_weight.max(dim=-1, keepdim=True)[0] + eps)
    elif reduce == 'norm':
        amps = torch.stack(amp_list, dim=0)         # (S, B, C, F)
        mu = amps.mean(dim=0)
        var = amps.var(dim=0, unbiased=False)

        freq_weight = mu / torch.sqrt(var + eps)
        freq_weight = freq_weight / (freq_weight.max(dim=-1, keepdim=True)[0] + eps)
    else:
        raise ValueError(f'Unsupported reduce method: {reduce}')

    if alpha != 1.0:
        freq_weight = freq_weight.pow(alpha)

    # 用一致性权重调制原始尺度频谱
    stabilized_fft = grad_fft * freq_weight

    # 回到时域
    if domain == 'time':
        stabilized_grad = torch.fft.irfft(stabilized_fft, n=sig_len, dim=-1)
    else:
        stabilized_grad = stabilized_fft

    if debug:
        info = dict(scales = scales,  amp_list = amp_list, input_grad = grad,  stabilized_grad = stabilized_grad, input_time_grad = time_grad, gs_list=gs_list, freq_weight= freq_weight)
        return stabilized_grad, info
    else:
        return stabilized_grad, None


def freq_to_time(freq_value, sig_len=128):
    time_value = torch.fft.irfft(freq_value, n=sig_len, dim=-1)
    return time_value

def time_to_freq(time_delta, sig_len=128):
    freq_complex = torch.fft.rfft(time_delta, n=sig_len, dim=-1)
    return freq_complex



class TFI(BaseAttackAlgo):
    def set_specific_params(self,):
        self.decay = 0.5
        self.epoch = 10
        self.alpha = self.epsilon / self.epoch
        self.continue_mask = False  # whether to use a continuous frequency band or discrete random bins for perturbation 
        self.focus_bandwidth_ratio = 1
        self.reduce = 'mean'  # 'mean' or 'norm'
        
        self.using_shrinkage = True
        self.shrinkage = 0.7
        
        self.scale_num = 3
        self.scale_interval = 2
         

    def mean_shrinkage(self, time_delta):
        mean = time_delta.mean(dim=(1, 2), keepdim=True)
        std = time_delta.std(dim=(1, 2), keepdim=True)
        normalized_delta = (time_delta - mean) / (std + 1e-8)  

        scaled_delta = normalized_delta * self.shrinkage * (std + 1e-8) 
        result = scaled_delta + mean
        
        # is_equal = torch.equal(result, time_delta)              # exact bitwise
        # is_close = torch.allclose(result, time_delta, atol=1e-6, rtol=1e-5)  # numeric tolerance
        # max_diff = (result - time_delta).abs().max()
        # print(is_equal, is_close, max_diff.item())

        return result
    
    
    def transform(self, x, momentum, **kwargs):
        """
        look ahead for NI-FGSM
        """
        epoch = kwargs.get('epoch', 0)
        if epoch != 0:
            time_momentum = freq_to_time(momentum, sig_len=self.model.hyper.sig_len)
            return x + self.alpha*self.decay*time_momentum
        else:
            return x
        
    
    def _focus_band_mask(self, signal_batch, focus_bandwidth_ratio):
        _, _, signal_length = signal_batch.shape
        
        mask = torch.zeros(
            (1, 1, signal_length),
            dtype=signal_batch.dtype,
            device=signal_batch.device,
        )
        
        if self.continue_mask:
            band_length = int(round(focus_bandwidth_ratio * signal_length))
            band_length = min(max(band_length, 1), signal_length)
            max_start = signal_length - band_length
            band_start = 0 if max_start == 0 else torch.randint(0, max_start + 1, (1,), device=signal_batch.device).item()
            band_end = band_start + band_length
            mask[:, :, band_start:band_end] = 1.0
        else:
            # random select discrete frequency bins to perturb
            num_bins_to_perturb = int(round(focus_bandwidth_ratio * signal_length))
            num_bins_to_perturb = min(max(num_bins_to_perturb, 1), signal_length)
            selected_bins = torch.randperm(signal_length, device=signal_batch.device)[:num_bins_to_perturb]
            mask[:, :, selected_bins] = 1.0

        return mask


    def init_freq_pert_time_fft(self, signal_batch, focus_bandwidth_ratio=1.0):
        time_delta = torch.empty_like(signal_batch).uniform_(-1.0, 1.0)
        time_delta = bound_pert(
            time_delta,
            self.alpha,
            self.norm,
        )

        freq_complex = torch.fft.rfft(time_delta, dim=-1) # (B, C, F)
        freq_complex = freq_complex * self._focus_band_mask(
            freq_complex, focus_bandwidth_ratio
        )
      
        return freq_complex.detach().requires_grad_(True)


    def forward(self, data, label, **kwargs):
        debug_folder = kwargs.get('debug_folder')
        if debug_folder is not None:
            self.debug = True
        else:
            self.debug = False
            
        
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation in frequency domain
        self.scales =  list(range(1, self.scale_interval * (self.scale_num+1), self.scale_interval)) 
        
        delta = self.init_freq_pert_time_fft(data, self.focus_bandwidth_ratio)
        
        

            
        self.save_debug_delta_freq(data, delta, debug_folder, file_stem='init')
        
        momentum = 0
        for _ in range(self.epoch):
            if self.debug:
                i_debug_folder = os.path.join(debug_folder, f'epoch_{_}')
                    
            # Separate and set requires_grad
            delta = delta.detach().requires_grad_(True)
            
            # Convert from frequency domain to time domain and bound the perturbation
            time_delta = freq_to_time(delta, sig_len=self.model.hyper.sig_len)
            
            if self.debug and _ > 0:
                print(f'Before renorm: max={time_delta.abs().max().item():.4e}, mean={time_delta.abs().mean().item():.4e}, std={time_delta.abs().std().item():.4e}')
            
            time_delta = bound_pert(time_delta, self.epsilon, self.norm)
            
            if self.debug and _ > 0:
                print(f'After renorm: max={time_delta.abs().max().item():.4e}, mean={time_delta.abs().mean().item():.4e}, std={time_delta.abs().std().item():.4e}')
                f_debug_folder = os.path.join(debug_folder, f'epoch_{_ - 1}')
                self.save_debug_delta_time(time_delta, f_debug_folder, file_stem='final_renorm')

            # Get logits
            logits = self.get_logits(self.transform(data + time_delta, momentum=momentum, epoch = _))

            # Calculate loss
            loss = self.get_loss(logits, label)

            # Calculate gradients
            grad = self.get_grad(loss, delta)
            if len(self.scales) > 1:
                freq_grad, debug_info = multiscale_frequency_consistency_gradient(
                grad, scales=self.scales, alpha=1.0, eps=1e-8, domain='freq', reduce=self.reduce, sig_len=self.model.hyper.sig_len, debug=self.debug)
                if self.debug:
                    self.save_debug_scale_amps(debug_info, i_debug_folder)
                
            else:
                freq_grad = grad
            # Update momentum
            momentum = self.get_momentum(freq_grad, momentum)
        
            # Update adversarial perturbation in frequency domain
            delta = self._update_delta(delta, momentum, self.alpha, debug_folder=i_debug_folder if self.debug else None)



        


        # Finally convert to time domain perturbation
        delta = freq_to_time(delta, sig_len=self.model.hyper.sig_len).detach()
        # self.save_debug_delta_time(delta, debug_folder)

        return delta


    def _update_delta(self, delta, grad, alpha, **kwargs):
        delta = _sanitize_tensor(delta)
        grad = _sanitize_tensor(grad)
        
        if self.debug:
            debug_folder= kwargs.get('debug_folder')
        
        if self.norm == 'linfty':
            delta = torch.clamp(
                delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=-1)
            grad_norm = grad_norm.view(-1, 1, 1)  # to debug
            scaled_grad = grad / (grad_norm + 1e-12)  # to debug
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -
                                                       1).view_as(delta)
        
        time_delta = freq_to_time(delta, sig_len=self.model.hyper.sig_len)
        
        if self.debug:
            self.save_debug_delta_time(time_delta, debug_folder, file_stem='before_shrinkage')
        
        if self.using_shrinkage:
            # cal. L2 norm of time delta
            time_delta = self.mean_shrinkage(time_delta)
        
        if self.debug:
            self.save_debug_delta_time(time_delta, debug_folder, file_stem='after_shrinkage')
        
        
        new_time_delta = time_delta.renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(time_delta)
        
        
        delta = time_to_freq(new_time_delta, sig_len=self.model.hyper.sig_len)        

        return delta.detach().requires_grad_(True)


    def save_debug_scale_amps(self, debug_info, debug_folder, file_stem='scale'):
        if not self.debug:
            return
        os_makedirs(debug_folder)
        
        torch.save(debug_info, os.path.join(debug_folder, f'{file_stem}.pt'))
        
        scales = debug_info['scales']
        amp_list = debug_info['amp_list']
        gs_list = debug_info['gs_list']
        # input_grad = debug_info['input_grad']
        input_time_grad = debug_info['input_time_grad']
        stabilized_grad = debug_info['stabilized_grad']
        input_grad= debug_info['input_grad']
        # freq_weight = debug_info['freq_weight']
        
        number_of_scales = len(scales)
        fig, axes = plt.subplots(number_of_scales + 2, 1, figsize=(10, 4 * number_of_scales))

        axes[0].plot(input_grad[0, 0].cpu(), color='tab:red', linewidth=0.8, label='I channel')
        axes[0].plot(input_grad[0, 1].cpu(), color='tab:green', linewidth=0.8, label='Q channel')
        axes[0].set_title('Amplitude spectrum of input gradient at original scale')
        
        axes[0].set_xlabel('Frequency bin')
        axes[0].set_ylabel('Normalized Amplitude')
        axes[0].legend(loc='upper right')
                
        for i, (scale, amp) in enumerate(zip(scales, amp_list)):
            axes[i+1].plot(amp[0, 0].cpu(), color='tab:red', linewidth=0.8, label='I channel')
            axes[i+1].plot(amp[0, 1].cpu(), color='tab:green', linewidth=0.8, label='Q channel')
            axes[i+1].set_title(f'Amplitude spectrum at scale {scale}')
            axes[i+1].set_xlabel('Frequency bin')
            axes[i+1].set_ylabel('Normalized Amplitude')
            axes[i+1].legend(loc='upper right')
            
        last_ax = axes[-1]
        
        last_ax.plot(stabilized_grad[0, 0].cpu(), color='tab:red', linewidth=0.8, label='I channel')
        last_ax.plot(stabilized_grad[0, 1].cpu(), color='tab:green', linewidth=0.8, label='Q channel')
        last_ax.set_title('Amplitude spectrum of stabilized gradient')
        last_ax.set_xlabel('Frequency bin')
        last_ax.set_ylabel('Normalized Amplitude')
        last_ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.freq.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        fig, axes = plt.subplots(number_of_scales + 1, 1, figsize=(10, 4 * number_of_scales))
        
        axes[0].plot(input_time_grad[0, 0].cpu(), color='tab:red', linewidth=0.8, label='I channel')
        axes[0].plot(input_time_grad[0, 1].cpu(), color='tab:green', linewidth=0.8, label='Q channel')
        axes[0].set_title('Amplitude waveform of input gradient at original scale')
        
        axes[0].set_xlabel('Time step')
        axes[0].set_ylabel('Normalized Amplitude')
        axes[0].legend(loc='upper right')
        
        for i, (scale, gs) in enumerate(zip(scales, gs_list)):
            axes[i+1].plot(gs[0, 0].cpu(), color='tab:red', linewidth=0.8, label=f'Scale {scale} I channel')
            axes[i+1].plot(gs[0, 1].cpu(), color='tab:green', linewidth=0.8, label=f'Scale {scale} Q channel')
            axes[i+1].set_title(f'Amplitude waveform at scale {scale}')
            axes[i+1].set_xlabel('Time step')
            axes[i+1].set_ylabel('Normalized Amplitude')
            axes[i+1].legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.wave.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # for amp, scale in zip(amp_list, scales):
        #     torch.


    def save_debug_delta_freq(self, data, delta, debug_folder, file_stem='delta_freq'):
        if not self.debug:
            return

        os_makedirs(debug_folder)

        data = data.detach().cpu()
        torch.save(data, os.path.join(debug_folder, f'{file_stem}.wave.pt'))
        
        fig, axe = plt.subplots(1, 1, figsize=(12, 6))
        axe.plot(data[0, 0], color='tab:gray', linewidth=0.8, label='Input I channel')
        axe.plot(data[0, 1], color='tab:olive', linewidth=0.8, label='Input Q channel')
        axe.set_title('Input signal in time domain')
        axe.set_xlabel('Time index')
        axe.set_ylabel('Amplitude')
        axe.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.wave.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
        delta_cpu = delta.detach().cpu()
        torch.save(delta_cpu, os.path.join(debug_folder, f'{file_stem}.freq.pt'))

        channels = delta_cpu.shape[1] // 2
        freq_view = delta_cpu[0].view(2, channels, delta_cpu.shape[-1])
        freq_real = freq_view[0]
        freq_imag = freq_view[1]

        fig, axes = plt.subplots(channels, 2, figsize=(12, 3 * channels), sharex=True)
        if channels == 1:
            axes = [axes]

        channel_names = ['I', 'Q']
        for channel_idx in range(channels):
            channel_name = channel_names[channel_idx] if channel_idx < len(channel_names) else f'C{channel_idx}'
            axes[channel_idx][0].plot(freq_real[channel_idx], color='tab:red', linewidth=0.8)
            axes[channel_idx][0].set_ylabel('Amplitude')
            axes[channel_idx][0].set_title(f'{channel_name} real spectrum')

            axes[channel_idx][1].plot(freq_imag[channel_idx], color='tab:blue', linewidth=0.8)
            axes[channel_idx][1].set_ylabel('Amplitude')
            axes[channel_idx][1].set_title(f'{channel_name} imag spectrum')

        axes[-1][0].set_xlabel('Frequency bin')
        axes[-1][1].set_xlabel('Frequency bin')

        fig.suptitle('Adversarial perturbation in frequency domain')
        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.freq.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_debug_delta_time(self, delta, debug_folder, file_stem='delta_time'):
        if not self.debug:
            return

        os_makedirs(debug_folder)

        delta_cpu = delta.detach().cpu()
        torch.save(delta_cpu, os.path.join(debug_folder, f'{file_stem}.pt'))

        fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        axes.plot(delta_cpu[0, 0], color='tab:red', linewidth=0.8, label='I channel')
        axes.plot(delta_cpu[0, 1], color='tab:green', linewidth=0.8, label='Q channel')
        axes.set_ylabel('Amplitude')
        axes.set_title('Waveform of perturbation in time domain {}'.format(file_stem))
        axes.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

