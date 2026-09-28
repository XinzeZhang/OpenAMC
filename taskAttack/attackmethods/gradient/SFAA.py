import os

from matplotlib import pyplot as plt
import torch
from taskAttack.util import bound_pert, os_makedirs

from taskAttack.attackmethods.gradient.fgsm import IFGSM


class SFAA(IFGSM):
    '''
    Zhang, S., Fu, J., Yu, J., Xu, H., Zha, H., Mao, S., Lin, Y., 2024. Channel-robust class-universal spectrum-focused frequency adversarial attacks on modulated classification models. IEEE Trans. on Cogn. Commun. Netw. 10, 1281–1293. https://doi.org/10.1109/TCCN.2024.3382126
    '''
    def set_specific_params(self, **kwargs):
        """freq_init should be one of 'band_uniform' or 'time_fft'. 'band_uniform' initializes the frequency perturbation with uniform random values in the low and high frequency bands, while 'time_fft' initializes it by applying FFT to a random time-domain perturbation. The focus_bandwidth_ratio determines the proportion of the frequency spectrum that is considered as the focus band for perturbation.
        """

        if not hasattr(self, 'freq_init'):
            self.freq_init = 'band_uniform'

    def _get_focus_band_edges(self, signal_length, focus_bandwidth_ratio, N_s = 16):

        if N_s is not None:
            band_points = N_s
        else:
            band_points = int(focus_bandwidth_ratio * signal_length)
            band_points = min(max(band_points, 0), signal_length)

        low_band_end = min(signal_length, band_points)
        high_band_start = max(0, signal_length - band_points)
        return band_points, low_band_end, high_band_start

    def _focus_band_mask(self, signal_batch, focus_bandwidth_ratio):
        _, _, signal_length = signal_batch.shape
        _, low_band_end, high_band_start = self._get_focus_band_edges(
            signal_length, focus_bandwidth_ratio
        )

        mask = torch.zeros(
            (1, 1, signal_length),
            dtype=signal_batch.dtype,
            device=signal_batch.device,
        )
        if low_band_end > 1:
            mask[:, :, :low_band_end] = 1.0
        if high_band_start < signal_length:
            mask[:, :, high_band_start:] = 1.0
        return mask

    def _stack_freq(self, freq_complex):
        batch_size, channels, signal_length = freq_complex.shape
        return torch.stack([freq_complex.real, freq_complex.imag], dim=1).view(
            batch_size, channels * 2, signal_length
        )

    def init_freq_pert_uniform(self, signal_batch, focus_bandwidth_ratio=0.125):
        batch_size, channels, signal_length = signal_batch.shape
        device = signal_batch.device

        freq_real = torch.zeros((batch_size, channels, signal_length), dtype=torch.float, device=device)
        freq_imag = torch.zeros((batch_size, channels, signal_length), dtype=torch.float, device=device)

        band_points, low_band_end, high_band_start = self._get_focus_band_edges(
            signal_length, focus_bandwidth_ratio
        )

        if band_points > 0:
            low_freq_real = (torch.randn((batch_size, channels, band_points), device=device)  * self.alpha )
            low_freq_imag = (torch.randn((batch_size, channels, band_points), device=device)  * self.alpha  )
            high_freq_real = (torch.randn((batch_size, channels, band_points), device=device)  * self.alpha )
            high_freq_imag = (torch.randn((batch_size, channels, band_points), device=device)  * self.alpha )

            freq_real[:, :, :low_band_end] = low_freq_real
            freq_imag[:, :, :low_band_end] = low_freq_imag
            freq_real[:, :, high_band_start:] = high_freq_real
            freq_imag[:, :, high_band_start:] = high_freq_imag


        # Stack real and imaginary parts
        combined_freq = torch.stack([freq_real, freq_imag], dim=1).view(batch_size, channels*2, signal_length)

        return combined_freq.clone().detach().requires_grad_(True)

    def init_freq_pert_time_fft(self, signal_batch, focus_bandwidth_ratio=0.125):
        time_delta = torch.empty_like(signal_batch).uniform_(-1.0, 1.0)
        time_delta = bound_pert(
            time_delta,
            self.alpha,
            self.norm,
        )

        freq_complex = torch.fft.fft(time_delta, dim=2)
        freq_complex = freq_complex * self._focus_band_mask(
            signal_batch, focus_bandwidth_ratio
        )
        freq_complex[:, :, 0] = 0.0

        combined_freq = self._stack_freq(freq_complex)
        return combined_freq.detach().requires_grad_(True)

    def init_freq_pert(self, signal_batch, focus_bandwidth_ratio=0.125, method=None):
        if method is None:
            method = self.freq_init

        if method == 'band_uniform':
            return self.init_freq_pert_uniform(signal_batch, focus_bandwidth_ratio)
        if method == 'time_fft':
            return self.init_freq_pert_time_fft(signal_batch, focus_bandwidth_ratio)

        raise ValueError(
            'Unsupported freq_init {}. Expected one of {}'.format(
                method, ['band_uniform', 'time_fft']
            )
        )

    def freq_to_time(self, combined_freq):
        batch_size, channels, signal_length = combined_freq.shape
        channels = channels // 2  # 实际通道数是一半

        # 直接重塑张量而不创建中间变量
        freq_real = combined_freq.view(batch_size, 2, channels, signal_length)[:, 0]
        freq_imag = combined_freq.view(batch_size, 2, channels, signal_length)[:, 1]

        # 创建复数张量，避免复制
        freq_complex = torch.complex(freq_real, freq_imag)

        # 执行IFFT，直接返回实部
        return torch.fft.ifft(freq_complex, dim=2).real

    def save_debug_delta_freq(self, delta, debug_folder, file_stem='delta_freq'):
        if debug_folder is None:
            return

        os_makedirs(debug_folder)

        delta_cpu = delta.detach().cpu()
        torch.save(delta_cpu, os.path.join(debug_folder, f'{file_stem}.pt'))

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
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_debug_delta_time(self, delta, debug_folder, file_stem='delta_time'):
        if debug_folder is None:
            return

        os_makedirs(debug_folder)

        delta_cpu = delta.detach().cpu()
        torch.save(delta_cpu, os.path.join(debug_folder, f'{file_stem}.pt'))

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].plot(delta_cpu[0, 0], color='tab:red', linewidth=0.8, label='I channel')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Final perturbation in time domain')
        axes[0].legend(loc='upper right')

        axes[1].plot(delta_cpu[0, 1], color='tab:green', linewidth=0.8, label='Q channel')
        axes[1].set_xlabel('Time index')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{file_stem}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)



    def sf_forward(self, data, label, **kwargs):
        debug_folder = kwargs.get('debug_folder')
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation in frequency domain
        delta = self.init_freq_pert(data)
        # self.save_debug_delta_freq(delta, debug_folder, file_stem='delta_freq_init')
        momentum = 0

        for _ in range(self.epoch):
            # Separate and set requires_grad
            delta = delta.detach().requires_grad_(True)

            # Convert from frequency domain to time domain and bound the perturbation
            time_delta = self.freq_to_time(delta)
            time_delta = bound_pert(time_delta, self.alpha, self.norm)

            # Get logits
            logits = self.get_logits(self.transform(data + time_delta, momentum=momentum))

            # Calculate loss
            loss = self.get_loss(logits, label)

            # Calculate gradients
            grad = self.get_grad(loss, delta)

            # Update momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation in frequency domain
            delta = self._update_delta(delta, momentum, self.alpha)

        # self.save_debug_delta_freq(delta, debug_folder)


        # Finally convert to time domain perturbation
        delta = self.freq_to_time(delta).detach()
        # self.save_debug_delta_time(delta, debug_folder)

        return delta


    def _update_delta(self, delta, grad, alpha, **kwargs):
        delta = delta.nan_to_num()
        grad = grad.nan_to_num()

        if self.norm == 'linfty':
            delta = torch.clamp(
                delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=-1)
            grad_norm = grad_norm.view(-1, 1, 1)  # to debug
            scaled_grad = grad / (grad_norm + 1e-12)  # to debug
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -
                                                       1).view_as(delta)

        time_delta = self.freq_to_time(delta)
        time_delta = time_delta.renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(time_delta)
        delta = torch.fft.fft(time_delta, dim=2)
        delta = self._stack_freq(delta)

            # delta = delta.renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        # delta = clamp(delta, self.data_min-data, self.data_max-data) # This line only have meanling in image domain, which is equal to delta + data = clamp(delta + data , self.data_min, self.data_max). However, it should not be operated in raido domain.
        return delta.detach().requires_grad_(True)


    def forward(self, data, label, **kwargs):
        """
        更贴近原论文
        """
        delta = self.sf_forward(data, label, **kwargs)
        return delta


# class SF_VT(SFAA):
#     def smooth_grad(self, grad, kernel_size=5):

#         smoothed = torch.nn.functional.avg_pool1d(
#             grad,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=kernel_size//2
#         )
#         return smoothed

#     def get_variance(self, data, delta, label, cur_grad, momentum):
#         self.num_neighbor = 10  # Number of neighboring perturbations to consider
#         self.window_size_list = list(range(3, 65, 2))
#         # Calculate the variance of the gradients across neighboring perturbations
#         neighbor_deltas = []
#         for _ in range(self.num_neighbor):
#             noise = torch.randn_like(delta) * self.alpha * 0.1 # Add small random noise
#             neighbor_deltas.append(delta + noise)

#         grad = 0

#         for neighbor_delta in neighbor_deltas:
#             time_delta = self.freq_to_time(neighbor_delta)
#             time_delta = bound_pert(time_delta, self.epsilon, self.norm)
#             # window_size = random.choice(self.window_size_list)

#             logits = self.get_logits(self.transform(data + time_delta, momentum=momentum))
#             loss = self.get_loss(logits, label)
#             neighbor_grad = self.get_grad(loss, neighbor_delta)

#             grad += neighbor_grad


#         grad /= self.num_neighbor
#         variance = grad - cur_grad
#         return variance

#     def forward(self, data, label, **kwargs):
#         debug_folder = kwargs.get('debug_folder')
#         data = data.clone().detach().to(self.device)
#         label = label.clone().detach().to(self.device)

#         # Initialize adversarial perturbation in frequency domain
#         delta = self.init_freq_pert(data)
#         momentum = 0
#         variance = 0
#         for _ in range(self.epoch):
#             # Separate and set requires_grad
#             delta = delta.detach().requires_grad_(True)

#             # Convert from frequency domain to time domain and bound the perturbation
#             time_delta = self.freq_to_time(delta)
#             time_delta = bound_pert(time_delta, self.epsilon, self.norm)

#             # Get logits
#             logits = self.get_logits(self.transform(data + time_delta, momentum=momentum))

#             # Calculate loss
#             loss = self.get_loss(logits, label)

#             # Calculate gradients
#             grad = self.get_grad(loss, delta)

#             # Update momentum
#             momentum = self.get_momentum(grad + variance, momentum)
#             variance = self.get_variance(data, delta, label, grad, momentum)

#             # Update adversarial perturbation in frequency domain
#             delta = self.update_delta(delta, momentum, self.alpha)

#         # Finally convert to time domain perturbation
#         delta = self.freq_to_time(delta).detach()
#         self.save_debug_delta(delta, debug_folder, file_stem='delta_sf_vt')
#         return delta

# uncomment the following lines to make unit test
# from models.nn.AWN import AWN_config as awn
if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser

    from data import data_zoo

    args, parser = get_parser()

    args.algo = 'sfaa'
    args.gid = 0

    args.surrogate_model = 'awn'
    args.target_model = 'mcd'

    args.psr = -10
    args.data = 'dr2'
    data_name = data_zoo[args.data]['data_name']
    defense = 'nature'

    ckp_folder = f'checkpoints/{data_name}/{defense}' if defense == 'nature' else f'checkpoints/{data_name}/psr-20at/{defense}'

    args.surrogate_ckp = f'{ckp_folder}/{data_name}_{args.surrogate_model}.best.pt'
    args.target_ckp = f'{ckp_folder}/{data_name}_{args.target_model}.best.pt'

    args.cuda = True
    args.test = True
    args.clean = True
    args.batch_size = 2000 # 12000 per snr

    args.snr = [0,25]
    task = Attack(args, parser)
    task.conduct()
    #debug
    # from taskAttack.util import  _PrintLogger
    # task.load_data(logger=_PrintLogger())
    # task.attacker_config(configs=task.AlgoArgs)
    # snr = 0
    # test_data = task.data_opts.snr_slice('test', snr)
    # sig_i, lab_i, _  = test_data
    # test_sig, test_lab = sig_i[0:1], lab_i[0:1]
    # print(test_sig.shape, test_lab.shape)
    # task.attacker.update_pnr_epsilon(test_sig, snr)
    # pert = task.attacker(test_sig, test_lab, debug_folder = task.pert_dir)
