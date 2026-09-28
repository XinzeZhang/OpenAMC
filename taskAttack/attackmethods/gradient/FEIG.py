import torch
import os

# from taskAttack.attackmethods.baseAttackAlgo import BaseAttackAlgo
from taskAttack.attackmethods.gradient.fgsm import IFGSM

"""
Meng, Y., Qi, P., Zheng, S., Cai, Z., Zhou, X., Jiang, T., 2025. Adversarial attack and reliable defense based on frequency domain feature enhancement for automatic modulation classification. IEEE Trans. Inf. Forensics Security 20, 3731–3744. https://doi.org/10.1109/TIFS.2025.3553806
"""


class FEIG(IFGSM):
    def set_specific_params(
        self,
    ):
        self.num_interpolations = 10
        self.mu = 0.20
        self.shifting = 0.075

    def generate_baseline_example(self, original_example):
        """
        通过上下平移原始样本生成基线样本

        参数：
            original_example: 原始信号样本 [batch_size, 2, 128]
        返回:
            baseline_example: 基线样本 [batch_size, 2, 128]
        """
        # 根据信号幅度自适应调整平移距离
        signal_power = torch.sqrt(
            torch.mean(torch.sum(original_example**2, dim=1), dim=-1)
        )
        signal_power = signal_power.view(-1, 1, 1)

        shift_distance = self.shifting * signal_power

        # 随机决定上移或下移
        direction = torch.randint(0, 2, (1,)).item() * 2 - 1  # 1 或 -1
        baseline_example = original_example + direction * shift_distance

        return baseline_example

    def generate_interpolated_examples(self, original_example, baseline_example):
        """
        生成原始样本和基线样本之间的N个插值样本

        参数:
            original_example: 原始样本 [batch_size, 2, 128]
            baseline_example: 基线样本 [batch_size, 2, 128]
        返回：
            interpolated_examples: N个插值样本 [N, batch_size, 2, 128]
        """
        alphas = (
            torch.linspace(0, 1, self.num_interpolations)
            .view(-1, 1, 1, 1)
            .to(self.device)
        )
        d = baseline_example - original_example

        # 基于公式(7): xbn = x′ + α · d
        interpolated_examples = original_example + alphas * d

        return interpolated_examples

    def generate_enhanced_baseline_examples(self, interpolated_examples):
        """
        为每个插值样本生成对应的增强基线样本

        参数:
            interpolated_examples: 插值样本 [N, batch_size, 2, 128]
        返回：
            enhanced_baseline_examples: N个增强基线样本 [N, batch_size, 2, 128]
        """
        N, batch_size, _, L = interpolated_examples.shape
        enhanced_baseline_examples = torch.zeros_like(interpolated_examples)

        for n in range(N):
            # 生成随机函数Z
            Z = torch.normal(1, 0.2, size=(batch_size, 2, L)).to(self.device)

            # 随机生成-1和1的函数I
            I = (
                2 * torch.randint(0, 2, size=(batch_size, 2, L)).float().to(self.device)
                - 1
            )

            # W = np.random.uniform(0.1, 0.5)

            # 生成频域信号
            freq_domain_signal = Z * I

            # 进行IFFT得到时域信号
            time_domain_signal = torch.fft.ifft(freq_domain_signal, dim=2).real

            # 功率归一化
            power = torch.sum(time_domain_signal**2, dim=(1, 2), keepdim=True)
            normalized_signal = time_domain_signal / (torch.sqrt(power) * L + 1e-10) * 2

            enhanced_baseline_examples[n] = normalized_signal

        return enhanced_baseline_examples

    def generate_enhanced_original_examples(
        self, interpolated_examples, enhanced_baseline_examples
    ):
        """
        生成增强原始样本

        参数:
            interpolated_examples: 插值样本 [N, batch_size, 2, 128]
            enhanced_baseline_examples: 增强基线样本 [N, batch_size, 2, 128]
        返回:
            enhanced_original_examples: N个增强原始样本 [N, batch_size, 2, 128]
        """
        # 基于公式(9): xen = xbn + μ · fn
        enhanced_original_examples = (
            interpolated_examples + self.mu * enhanced_baseline_examples
        )

        return enhanced_original_examples

    def compute_normalized_gradients(
        self, enhanced_original_examples, true_labels, clean_example
    ):
        """
        计算N个增强原始样本的归一化梯度

        参数：
            enhanced_original_examples: 增强原始样本 [N, batch_size, 2, 128]
            true_labels: 真实标签 [batch_size]
            clean_example: 干净样本 [batch_size, 2, 128]
        返回:
            normalized_gradients: 归一化梯度 [N, batch_size, 2, 128]
        """
        N, batch_size = (
            enhanced_original_examples.shape[0],
            enhanced_original_examples.shape[1],
        )
        normalized_gradients = torch.zeros_like(enhanced_original_examples)

        # 干净样本只需克隆一次
        clean_example_detached = clean_example.clone().detach()

        for n in range(N):
            # 提取当前增强原始样本
            curr_examples = enhanced_original_examples[n].clone().detach()

            # 创建一个新的需要梯度的差值张量
            delta = (curr_examples - clean_example_detached).requires_grad_(True)

            # Obtain the output (使用原始样本 + delta)
            logits = self.get_logits(clean_example_detached + delta)

            # Calculate the loss
            loss = self.get_loss(logits, true_labels)

            # 计算delta的梯度
            gradients = self.get_grad(loss, delta)

            # # L2归一化梯度 (公式10)
            grad_norm = (
                torch.sqrt(torch.sum(gradients**2, dim=(1, 2), keepdim=True)) + 1e-10
            )
            normalized_gradients[n] = gradients / grad_norm

            # gradients=gradients / (gradients.abs().mean(dim=(1, 2), keepdim=True))
            # normalized_gradients[n] = gradients

        return normalized_gradients

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, 2, T): tensor for input signal
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)
        momentum = 0
        for _ in range(self.epoch):
            adv_example = self.transform(data + delta, momentum=momentum)
            # 生成基线样本 上下平移得到
            baseline_examples = self.generate_baseline_example(adv_example)

            # 生成插值样本 原始对抗样本和平移后的baseline，进行插值，生成N个插值样本
            interpolated_examples = self.generate_interpolated_examples(
                adv_example, baseline_examples
            )

            # 生成增强基线样本   这个就是生成用于增强的频域信息
            enhanced_baseline_examples = self.generate_enhanced_baseline_examples(
                interpolated_examples
            )

            # 生成增强原始样本   这一步相当于插值样本+频域信息，最后生成N个增强原始样本
            enhanced_original_examples = self.generate_enhanced_original_examples(
                interpolated_examples, enhanced_baseline_examples
            )
            # enhanced_original_examples=interpolated_examples

            # 计算归一化梯度
            normalized_gradients = self.compute_normalized_gradients(
                enhanced_original_examples, true_labels=label, clean_example=data
            )

            # 计算积分梯度 (公式11)
            N = normalized_gradients.shape[0]
            integral_term = 0

            # for n in range(N - 1):
            #     # 获取相邻样本差异
            #     delta_x = enhanced_original_examples[n + 1] - enhanced_original_examples[n]

            #     # 计算平均梯度
            #     avg_gradient = 0.5 * (normalized_gradients[n] + normalized_gradients[n + 1])

            #     # 累积积分项
            #     integral_term += delta_x * avg_gradient

            for n in range(N):
                integral_term += normalized_gradients[n]
            integral_term = integral_term / N

            # Update adversarial perturbation
            delta = self.update_delta(delta, integral_term, self.alpha)
        return delta.detach()


# uncomment the following lines to make unit test
# from models.nn.AWN import AWN_config as awn
if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser
    from data import data_zoo

    args, parser = get_parser()

    args.algo = "feig"
    args.gid = 0
    model = "awn"
    args.surrogate_model = model
    args.target_model = model

    args.psr = -20
    args.data = "dr2"
    data_name = data_zoo[args.data]["data_name"]
    model_ckp = f"checkpoints/{data_name}/nature/{data_name}_{model}.best.pt"
    args.surrogate_ckp = model_ckp
    args.target_ckp = model_ckp

    args.cuda = True
    args.test = True
    args.clean = True
    args.batch_size = 200  # 12000 per snr

    # args.snr = [0, 10]
    task = Attack(args, parser)
    task.conduct(ave_confMax=False, show_variance=False)
