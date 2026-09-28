"""Multi-Variance-Reduced Gradients (MVRG) attack.

From “MVRG: Transferable Adversarial Attacks on Automatic Modulation
Classification via Multi-Variance-Reduced Gradients,” submitted to ICASSP 2027.
"""

import random

import torch
import torch.nn.functional as F

from taskAttack.attackmethods._baseAttackAlgo import BaseAttackAlgo


class MultiVarianceReduction(BaseAttackAlgo):
    """Variance-tuned Nesterov attack with multi-scale gradient smoothing."""

    def __init__(
        self,
        model,
        logger=None,
        epsilon=16 / 255,
        alpha=None,
        epoch=30,
        decay=1.0,
        num_neighbor=20,
        window_sizes=None,
        targeted=False,
        random_start=False,
        norm="l2",
        loss="crossentropy",
        device=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            logger=logger,
            epsilon=epsilon,
            targeted=targeted,
            random_start=random_start,
            loss=loss,
            norm=norm,
            device=device,
            epoch=epoch,
            alpha=epsilon / epoch if alpha is None else alpha,
            decay=decay,
            num_neighbor=num_neighbor,
            window_sizes=window_sizes or list(range(3, 64, 2)),
            **kwargs,
        )

    def set_specific_params(self):
        """Parameters are initialized by the constructor for config compatibility."""

    def smooth_grad(self, gradient, window_size=None):
        size = window_size or 5
        if size % 2 == 0:
            size += 1
        return F.avg_pool1d(gradient, size, stride=1, padding=size // 2)

    def transform(self, data, momentum, **kwargs):
        return data + self.alpha * self.decay * momentum

    def get_variance(self, data, delta, label, current_grad, momentum):
        average = torch.zeros_like(current_grad)
        for _ in range(self.num_neighbor):
            neighbor = torch.empty_like(delta).uniform_(-self.alpha, self.alpha)
            logits = self.get_logits(
                self.transform(data + delta + neighbor, momentum=momentum)
            )
            loss = self.get_loss(logits, label)
            gradient = super().get_grad(loss, delta)
            average += self.smooth_grad(
                gradient, window_size=random.choice(self.window_sizes)
            )
        return average / self.num_neighbor - current_grad

    def forward(self, data, label, **kwargs):
        if self.targeted:
            label = label[1]
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        delta = self.init_delta(data)
        momentum, variance = 0, 0
        for _ in range(self.epoch):
            logits = self.get_logits(
                self.transform(data + delta, momentum=momentum)
            )
            gradient = self.get_grad(self.get_loss(logits, label), delta)
            momentum = self.get_momentum(gradient + variance, momentum)
            variance = self.get_variance(
                data, delta, label, gradient, momentum
            )
            delta = self.update_delta(delta, momentum, self.alpha)
        return delta.detach()


class MVRG(MultiVarianceReduction):
    """MVRG using a Gaussian kernel to smooth neighboring gradients."""

    def set_specific_params(self):
        super().set_specific_params()
        self.kernel_size = getattr(self, "kernel_size", 11)
        self.sigma = getattr(self, "sigma", 2.0)

    def smooth_grad(self, gradient, window_size=None, sigma=None):
        size = window_size or self.kernel_size
        if size % 2 == 0:
            size += 1
        sigma = sigma or self.sigma
        radius = size // 2
        positions = torch.arange(
            -radius, radius + 1, device=gradient.device, dtype=gradient.dtype
        )
        kernel = torch.exp(-(positions**2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        original_shape = gradient.shape
        smoothed = F.conv1d(
            gradient.reshape(-1, 1, original_shape[-1]),
            kernel,
            padding=radius,
        )
        return smoothed.reshape(original_shape)


# Compatibility with the old experimental class name.
Multi_VarianceReduce = MultiVarianceReduction
