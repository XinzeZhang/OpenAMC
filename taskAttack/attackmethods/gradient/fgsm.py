import torch

from taskAttack.attackmethods._baseAttackAlgo import BaseAttackAlgo

class FGSM(BaseAttackAlgo):
    """
    FGSM Attack
    'Explaining and Harnessing Adversarial Examples (ICLR 2015)'(https://arxiv.org/abs/1412.6572)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255
    """

    def __init__(self, model, logger, epsilon=16/255, targeted=False, random_start=False, norm='l2', loss='crossentropy', device=None, **kwargs, ):
        super().__init__(model, logger, epsilon, targeted, random_start,loss, norm, device, **kwargs)
        self.alpha = epsilon
        self.epoch = 1
        self.decay = 0

class IFGSM(BaseAttackAlgo):
    """
    I-FGSM Attack
    'Adversarial Examples in the Physical World (ICLR 2017)'(https://arxiv.org/abs/1607.02533)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        targeted (bool): targeted/untargeted attack
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10
    """

    def __init__(self, model, logger,  epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=False, norm='l2', loss='crossentropy',device=None, **kwargs):
        super().__init__(model, logger,  epsilon, targeted, random_start,loss, norm, device, **kwargs)
        self.epoch = epoch
        self.alpha = epsilon / epoch
        self.decay = 0

class PGD(BaseAttackAlgo):
    """
    FGSM Attack
    'Explaining and Harnessing Adversarial Examples (ICLR 2015)'(https://arxiv.org/abs/1412.6572)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255
    """

    def __init__(self, model, logger, epsilon=16/255, alpha=1.6/255, epoch=10, targeted=False, random_start=True, norm='l2', loss='crossentropy',device=None, **kwargs):
        super().__init__(model,logger, epsilon, targeted, random_start,loss, norm, device, **kwargs)
        self.epoch = epoch
        self.alpha = epsilon / epoch
        self.decay = 0

class MIFGSM(BaseAttackAlgo):
    """
    MI-FGSM Attack
    'Boosting Adversarial Attacks with Momentum (CVPR 2018)'(https://arxiv.org/abs/1710.06081)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """

    def __init__(self, model, logger, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='l2', loss='crossentropy', device=None, **kwargs):
        super().__init__(model, logger, epsilon, targeted, random_start, loss, norm, device, **kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay


class NIFGSM(MIFGSM):
    """
    NI-FGSM Attack
    'Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)'(https://arxiv.org/abs/1908.06281)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """

    def __init__(self, model, logger, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='l2', loss='crossentropy', device=None,  **kwargs):
        super().__init__(model, logger, epsilon, alpha, epoch, decay, targeted, random_start,  norm, loss,  device, **kwargs)

    def transform(self, x, momentum, **kwargs):
        """
        look ahead for NI-FGSM
        """
        return x + self.alpha*self.decay*momentum

class VMIFGSM(BaseAttackAlgo):
    """
    VMI-FGSM Attack
    'Enhancing the transferability of adversarial attacks through variance tuning (CVPR 2021)'(https://arxiv.org/abs/2103.15571)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1.

    """

    def __init__(self,model, logger, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1., targeted=False,
                random_start=False, norm='l2', loss='crossentropy', device=None, **kwargs):
        super().__init__(model, logger, epsilon, targeted, random_start, loss, norm, device, **kwargs)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor

    def get_variance(self, data, delta, label, cur_grad, momentum, **kwargs):
        """
        Calculate the gradient variance
        """
        grad = 0
        for _ in range(self.num_neighbor):
            # Obtain the output
            # This is inconsistent for transform!
            logits = self.get_logits(self.transform(data+delta+torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device), momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta)

        return grad / self.num_neighbor - cur_grad

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for VMI-FGSM

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum, variance = 0, 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad+variance, momentum)

            # Calculate the variance
            variance = self.get_variance(data, delta, label, grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, momentum, self.alpha)

        return delta.detach()

class VNIFGSM(VMIFGSM):
    """
    VNI-FGSM Attack
    'Enhancing the transferability of adversarial attacks through variance tuning (CVPR 2021)'(https://arxiv.org/abs/2103.15571)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1.

    """

    def __init__(self, model, logger, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1., targeted=False,
                random_start=False, norm='l2', loss='crossentropy', device=None, **kwargs):
        super().__init__(model, logger, epsilon, alpha, beta, num_neighbor, epoch, decay, targeted, random_start, norm, loss, device, **kwargs)

    def transform(self, x, momentum):
        """
        look ahead for NI-FGSM
        """
        return x + self.alpha*self.decay*momentum