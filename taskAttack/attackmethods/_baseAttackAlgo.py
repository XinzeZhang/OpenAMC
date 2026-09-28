import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.path.pardir, os.path.pardir))

import torch
import torch.nn as nn
# from taskAttack.util import clamp

from taskRecog.util import Opt
from tqdm.auto import tqdm
from taskAttack.util import cal_PNR, _PrintLogger
from taskAttack.util import bound_pert




class BaseAttackAlgo(Opt):
    '''
    Base class for all attacks. Refer from https://github.com/Trustworthy-AI-Group/TransferAttack
    '''
    def __init__(self, model, logger=None, epsilon=None, targeted=False, random_start=False, loss='crossentropy', norm='l2', device=None, **kwargs):
        """
        Initialize the hyperparameters

        Arguments:
            attack (str): the name of attack.
            model (str): the surrogate model for attack.
            epsilon (float): the perturbation budget.
            targeted (bool): targeted/untargeted attack.
            random_start (bool): whether using random initialization for delta.
            norm (str): the norm of perturbation, l2/linfty.
            loss (str): the loss function.
            device (torch.device): the device for data. If it is None, the device would be same as model
        """
        if norm not in ['l2', 'linfty']:
            raise Exception("Unsupported norm {}".format(norm))
        if epsilon is None:
            raise TypeError("epsilon must be provided")
        # self.attack = attack
        self.model = model
        self.logger = logger if logger is not None else _PrintLogger()

        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.device = next(self.model.parameters()
                           ).device if device is None else device
        self.loss = self.loss_function(loss)

        self.epoch = 1
        self.alpha = epsilon / 1
        self.decay = 0



        for key, value in kwargs.items():
            setattr(self, key, value)

        self.set_specific_params()

    def set_specific_params(self, **kwargs):
        pass


        # self.merge(kwargs)
        # self.data_min, self.data_max = data_min, data_max

    def update_params(self, **kwargs):
        # self.merge(kwargs, ele_s=['logger'])
        for (arg, value) in self.dict.items():
            self.logger.info("Argument %s: %r", arg, value)
        updated_dict = self.update(kwargs)
        for (arg, value) in updated_dict.items():
            self.logger.info("Updated Argument %s: %r", arg, value)

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
            # Obtain the output
            logits = self.get_logits(
                self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, momentum, self.alpha)

        delta = bound_pert(delta, self.epsilon, self.norm)
        return delta.detach()

    def get_logits(self, x, **kwargs):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.model.logits(x)

    def get_loss(self, logits, label):
        """
        The loss calculation, which should  be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    def get_momentum(self, grad, momentum, **kwargs):
        """
        The momentum calculation
        """
        return momentum * self.decay + grad / (grad.abs().mean(dim=(1, 2), keepdim=True))

    def init_delta(self, data, **kwargs):
        '''
        Note! There are some different versions of implementing random perturb.\n
        E.g.
        1) uniform noise as https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py \n
        2) guassian noise as https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py \n
        3) mixed noise as https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/attack.py
        '''
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            # if self.norm == 'linfty':
            #     # refer to above-mentioned method 1.
            #     delta.uniform_(-self.epsilon, self.epsilon)
            # else:
            #     delta.normal_()
            #     n = torch.norm(delta.view(delta.size(0), -1),
            #                    dim=1).view(-1, 1, 1)
            #     r = torch.zeros_like(data).uniform_(0, 1).to(self.device)
            #     delta *= r*self.epsilon / n
            delta.uniform_(-self.epsilon, self.epsilon)
            # delta = clamp(delta, self.data_min-data, self.data_max-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, grad, alpha, **kwargs):
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
                                                       1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        # delta = clamp(delta, self.data_min-data, self.data_max-data) # This line only have meanling in image domain, which is equal to delta + data = clamp(delta + data , self.data_min, self.data_max). However, it should not be operated in raido domain.
        return delta.detach().requires_grad_(True)

    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception("Unsupported loss {}".format(loss))

    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        # Use eval mode for attack generation (deterministic behaviour)
        self.model.eval()

        # Safely call the attack while handling cudnn/RNN mode issues.
        # Use getattr to tolerate models without `has_rnn` attribute.
        has_rnn = getattr(self.model, 'has_rnn', False)

        if torch.cuda.is_available() and has_rnn:
            # torch.backends.cudnn.enabled = False
            self.model.rnn_reactivate()

        delta = self.forward(*input, **kwargs)
        delta = bound_pert(delta, self.epsilon, self.norm)
        # finally: Always re-enable cudnn when available
        # if torch.cuda.is_available():
        #     torch.backends.cudnn.enabled = True
        return delta


    def update_pnr_epsilon(self, signals, snr):
        '''We use the average sig_power to determine the perturbation power like in https://github.com/meysamsadeghi/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems/blob/c1a2108ef0f22ee844ade40342c62adc4df6e191/Adv_Attack_Modulation_Classification/dirty_old_implementation/fig3/fig3_function.py#L33. The linked code has problem and been corrected in this implementation.
        '''
        if self.bound != 'epsilon':
            from taskAttack.util import signal_energy, signal_db2pow
            sig_energy = signal_energy(signals).mean()
            # sig_energy = sig_energy
            # print('Signal Power --> Max:{:.2E}\tMin:{:.2E}\tAvg:{:.2E}\tStd:{:.2E}'.format(sig_power.max(),sig_power.min(),sig_power.mean(),sig_power.std()))
            if self.bound == 'pnr':
                PSR, PNR = self.pnr - snr, self.pnr
            elif self.bound == 'psr':
                PSR, PNR = self.psr, self.psr + snr
            gain = signal_db2pow(PSR)
            pert_norm = torch.sqrt(sig_energy * gain)
            self.epsilon = pert_norm
            self.alpha = self.epsilon / self.epoch
            self.logger.info('Update pert. bound (dB) with PSR {} <->  PNR {} - SNR {} <-> Epsilon {:.2g} <-> Alpha {:.2g}'.format(PSR, PNR, snr, self.epsilon, self.alpha))


    def snr_attack(self, data_loader, snr):
        if self.bound in ['pnr', 'psr']:
            signals = torch.cat([x for x,_ in data_loader])
            self.update_pnr_epsilon(signals, snr)

        pert= []
        with tqdm(total=len(data_loader), desc='Batches', mininterval=0.3, colour='blue', leave=False) as bbar:
            for _signals, _labels in data_loader:

                perturbations = self(_signals, _labels)  # return tensor
                perturbations = perturbations.detach().cpu()
                # x.append(_signals)
                # label.append(_labels)
                pert.append(perturbations)
                bbar.update(1)
        # x, pert, label = torch.cat(x), torch.cat(
        #     pert),  torch.cat(label)
        pert = torch.cat(pert)
        if torch.isnan(pert).any():
            self.logger.error("NaN detected in perturbations! Setting NaNs to 0")
        if torch.isinf(pert).any():
            self.logger.error("Inf detected in perturbations! Setting Inf to 0")
        pert = pert.nan_to_num()

        pert =  bound_pert(pert, self.epsilon, self.norm)
        if torch.isnan(pert).any():
            self.logger.error("NaN detected in perturbations! Setting NaNs to 0")
        if torch.isinf(pert).any():
            self.logger.error("Inf detected in perturbations! Setting Inf to 0")
        # snr_pert_dict = dict(x=x,pert = pert, label = label, snr=snr, epsilon = self.epsilon)
        snr_pert_dict = dict(pert = pert, snr=snr, epsilon = self.epsilon)

        pnorm = torch.norm(pert.view(pert.size(0), -1), dim=-1).mean().item()
        PSR, PNR = cal_PNR(signals, pert, snr)
        pinfo = 'SNR {}: P.E Diff: {:.2g}, PSR {:.1f} <-> PNR {:.1f}'.format(
            snr, self.epsilon - pnorm, PSR, PNR)
        self.logger.info(pinfo)
        return snr_pert_dict

    # def bound_pert(self, signals, snr, delta):
    #     self.update_pnr_epsilon(signals, snr)
    #     delta = bound_pert(delta, self.epsilon, self.norm)
    #     return delta
    # def snrs_attack(self, dataloader_list):

    #     # X,Pert,X_adv,Label = [],[],[],[]
    #     snr_pert_dict = {} # toDo

    #     with tqdm(total=len(dataloader_list), desc='SNRs', mininterval=0.3, colour='red') as pbar:
    #         for snr, snr_dataloader in dataloader_list:
    #             pbar.set_description_str(f'SNR = {snr} ')

    #             x, pert, x_adv, label = self.snr_attack(snr_dataloader)
    #             snr_pert_dict[snr] = dict(x=x,pert = pert, x_adv = x_adv, label = label)
    #             # X.append(x), Pert.append(pert), X_adv.append(x_adv), Label.append(label)
    #             pbar.update(1)
    #     # snr_pert_dict = dict(X=X, Pert=Pert, X_adv=X_adv, Label=Label)

    #     return snr_pert_dict

if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser, get_group_args


    args, parser = get_parser()

    args.snr = [0, 10]

    args.algo = 'pgd'
    args.gid = 1
    args.surrogate_model = 'mcl'
    args.target_model = 'mcl'
    args.cuda = True
    args.test = True
    args.clean = True
    args.data = 'rml16b'
    args.batch_size = 3000 # 12000 per snr

    task = Attack(args, parser)
    task.conduct()
