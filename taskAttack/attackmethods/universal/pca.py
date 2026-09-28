# import os,sys
# sys.path.append(os.path.join(os.path.dirname(
#     __file__), os.path.pardir, os.path.pardir, os.path.pardir))

# from baseAttacker import *
from taskAttack.attackmethods._baseAttackAlgo import *

class PCA_UAP(BaseAttackAlgo):
    '''
    Refer to M. Sadeghi and E. G. Larsson, “Adversarial Attacks on Deep-Learning Based Radio Signal Classification,” IEEE Wireless Communications Letters, vol. 8, no. 1, pp. 213–216, Feb. 2019, doi: 10.1109/LWC.2018.2867459.\n
    '''
    def __init__(self, model, logger, epsilon, targeted=False, random_start=False, loss='crossentropy', norm='l2', device=None, **kwargs):
        super().__init__(model, logger, epsilon, targeted, random_start,loss, norm, device, **kwargs)
        self.sample_nums = 50 if 'sample_nums' not in kwargs else kwargs['sample_nums'] # per snr
        self.alpha = epsilon
        self.epoch = 1
        self.decay = 0

    def snr_attack(self, data_loader, snr):
        snr_pert_dict = super().snr_attack(data_loader, snr)
        x,pert = snr_pert_dict['x'],snr_pert_dict['pert']

        import numpy as np
        pert = pert.view(pert.size(0), -1).t() # shape(2*sig_len, N)

        if self.sample_nums > x.size(0):
            self.logger.warning('!!!! The sample_num {} > data_num {}. sample_num has been changed to the data_num!!!!'.format(self.sample_nums, x.size(0)))
            self.sample_nums = x.size(0)
        select_N = np.random.choice(range(pert.size(1)), size = self.sample_nums, replace = False)
        pert = pert[:, select_N].cpu().numpy()

        from sklearn.decomposition import PCA
        uni_pert = PCA(n_components=1).fit_transform(pert)
        uni_pert = torch.tensor(uni_pert, device=x.device).t().view(2,-1)

        delta = torch.ones_like(x) * uni_pert
        delta = self.bound_pert(signals=x, snr= snr, delta=delta)
        # delta = self.init_delta(x)
        # grad = grad.to(delta.device)
        # momentum = self.get_momentum(grad, momentum=0)
        # delta = self.update_delta(delta, momentum, self.alpha).detach().nan_to_num().cpu()
        snr_pert_dict['pert'] = delta
        snr_pert_dict['uap'] = delta.mean(dim=0)

        self.logger.info('---------Within UAP---------')
        pnorm = torch.norm(delta.view(delta.size(0), -1), dim=-1).mean().item()
        PSR, PNR = cal_PNR(x, delta, snr)
        pinfo = 'SNR {}: P.E Diff: {:.2g}, PSR {:.1f} <-> PNR {:.1f}'.format(
            snr, self.epsilon - pnorm, PSR, PNR)
        self.logger.info(pinfo)

        return snr_pert_dict



if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser, get_group_args


    args, parser = get_parser()

    args.algo = 'pca'
    args.pnr = 0
    args.bound = 'pnr'
    args.psr = -10
    args.surrogate_model = 'awn'
    args.target_model = 'awn'
    args.cuda = True
    args.test = True
    args.clean = True
    task = Attack(args, parser)

    AlgoArgs = get_group_args(parser,new_args=args)

    task.conduct()