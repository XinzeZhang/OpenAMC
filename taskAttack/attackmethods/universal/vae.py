from taskAttack.attackmethods._baseAttackAlgo import *
from models.nn._baseNet import BaseNet, hyper
import torch
import torch.nn as nn
from taskRecog.util import set_fitset, save_training_process, set_logger
import numpy as np

class VAE_UAP(BaseAttackAlgo):
    '''
    Refer to M. Sadeghi and E. G. Larsson, “Adversarial Attacks on Deep-Learning Based Radio Signal Classification,” IEEE Wireless Communications Letters, vol. 8, no. 1, pp. 213–216, Feb. 2019, doi: 10.1109/LWC.2018.2867459.\n
    '''
    def __init__(self, model, logger, epsilon, targeted=False, random_start=False, loss='crossentropy', norm='l2', device=None, **kwargs):
        super().__init__(model, logger, epsilon, targeted, random_start,loss, norm, device, **kwargs)
        self.sample_nums = 40 if 'sample_nums' not in kwargs else kwargs['sample_nums'] # per snr
        self.alpha = epsilon
        self.epoch = 1
        self.decay = 0

    def logger_config(self, dir, stage, model_name, rewrite=True):
        log_path = os.path.join(dir, '{}.log'.format(stage))
        log_name = '{}'.format(model_name)
        logger = set_logger(log_path, log_name, rewrite=rewrite)
        return logger

    def load_vae_model(self, snr):
        self.vae_fit_dir = os.path.join(
            self.pert_dir, f'snr{snr}')
        l_logger = self.logger_config(
            self.vae_fit_dir, stage='VAE', model_name='VAE')

        l_hyper = hyper()
        l_hyper.patience = 5
        l_hyper.epochs = 100
        l_hyper.device = self.device
        l_hyper.merge(opts=self.data_info)


        l_hyper.trainer_module = ('taskAttack/channelAttack/neural_inverse_model.py'.replace(
            '.py', '').replace('/', '.'), 'ChannelTrainer')
        l_hyper.model_fit_dir = self.vae_fit_dir
        l_hyper.model_name = 'VAE'

        model = Vae(l_hyper, l_logger)
        return model

    def snr_attack(self, data_loader, snr):
        snr_pert_dict = super().snr_attack(data_loader, snr)
        x,pert = snr_pert_dict['x'],snr_pert_dict['pert']

        dataset = (pert,pert)
        vae = self.load_vae_model(snr)
        train_loader, _ = set_fitset(train_set=dataset, val_set=dataset, batch_size=200)
        epochs_stats = vae.xfit(train_loader, train_loader, xfit_stats=False)
        # 保存训练过程
        save_training_process(epochs_stats, plot_dir=os.path.join(self.vae_fit_dir, 'loss_curve', 'figure'))
        vae.eval()

        if self.sample_nums > x.size(0):
            self.logger.warning('!!!! The sample_num {} > data_num {}. sample_num has been changed to the data_num!!!!'.format(self.sample_nums, x.size(0)))
            self.sample_nums = x.size(0)
        select_N = np.random.choice(range(pert.size(0)), size = self.sample_nums, replace = False)
        _pert = pert[select_N,:,:]

        with torch.no_grad():
            Z = vae.forward_encoder(_pert.to(vae.hyper.device))
            Z = torch.mean(Z, dim=0, keepdim=True)
            Z = vae.forward_decoder(Z)
            delta = torch.ones_like(pert) * Z.cpu()
            delta = self.bound_pert(signals=x, snr= snr, delta=delta)
        snr_pert_dict['uap'] = delta.mean(dim=0)
        snr_pert_dict['pert'] = delta

        self.logger.info('---------Within UAP---------')
        pnorm = torch.norm(delta.view(delta.size(0), -1), dim=-1).mean().item()
        PSR, PNR = cal_PNR(x, delta, snr)
        pinfo = 'SNR {}: P.E Diff: {:.2g}, PSR {:.1f} <-> PNR {:.1f}'.format(
            snr, self.epsilon - pnorm, PSR, PNR)
        self.logger.info(pinfo)

        return snr_pert_dict

class Vae(BaseNet):# remain debug
    def initialize_arch(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(1,3), padding='same'),
            nn.Conv2d(128,40, kernel_size=(2,3), padding='same'),
            nn.Flatten(),
            nn.Linear(10240, 16),
            nn.Linear(16,4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 10240),
            nn.Unflatten(1, [40, 2, self.hyper.sig_len]),
            nn.Conv2d(40,128, kernel_size=(2,3), padding='same'),
            nn.Conv2d(128,128, kernel_size=(1,3), padding='same'),
            nn.Conv2d(128,1, kernel_size=(3,3), padding='same')
        )

    def forward(self, x):
        z = self.forward_encoder(x)
        out = self.forward_decoder(z)
        return out

    def forward_encoder(self, x):
        out = torch.unsqueeze(x, 1)
        for m in self.encoder.children():
            out = m(out)
        return out

    def forward_decoder(self, z):
        for m in self.decoder.children():
            z = m(z)
        z = torch.squeeze(z, dim=1)
        return z

    def loader_predict(self, data_loader):
        with torch.no_grad():
            pre_lab_all = []
            label_all = []

            for sig_batch, lab_batch in data_loader:
                sig_batch = sig_batch.to(self.hyper.device)
                pre_lab = self.forward(sig_batch).cpu()
                pre_lab_all.append(pre_lab)
                label_all.append(lab_batch)

            pre_lab_all = torch.cat(pre_lab_all)
            label_all = torch.cat(label_all)

        return pre_lab_all, label_all

if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser, get_group_args
    args, parser = get_parser()

    args.algo = 'vae'
    args.pnr = 0
    args.snr = [4]
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

    # from models.nn._baseNet import hyper
    # from taskRecog.util import set_logger

    # vhyper = hyper()
    # vhyper.patience = 10
    # vhyper.epochs = 20
    # gid = 2
    # cuda_exist = torch.cuda.is_available()
    # if cuda_exist:
    #     vhyper.device = torch.device(
    #         'cuda:{}'.format(gid))
    #     torch.cuda.set_device(gid)
    # else:
    #     vhyper.device = torch.device('cpu')

    # vhyper.sig_len = 128
    # vhyper.batch_size = 64
    # vhyper.trainer_module = ('taskAttack/channelAttack/neural_inverse_model.py'.replace(
    #     '.py', '').replace('/', '.'), 'ChannelTrainer')
    # vhyper.model_fit_dir = 'yield_test/testVAE'
    # vhyper.model_name = 'vae'

    # import os
    # logger = set_logger(log_path=os.path.join(vhyper.model_fit_dir, 'vae.log'), log_name='vae')

    # vae = Vae(vhyper, logger)

    # x = torch.rand((100,2,128)).float().to(vhyper.device)
    # y = vae(x)
