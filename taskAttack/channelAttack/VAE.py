from taskAttack.channelAttack.Wrapper import *
from models.nn._baseNet import hyper
import torch
from taskRecog.util import Opt, set_dataloader, set_fitset, save_training_process
from taskAttack.Parser import get_group_args
from taskAttack.util import signal_power, unit_gain, timedelay_channel,bound_pert
import os
from taskAttack.channelAttack.fim import FIMAttack
from taskAttack.attackmethods.universal.vae import Vae
class VaeAttack(FIMAttack):
    def evaluation(self, snrs_pert_dict, method = 'vae', ave_confMax=False):
        return super().evaluation(snrs_pert_dict, method, ave_confMax)

    def load_vae_model(self, snr):
        self.vae_fit_dir = os.path.join(
            self.pert_dir, f'VAE', f'snr{snr}')
        l_logger = self.logger_config(
            self.vae_fit_dir, stage='VAE', model_name='VAE')

        l_hyper = hyper()
        l_hyper.patience = 5
        l_hyper.epochs = 100
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and self.args.cuda:
            l_hyper.device = torch.device(
                'cuda:{}'.format(self.args.gid))
            torch.cuda.set_device(self.args.gid)
        else:
            l_hyper.device = torch.device('cpu')

        l_hyper.merge(opts=self.data_opts.info)
        l_hyper.trainer_module = ('taskAttack/channelAttack/neural_inverse_model.py'.replace(
            '.py', '').replace('/', '.'), 'ChannelTrainer')
        l_hyper.model_fit_dir = self.vae_fit_dir
        l_hyper.model_name = 'VAE'

        model = Vae(l_hyper, l_logger)
        return model


    def _snr_attack(self, snr):
        snr_pert_dict = self.attacker.snr_attack(
                    self.test_snrloaders[snr], snr)
        pert_ori = snr_pert_dict['pert']

        channels = []
        for i in range(self.FIMArgs.nim_num_channels):
            _, channel_gain, _ = self.channel_tranform(
                sig=torch.ones_like(pert_ori),
                snr_seed=self.FIMArgs.nim_seed + i,
                channel_seed=self.FIMArgs.nim_seed + i,
                dynamic=False, margin_sample=True
            )
            gain = torch.mean(channel_gain['gain'], dim=0, keepdim=True)
            channels.append(gain)

        if len(channels[0].size()) == 2:
            X = torch.stack(channels)
        else:
            X = torch.cat(channels)

        dataset = (X, X)
        vae = self.load_vae_model(snr)
        train_loader, _ = set_fitset(train_set=dataset, val_set=dataset, batch_size=200)
        epochs_stats = vae.xfit(train_loader, train_loader, xfit_stats=False)
        # 保存训练过程
        save_training_process(epochs_stats, plot_dir=os.path.join(self.vae_fit_dir, 'loss_curve', 'figure'))

        vae.eval()
        with torch.no_grad():
            Z = vae.forward_encoder(X.to(vae.hyper.device))
            Z = torch.mean(Z, dim=0, keepdim=True)
            Z = vae.forward_decoder(Z)

            pert_ori = snr_pert_dict['pert']
            pert_new = pert_ori / Z.cpu()
            snr_pert_dict['vae'] = pert_new
        return snr_pert_dict


    def snr_result(self, model, snr_pert_dict, snr, tag, logger):
        if tag == 'clean' or tag == 'adv.rece':
            result = super(Attack, self).snr_result(model, snr_pert_dict, snr, tag, logger)
        else:
            # conduct channel eval
            c_results = Results(list(range(self.ChannelArgs.num_channel)), model.hyper.num_classes)

            if self.ChannelArgs.rebound:
                self.amp_epsilon = self.get_amp(snr_pert_dict)

            for i in range(self.ChannelArgs.num_channel):
                _snr_pert_dict = deepcopy(snr_pert_dict)

                margin_tag = True if i > 0 else False
                if tag == 'adv.vae':
                    _pert = deepcopy(_snr_pert_dict['vae'])
                    _pert = bound_pert(_pert, snr_pert_dict['epsilon'], self.args.norm)
                # rebound the pert. with amp. epsilon
                    if self.ChannelArgs.rebound:
                        _pert = bound_pert(_pert, self.amp_epsilon, self.args.norm)
                else:
                    _pert = deepcopy(_snr_pert_dict['pert'])
                _pert, _, cinfo = self.channel_tranform(sig=_pert, snr_seed=self.ChannelArgs.channel_seed + i,
                                                        channel_seed=self.ChannelArgs.channel_seed + i,
                                                        dynamic=self.ChannelArgs.dynamic, margin_sample=margin_tag)
                logger.info(f'SNR {snr}: ' + cinfo)
                _snr_pert_dict['pert'] = _pert
                snr_result = super(Attack, self).snr_result(model, _snr_pert_dict, snr, tag, logger)
                c_results(snr_result, i)


            logger.critical(
                'SNR {}: Avg. P.E. {:.1E}\t PSR {:.1f} <-> PNR {:.1f}'.format(snr, c_results.pe.mean(), c_results.psr.mean(), c_results.pnr.mean()))

            result = dict(
                acc = c_results.Acc.mean(),
                cm = c_results.Confmat_Set.mean(axis=0),
                pe = c_results.pe.mean(),
                psr = c_results.psr.mean(),
                pnr = c_results.pnr.mean()
            )
        return result
