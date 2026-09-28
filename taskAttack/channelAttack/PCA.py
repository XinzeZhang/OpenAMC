# from taskAttack.channelAttack.Wrapper import *
# from models.nn._baseNet import hyper
# import torch
# from taskRecog.util import Opt, set_dataloader, set_fitset, save_training_process
# from taskAttack.Parser import get_group_args
# from taskAttack.util import signal_power, unit_gain, timedelay_channel
from taskAttack.channelAttack.fim import FIMAttack
from copy import copy, deepcopy
from taskAttack.Wrapper import Results
import torch
from taskAttack.channelAttack.Wrapper import Attack
from taskAttack.util import timedelay_channel,bound_pert
from sklearn.decomposition import PCA
from tqdm.auto import trange

class PcaAttack(FIMAttack):
    '''B. Kim, Y. E. Sagduyu, K. Davaslioglu, T. Erpek, and S. Ulukus, “Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers,” IEEE Transactions on Wireless Communications, vol. 21, no. 6, pp. 3868–3880, Jun. 2022, doi: 10.1109/TWC.2021.3124855.
    '''
    def evaluation(self, snrs_pert_dict, method = 'pca', ave_confMax=False):
        return super().evaluation(snrs_pert_dict, method, ave_confMax)

    def _snr_attack(self, snr):
        '''The actual attack codes of attacking the model for an snr-specific data'''
        snr_pert_dict = self.attacker.snr_attack(
                    self.test_snrloaders[snr], snr)
        pert_ori = snr_pert_dict['pert']

        Perts = []
        for i in range(self.FIMArgs.nim_num_channels):
            _, channel_gain, _ = self.channel_tranform(
                sig=torch.ones_like(pert_ori),
                snr_seed=self.FIMArgs.nim_seed + i,
                channel_seed=self.FIMArgs.nim_seed + i,
                dynamic=False, margin_sample=True
            )
            # pert_trans = (pert_ori - channel_gain['noise'])/ channel_gain['gain']
            # T_value = channel_gain['T']
            # pert_trans = timedelay_channel(pert_trans, T= -T_value)
            pert_trans = pert_ori / channel_gain['gain']
            Perts.append(pert_trans)

        Perts = torch.stack(Perts, dim=-1)
        pert_new = torch.empty_like(pert_ori)
        for j in trange(Perts.size(0)):
            _pert = Perts[j] # size (2, len, channel_count)
            _pert = _pert.view(-1, _pert.size(-1)) # size (2 * len, channel_count)
            _pert = _pert.cpu().numpy()
            uni_pert = PCA(n_components=1).fit_transform(_pert)
            uni_pert = torch.tensor(uni_pert, device=pert_ori.device).view(2,-1) # size (2, len)
            pert_new[j] = uni_pert

        # toDo: pert_new need to be bounded. May be with the average energy of Perts[j], such as _pert.mean(dim=-1), as the expected ampl. energy for sending pert.
        snr_pert_dict['pca'] = pert_new
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
                if tag == 'adv.pca':
                    _pert = deepcopy(_snr_pert_dict['pca'])
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
