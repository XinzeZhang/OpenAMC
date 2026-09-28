import os
from taskRecog.util import fix_seed
import numpy as np
# from taskAttack.attackmethods.util import AttackConfig
from tqdm.auto import tqdm
from taskAttack.Wrapper import Attack as BaseAttack
from taskAttack.Wrapper import Results
from random import uniform
import random
from copy import copy, deepcopy
from taskAttack.util import rayleigh_channel, pathloss_shadowing_channel, timedelay_channel, AWGN_channel, signal_power, cal_PNR, plot_snr_acc,signal_energy, bound_pert
from taskAttack.Parser import get_group_args
import torch


class Attack(BaseAttack):
    def __init__(self, args, parser=None):
        super().__init__(args, parser)
        self.ChannelArgs = get_group_args(
            self.parser, new_args=self.args, group_name='Channel')

    def evaluation(self, snrs_pert_dict, ave_confMax=False):
        self.eval_dir = os.path.join(self.pert_dir, f'target.{self.args.target_model}')
        elogger = self.logger_config(dir=self.eval_dir, stage='evaluation',
                                     model_name='{}->{}'.format(self.args.surrogate_model, self.args.target_model))

        try:
            self.victim = self.model_import(
                elogger, model_name=self.args.target_model)
            elogger.info('Save accuracy file to the location: {}/'.format(
                os.path.join(self.pert_dir, f'target.{self.args.target_model}')))

            # sig_power = {f'SNR {snr}': '{:.1E}'.format(signal_power(
            #     snrs_pert_dict[snr]['x']).mean()) for snr in self.exp_snrs}
            # elogger.info(
            #     ' '*5 + f'Rece. Signal. (after channel) Power: \n{sig_power}')

            clean = super().record_result(
                snrs_pert_dict, self.victim, elogger, 'clean', ave_confMax)

            advRece = self.rece_result(
                snrs_pert_dict, self.victim, elogger, 'adv.rece', ave_confMax)

            advTran = self.rece_result(
                snrs_pert_dict, self.victim, elogger, 'adv.trans', ave_confMax)

            elogger.critical(
                'Clean Acc. -> Adv.Rece. Acc. <- Adv.Trans. Acc. ')
            elogger.critical(
                'Acc: Clean -> Adv.R(w/o ce) <- Adv.T(w/ ce)')
            for i, snr in enumerate(self.exp_snrs):
                elogger.critical(
                    'SNR {}: {:.2f}% -> {:.2f}% <- {:.2f}%'.format(snr, clean.Acc[i] * 100, advRece.Acc[i] * 100, advTran.Acc[i] * 100))

            elogger.critical(
                'Overall: {:.2f}% -> {:.2f}% <- {:.2f}%'.format(clean.Acc.mean() * 100, advRece.Acc.mean() * 100, advTran.Acc.mean() * 100))

            acc_list = [(clean.Acc, 'clean'),
                        (advRece.Acc, 'adv.rece'),
                        (advTran.Acc, 'adv.trans')]
            plot_snr_acc(acc_list, self.exp_snrs,
                         self.data_name, self.eval_dir)

            return dict(clean=clean, advtrans=advTran, advrece=advRece)

        except:
            elogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def rece_result(self, snrs_pert_dict, model, logger, tag='adv.rece', ave_confMax=False):
        # logger.info('>'*10 + f'Acc. on Pert. Data' + '>'*10)
        results = super().record_result(
            snrs_pert_dict, model, logger, tag, ave_confMax)
        self.logging_channelpower(tag,results,logger)
        return results

    def get_amp(self, snr_pert_dict):
        pilot = torch.ones_like(snr_pert_dict['pert'])
        pcr = torch.empty(self.ChannelArgs.pilot_n_channel)
        for i in range(self.ChannelArgs.pilot_n_channel):
            pilot_after, _, _ = self.channel_tranform(sig=pilot, snr_seed=self.ChannelArgs.pilot_seed + i,
                                                        channel_seed=self.ChannelArgs.pilot_seed + i, dynamic=False, margin_sample = True)
            pratio = signal_energy(pilot) / signal_energy(pilot_after)
            pcr[i] = torch.sqrt(pratio.mean())
        pcr_mean = pcr.mean()
        self.amp_epsilon = snr_pert_dict['epsilon'] * pcr_mean
        return self.amp_epsilon


    def snr_result(self, model, snr_pert_dict, snr, tag, logger):
        if tag != 'adv.trans':
            result = super().snr_result(model, snr_pert_dict, snr, tag, logger)
        else:
            c_results = Results(list(range(self.ChannelArgs.num_channel)), model.hyper.num_classes)

            if self.ChannelArgs.rebound:
                self.amp_epsilon = self.get_amp(snr_pert_dict)

            with tqdm(total=self.ChannelArgs.num_channel, desc='Channels', mininterval=0.3, colour='green', leave=False) as pbar:
                for i in range(self.ChannelArgs.num_channel):
                    _snr_pert_dict = deepcopy(snr_pert_dict)
                    margin_tag = True if i > 0 else False
                    _pert = deepcopy(_snr_pert_dict['pert'])
                    # rebound the pert. with amplified epsilon
                    if self.ChannelArgs.rebound:
                        _pert = bound_pert(_pert, self.amp_epsilon, self.args.norm)

                    _pert, _, cinfo = self.channel_tranform(sig=_pert, snr_seed=self.ChannelArgs.channel_seed + i,                                                           channel_seed=self.ChannelArgs.channel_seed + i, dynamic=self.ChannelArgs.dynamic, margin_sample=margin_tag)
                    logger.info(f'SNR {snr}: ' + cinfo)
                    _snr_pert_dict['pert'] = _pert
                    snr_result = super().snr_result(model, _snr_pert_dict, snr, tag, logger)
                    c_results(snr_result, i)
                    pbar.update(1)


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

    def channel_tranform(self, sig, snr_seed, channel_seed, margin_sample=True, dynamic=False):
        # with temporary_seed(seed):
        fix_seed(snr_seed)
        signal = sig.detach().clone()
        gain = torch.ones_like(signal)
        T, noise, ChannelInfo = 0, torch.ones_like(signal), ''
        if self.ChannelArgs.use_timedelay:
            T = random.sample(range(1, 127),
                              1)[0] if margin_sample else self.ChannelArgs.timedelay
            ChannelInfo += f'Time Delay: {dict(T=T)}; '
            signal = timedelay_channel(X=signal, T=T)

        if self.ChannelArgs.use_cpls:
            cpls_config = dict(K=self.ChannelArgs.k, distance=self.ChannelArgs.d, gamma=self.ChannelArgs.gamma,
                               sigma=self.ChannelArgs.shadow_std, dynamic=dynamic, unit = self.ChannelArgs.unit_gain)
            if margin_sample:
                for key in ['K', 'distance', 'gamma']:
                    value = cpls_config[key]
                    cpls_config[key] = uniform(
                        value - self.ChannelArgs.margin*value, value + self.ChannelArgs.margin*value)
            log_cpls = {key: value if isinstance(value, bool) else "{:.1f}".format(
                        value) for key, value in cpls_config.items()}
            cpls_config['seed'], log_cpls['seed'] = channel_seed, channel_seed
            ChannelInfo += f'Combined Path loss and Shadowing: {log_cpls}; '

            signal, _gain = pathloss_shadowing_channel(X=signal, **cpls_config)
            gain = gain * _gain

        if self.ChannelArgs.use_rayleigh:
            rf_config = dict(
                sigma=self.ChannelArgs.rayleigh_std, dynamic=dynamic, unit = self.ChannelArgs.unit_gain)
            log_rf = copy(rf_config)
            log_rf = {key: value if isinstance(value, bool) else "{:.1f}".format(
                value) for key, value in rf_config.items()}

            rf_config['seed'], log_rf['seed'] = channel_seed, channel_seed
            ChannelInfo += f'Rayleigh Fading: {log_rf}'
            signal, _gain = rayleigh_channel(X=signal, **rf_config)
            gain = gain * _gain

        if self.ChannelArgs.use_awgn:
            signal, noise = AWGN_channel(X=signal)

        channel_gain = dict(gain=gain, noise=noise, T=T)
        return signal, channel_gain, ChannelInfo


    def logging_channelpower(self, tag, result, logger):
        newPert_power = {f'SNR {snr}': 'P.E: {:.1E}, PSR {:.1f} <-> PNR {:.1f}'.format(
            result.pe[s], result.psr[s], result.pnr[s]) for s, snr in enumerate(self.exp_snrs)}
        if tag == 'adv.rece':
            logger.info(f'Rece. {tag}. (before channel) {newPert_power}')
        else:
            logger.info(
            ' '*5 + f'Trans. Pert. (after channel) {newPert_power} ')


if __name__ == "__main__":
    from taskAttack.Parser import get_parser, get_group_args

    args, parser = get_parser()

    args.snr = [10]

    args.exp_name = 'channel_test'
    # args.unit_gain = True
    args.channel_seed = 8000
    args.algo = 'fgm'
    args.pnr = 0
    args.bound = 'psr'
    args.psr = -10
    args.surrogate_model = 'awn'
    args.target_model = 'awn'
    args.cuda = True
    args.test = True
    args.clean = True
    # args.seed = 2024
    args.num_channel = 30
    args.rebound = True
    # args.log_level = 'i'
    # args.data = 'rml16b'
    # args.bound = 'epsilon'

    task = Attack(args, parser)
    task.conduct()
