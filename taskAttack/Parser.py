import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse


def get_parser(parsing = True):
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="Automatic Modulation Classification with pytorch")

    # -----------------------------------------------------------------------------------
    # Model name
    AttackArgs = parser.add_argument_group('Attack')
    AttackArgs.add_argument('-surrogate_model', type=str, default='awn', help='name of the surrogate model')
    AttackArgs.add_argument('-surrogate_ckp', type=str, default='', help='checkpoint file path of the surrogate model')
    AttackArgs.add_argument('-target_model', type=str, default='res', help='name of the target model')
    AttackArgs.add_argument('-target_ckp', type=str, default='', help='checkpoint file path of the target model')    # parser.add_argument('-target_model_path', type=str, default='data/RML2016.10a/pretrain_models/RML2016.10a_ResNet.best.pt', help='path of the surrogate model')
    AttackArgs.add_argument('-pnr', type=float, default=0, help='PNR(dB) of pertuation to noise. PSR(dB) = PNR(dB) - SNR(dB). Default with 0')
    AttackArgs.add_argument('-psr', type=float, default=-10, help='PSR(dB) of pertuation to noise. . Default with 0')
    AttackArgs.add_argument('-bound', type=str, default='psr', choices=['pnr', 'psr', 'epsilon'], help='Bound of the perturbations')
    AttackArgs.add_argument('-algo', default='fgsm', type=str, help='the attack algorithm')
    AttackArgs.add_argument('-algoconfig_file', type=str, default=None, help='the config file of the attack algorithm')
    AttackArgs.add_argument('-expconfig_dir', type=str, default='exp/attack/config', help='the config folder of the models')
    AttackArgs.add_argument('-norm', default='l2', choices=['l2', 'linfty'], help='the norm for constraining the perturbation')
    AttackArgs.add_argument('-epsilon', default=0.015, type=float, help='the stepsize to update the perturbation')
    AttackArgs.add_argument('-batch_size', '-bs', default=200, type=int, help='the batch_size to perform the attack')
    AttackArgs.add_argument('-model_init_path', type=str, default='', help='optional module containing model configs; empty uses <expconfig_dir>/<data>.')
    AttackArgs.add_argument('-attackset', type=str, default='test', help='the set of data to perform the attack on', choices=['test', 'attack'])
    # -----------------------------------------------------------------------------------
    # experimental Channel parameters
    ChannelArgs = parser.add_argument_group('Channel')
    ChannelArgs.add_argument('-k', type=float, default=1, help='k of path loss')
    ChannelArgs.add_argument('-d', type=float, default=10, help='d of path loss')
    ChannelArgs.add_argument('-gamma', type=float, default=2.7, help='gamma of path loss')
    ChannelArgs.add_argument('-margin', type=float, default= 0.1, help='margin of channel parameters')
    ChannelArgs.add_argument('-shadow_std', type=float, default=1, help='sigma of shadowing fading')
    ChannelArgs.add_argument('-rayleigh_std', type=float, default=1, help='sigma of rayleigh fading')
    ChannelArgs.add_argument('-timedelay', type=int, default=1, help='delay of time delay')
    ChannelArgs.add_argument('-use_cpls', default=True, action='store_true', help='experiment with combined path loss and shadowing')
    ChannelArgs.add_argument('-use_rayleigh', default=True, action='store_true', help='experiment with rayleigh fading')
    ChannelArgs.add_argument('-use_timedelay', default=False, action='store_true', help='experiment with time delay')
    ChannelArgs.add_argument('-use_awgn', default=False, action='store_true', help='experiment with AWGN')
    ChannelArgs.add_argument('-dynamic', default=False, action='store_true', help='experiment with dynamic channel')
    ChannelArgs.add_argument('-unit_gain', default=False, action='store_true', help='experiment with unit gain')
    ChannelArgs.add_argument('-channel_seed',type=int, default=2024, help='seed of channel effect')
    ChannelArgs.add_argument('-num_channel', type=int, default= 30, help='num of eval channel')
    ChannelArgs.add_argument('-rebound', action='store_true', default=False, help='whether use pilot-base rebound to amplify the power of adversary')
    ChannelArgs.add_argument('-pilot_n_channel', type=int, default= 200, help='num of pilot channel')
    ChannelArgs.add_argument('-pilot_seed', type=int, default= 3030, help='num of pilot channel')
    # Fading-Invariant Method (FIM) / Neural Inverse Model (NIM)
    FIMArgs = parser.add_argument_group('FIM')
    FIMArgs.add_argument('-nim_seed', type=int, default=2024,
                         help='first channel seed used to construct NIM pairs')
    FIMArgs.add_argument('-nim_model', type=str, default='nim', choices=['nim'],
                         help='Neural Inverse Model architecture')
    FIMArgs.add_argument('-nim_num_channels', type=int, default=200,
                         help='number of sampled channels used to train NIM')
    FIMArgs.add_argument('-nim_epochs', type=int, default=20,
                         help='maximum NIM training epochs')
    FIMArgs.add_argument('-nim_patience', type=int, default=50,
                         help='NIM early-stopping patience')
    FIMArgs.add_argument('-nim_batch_size', type=int, default=256,
                         help='NIM training batch size')
    # -----------------------------------------------------------------------------------
    # experimental location parameters
    ExpArgs = parser.add_argument_group('Experiment')
    ExpArgs.add_argument('-data', type=str, default='a', choices=['a','b','c', 'p', 'dr2', 'dr4', 'dr16', 'h'])
    ExpArgs.add_argument('-snr', nargs='+', default=['all'], help='experimental SNR in testing set. if "all" means all snr')
    ExpArgs.add_argument('--seed', type=int, default=2022)
    if os.name == 'nt':
        exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'\\', '').replace('\\', '/')
    else:
        exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    ExpArgs.add_argument('-exp_config', type=str, default=exp_config, help='folder name of the dataset')
    exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ExpArgs.add_argument('-exp_file', type=str, default=exp_file,help='file name of the dataset')
    exp_name = '{}.{}'.format(exp_config.replace('exp_config/', '').replace('/','.'), exp_file)

    ExpArgs.add_argument('-exp_name', type=str, default=exp_name, metavar='N',
                        help='exp_name  (default: mimo)')
    ExpArgs.add_argument('-exp_dataname_first', default=False,action='store_true',
                        help='Whether to use dataset name as the exp. host folder.')
    ExpArgs.add_argument('-tag',type=str, default='', help='additional experimental model tag')

    ExpArgs.add_argument('-test', default=False,action='store_true',
                        help='Whether to use unitTest mode')
    ExpArgs.add_argument('-clean', default=False, action='store_true',
                        help='Whether to clean the unitTest folder after each trail')
    ExpArgs.add_argument('-log_level', type=str, default='i', choices=['i','c'],help='experiment log level')

    # -----------------------------------------------------------------------------------
    # experiment hyper parameters
    # parser.add_argument('-epsilon', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    # parser.add_argument('-norm', default='l2', choices=['l2', 'linfty'], help='the norm for constraining the perturbation')
    # parser.add_argument('-alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    # parser.add_argument('-momentum', default=0., type=float, help='the decay factor for momentum based attack')
    # # parser.add_argument('-ensemble', action='store_true', help='enable ensemble attack')
    # parser.add_argument('-random_start', default=False, action='store_true', help='set random start')

    ExpArgs.add_argument('-cuda',default=False, action='store_true', help='experiment with cuda')
    ExpArgs.add_argument('-gid',type=int, default=0, help='default gpu id')

    # -----------------------------------------------------------------------------------
    # tune resource parameters

    if parsing:
        args = parser.parse_known_args()[0]
        return args, parser
    else:
        return parser

def get_group_args(parser, group_name = 'Attack', new_args = None):
    args = parser.parse_args()

    group_dict = {}
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}

    group =  argparse.Namespace(**group_dict)
    if new_args is not None:
        for key, value in vars(new_args).items():
            if key in group:
                setattr(group, key, value)
    return group
