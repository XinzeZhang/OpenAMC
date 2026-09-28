import argparse, os, sys

def get_parser(parsing = True):
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description="Automatic Modulation Classification with pytorch")

    # -----------------------------------------------------------------------------------
    # Attack args
    AttackArgs = parser.add_argument_group('Attack')
    # AttackArgs.add_argument('-surrogate_model', type=str, default='awn', help='name of the surrogate model')
    # parser.add_argument('-target_model_path', type=str, default='data/RML2016.10a/pretrain_models/RML2016.10a_ResNet.best.pt', help='path of the surrogate model')
    AttackArgs.add_argument('-pnr', type=float, default=0, help='PNR(dB) of pertuation to noise. PSR(dB) = PNR(dB) - SNR(dB). Default with 0')
    AttackArgs.add_argument('-psr', type=float, default=-10, help='PSR(dB) of pertuation to noise. . Default with 0')
    AttackArgs.add_argument('-bound', type=str, default='psr', choices=['pnr', 'psr', 'epsilon'], help='Bound of the perturbations')
    AttackArgs.add_argument('-algo', default='fgm', type=str, help='the attack algorithm')
    AttackArgs.add_argument('-algoconfig_file', type=str, default=None, help='the config file of the attack algorithm')
    AttackArgs.add_argument('-norm', default='l2', choices=['l2', 'linfty'], help='the norm for constraining the perturbation')
    AttackArgs.add_argument('-epsilon', default=0.015, type=float, help='the stepsize to update the perturbation')
    AttackArgs.add_argument('-batch_size', '-bs', default=64, type=int, help='the batch_size to perform the attack and the adversarial training')
    AttackArgs.add_argument('-snr', nargs='+', default=['all'], help='experimental SNR in testing set. if "all" means all snr')
    AttackArgs.add_argument('-model_init_path', type=str, default='models', help='model_init_path, should be the relative path from the project root to the folder where the model is defined, default with "models" as the models/__init__.py is the entry of model import.')
    # -----------------------------------------------------------------------------------
    # Defense args
    defenseArgs = parser.add_argument_group('defense')
    defenseArgs.add_argument('-expconfig_dir', type=str, default='exp_config/defense/config', help='the config folder of the models')
    parser.add_argument('-using_snr', default=False, action='store_true',
                        help='Whether to using SNR values in the training and validation dataset, default is False')
    parser.add_argument('-finetune', default=False, action='store_true', help='Whether to finetune the model with provided pre-training file.')
    parser.add_argument('-defense_method', type=str, default='pgdat', help='the defense method to use, default is pgdat')
    parser.add_argument('-warmup_epoch', type=int, default=10, help='the number of epochs to warm up the training process')

    # ------------------------------------------------------------------------
    # experimental location parameters
    ExpArgs = parser.add_argument_group('Experiment')
    ExpArgs.add_argument('-model', '-target_model', type=str, default='awn', help='name of the target model')
    ExpArgs.add_argument('-data', type=str, default='a', choices=['a','b','c', 'p', 'dr2', 'dr4', 'dr16', 'h'])
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

    ExpArgs.add_argument('-cuda',default=True, action='store_true', help='experiment with cuda')
    ExpArgs.add_argument('-gid',type=int, default=0, help='default gpu id')

    # -----------------------------------------------------------------------------------
    # tune resource parameters

    if parsing:
        args = parser.parse_known_args()[0]
        return args, parser
    else:
        return None, parser

if __name__ == "__main__":
    args, parser = get_parser(parsing=True)
    print(args.surrogate_model)
    print(args.model)
    print(args.target_model)

    # Example usage
    # args, parser = get_parser()
    # print(args.data)
    # print(args.model)