import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import argparse


def get_parser(parsing = True):
    """
    Generate a parameters parser.
    """
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    parser = argparse.ArgumentParser(description="Automatic Modulation Classification with pytorch")

    # -----------------------------------------------------------------------------------
    # Model name
    parser.add_argument('-model', '-m', type=str, default='res', help='name of the implemented model')
    parser.add_argument('-seed', type=int, default=2022)
    # data name
    parser.add_argument('-data', type=str, default='a', choices=['a','b','c', 'p', 'dr2', 'dr4', 'dr16', 'h'], help='dataset name')
    parser.add_argument('-using_snr', default=False, action='store_true',
                        help='Whether to using SNR values in the training and validation dataset, default is False')
    # -----------------------------------------------------------------------------------
    # experimental location parameters
    exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    # if os.name == 'nt':
    #     exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'\\', '').replace('\\', '/')
    parser.add_argument('-exp_config', type=str, default=exp_config, help='folder name of the dataset')
    exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    parser.add_argument('-exp_file', type=str, default=exp_file,help='file name of the dataset')
    exp_name = '{}.{}'.format(exp_config.replace('exp_config/', '').replace('/','.'), exp_file)
    parser.add_argument('-exp_name', type=str, default=exp_name, metavar='N',
                        help='exp_name  (default: mimo)')
    parser.add_argument('-exp_dataname_first', default=False,action='store_true',
                        help='Whether to use dataset name as the exp. host folder.')
    parser.add_argument('-tag',type=str, default='', help='additional experimental model tag')
    parser.add_argument('-expconfig_dir', type=str, default='exp_config/aug/config/ori')
    parser.add_argument('-model_init_path', type=str, default='models', help='model_init_path, should be the relative path from the project root to the folder where the model is defined, default with "models" as the models/__init__.py is the entry of model import.')

    # -----------------------------------------------------------------------------------
    # experimental log parameters
    parser.add_argument('-force_update', default=False,action='store_true',
                        help='Whether to force execute the conduct function.')
    parser.add_argument('-finetune', default=False, action='store_true', help='Whether to finetune the model with provided pre-training file.')
    parser.add_argument('-test', default=False,action='store_true',
                        help='Whether to use unitTest mode')
    parser.add_argument('-clean', default=False, action='store_true',
                        help='Whether to clean the unitTest folder after each trail')
    # parser.add_argument('-logger_level', type=int, default=20, help='experiment log level')

    # -----------------------------------------------------------------------------------
    # experiment repetitive times
    parser.add_argument('-rep_times', type=int, default=1, help='experiment repetitive times')

    parser.add_argument('-cuda',default=True, action='store_true', help='experiment with cuda')
    parser.add_argument('-gid', '-g',type=int, default=0, help='default gpu id')
    parser.add_argument('-ocm', type=float, default=0,)
    # -----------------------------------------------------------------------------------
    # tune resource parameters
    # parser.add_argument('-tune', default=False, action='store_true', help='execute tune or not')
    # parser.add_argument('-cores', type=int, default=2, help='cpu cores per trial in tune')
    # parser.add_argument('-cards',type=int, default=0.25, help='gpu cards per trial in tune')
    # parser.add_argument('-tuner_iters', type=int, default=50, help='hyper-parameter search times')
    # parser.add_argument('-tuner_epochPerIter',type=int,default=1)

    if parsing:
        args = parser.parse_args()
        return args
    else:
        return parser