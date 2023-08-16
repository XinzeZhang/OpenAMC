import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.path.pardir, os.path.pardir))
from models._baseSetting import AMC_Net_base, AWN_base, mcldnn_base, vtcnn2_base, dualnet_base, resnet_base
from ray import tune
import torch
import numpy as np
import pickle
from task.TaskLoader import TaskDataset
from task.TaskWrapper import Task
from task.TaskParser import get_parser


class Data(TaskDataset):
    def __init__(self, opts):
        '''Merge the input args to the self object'''
        super().__init__(opts)

    def rawdata_config(self) -> object:
        self.data_name = 'RML2016.10a'
        self.batch_size = 64
        self.sig_len = 128

        self.val_size = 0.2
        self.test_size = 0.2

        self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                        b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}

        self.post_data_file = 'data/RML2016.10a/RML2016.10a_dict.split.pt'

    def load_rawdata(self, logger=None):
        file_pointer = 'data/RML2016.10a/RML2016.10a_dict.pkl'

        if logger is not None:
            logger.info('*'*80 + '\n' +
                        f'Loading raw file in the location: {file_pointer}')

        Signals = []
        Labels = []
        SNRs = []

        Set = pickle.load(open(file_pointer, 'rb'), encoding='bytes')
        snrs, mods = map(lambda j: sorted(
            list(set(map(lambda x: x[j], Set.keys())))), [1, 0])
        for mod in mods:
            for snr in snrs:
                Signals.append(Set[(mod, snr)])
                for i in range(Set[(mod, snr)].shape[0]):
                    Labels.append(mod)
                    SNRs.append(snr)

        Signals = np.vstack(Signals)
        Signals = torch.from_numpy(Signals.astype(np.float32))

        # mapping modulation formats(str) to int
        Labels = [self.classes[i] for i in Labels]
        Labels = np.array(Labels, dtype=np.int64)
        Labels = torch.from_numpy(Labels)

        return Signals, Labels, SNRs, snrs, mods


class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/2016.10a_AMC_Net.best.pt'


class awn(AWN_base):
    def task_modify(self):
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/2016.10a_AWN.best.pt'
        # self.hyper.pretraining_file = 'exp_results/RML2016.10a/ICASSP23/fit/awn/tuner/tpe/TuningCell_743b4993_8_gamma=0.6784,lr=0.0004,milestone_step=5_2023-08-09_03-35-19/checkpoint_000054/model.pth'

        self.tuner.resource = {"gpu": 0.5}

class dualnet(dualnet_base):
    def task_modify(self):
        self.hyper.epochs = 200
        self.hyper.patience = 15
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_dualnet.best.pt'

class vtcnn(vtcnn2_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.patience = 10
        self.hyper.gamma = 0.5
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_vtcnn.best.pt'

        self.tuner.resource = {"gpu": 0.5}  # set this for GPUs


class mcl(mcldnn_base):
    def task_modify(self):
        # self.hyper.batch_size = 64
        self.hyper.epochs = 200
        self.hyper.patience = 20
        # self.hyper.gamma = 0.5502
        # self.hyper.lr = 0.0015
        self.hyper.milestone_step = 2
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/2016.10a_MCLDNN.best.pt'

        self.tuner.num_samples = 40
        self.tuner.training_iteration = self.hyper.epochs
        # self.tuner.num_cpus = 32 * 3
        self.tuner.resource = {"gpu": 0.5}
        # self.tuner.points_to_evaluate=[{
        #     'lr':0.001,
        #     'gamma':0.8,
        #     'milestone_step':5,
        #     'batch_size': 400
        # }]
        # self.tuner.using_sched = False
        self.tuning.lr = tune.loguniform(1e-4, 2e-3)
        self.tuning.gamma = tune.uniform(0.5, 0.99)
        self.tuning.milestone_step = tune.qrandint(1, 10, 1)
        self.tuning.batch_size = tune.choice([64, 96, 128, 160, 192])

# resnet_base


class res(resnet_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1
        self.hyper.patience = 20
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_res.best.pt'
        
        self.tuner.num_samples = 40
        self.tuner.resource = {"gpu": 0.5}
        self.tuning.lr = tune.loguniform(1e-4, 2e-3)
        self.tuning.gamma = tune.uniform(0.5, 0.99)
        self.tuning.milestone_step = tune.qrandint(1, 10, 1)
        self.tuning.batch_size = tune.qrandint(64, 1024, 64)


if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    # args.gid = 2

    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # args.exp_name = 'ICASSP23'
    args.exp_name = 'res.tuning'
    # args.force_update = True # if need to rerun the model fiting or reload the pretraining_file (if os.path.exit(hyper.pretraining_file) to get the results, uncomment this line.)

    args.test = True
    args.clean = True
    args.model = 'res'

    task = Task(args)
    task.tuning()
    task.conduct()
