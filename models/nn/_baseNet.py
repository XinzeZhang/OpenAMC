import importlib
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy
import numpy as np
from tqdm.auto import tqdm
from taskRecog.util import Opt

from torchinfo import summary
# from models.nn._baseTrainer import Trainer
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

class BaseNet(nn.Module):
    def __init__(self, hyper = None, logger = None):
        '''
        self.hyper = hyper\n
        logger = logger
        '''
        super(BaseNet, self).__init__()
        self.hyper = hyper
        self.has_rnn = False

        self.initialize_arch()

        self.to(self.hyper.device)

        info = logger.info if logger is not None else print
        for (arg, value) in hyper.dict.items():
            info(f"Argument {arg}: {value}")

        info(f'Initializing Model: \n{str(self)}') if logger is not None else print(f'Initializing Model: \n{str(self)}')

        with torch.no_grad():
            info(summary(self, (2, hyper.sig_len), batch_dim=0, verbose = 0, device = self.hyper.device))

        # param_num = sum(p.numel() for p in list(self.parameters()))  / 1000000.0
        # info(f">>> Total params: {param_num:.2f} M") if logger is not None else print(">>> Total params: {:.2f} M".format(param_num))

    def initialize_arch(self,):
        pass

    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x

    def load_pretraing_file(self, file_path = None, tag = 'pretraining', logger = None):
        try:
            device_id = next(self.parameters()).get_device()

            if device_id > -1:
                model_state = torch.load(file_path, map_location = f'cuda:{device_id}')
            else:
                model_state = torch.load(file_path, map_location='cpu')
            # model_state = torch.load(file_path, map_location = f'cuda:{device_id}') if device_id > -1 else torch.load(file_path)

            self.load_state_dict(model_state)

            load_info = f'Successfully loading the {tag} file in the location: {file_path}!'
            logger.info(load_info) if logger is not None else print(load_info)
        except Exception:
            error_info = '{}\nGot an error on loading the {} file in the location: {}.\n{}'.format('!'*50, tag, file_path, '!'*50)
            logger.exception(error_info) if logger is not None else print(error_info)
            raise SystemExit()

    def xfit(self, train_loader, val_loader, trainer_class=None, finetune = False, xfit_stats=True, auto_pt_check = True, logger = None, **kwargs):
        """
        If self.hyper has pretraining_file, then directly loading the pretraining states;\n
        else, training the model with the Traniner in models.nn._baseTrainer.\n

        Return: fit_info
        """
        logger.critical('Start fit.')
        pretraining_tag = False
        fit_info = None
        # print('pretraining_file' in self.hyper.dict)
        # print(self.hyper.pretraining_file is not None)
        # print(os.path.exists(self.hyper.pretraining_file))
        if 'pretraining_file' in self.hyper.dict and self.hyper.pretraining_file is not None and os.path.exists(self.hyper.pretraining_file):

            logger.info(f'Finding pretraining file in the location {self.hyper.pretraining_file}')
            self.load_pretraing_file(file_path=self.hyper.pretraining_file, logger = logger)
            pretraining_tag = True

            if xfit_stats:
                logger.critical('>'*40)
                logger.critical('Evaluation on the training set.')
                acc,_,_ = self.loader_predict(train_loader)
                logger.critical('Overall Training Accuracy is: {:.2f}%'.format(acc * 100))

                logger.critical('>'*40)
                logger.critical('Evaluation on the validation set.')
                acc,_,_  = self.loader_predict(val_loader)
                logger.critical('Overall Validation Accuracy is: {:.2f}%'.format(acc * 100))

        if pretraining_tag is False or finetune:
            load, net_trainer = self.initialize_trainer(trainer_class, auto_pt_check, logger, **kwargs)
            if not load:
                fit_info = self._xfit(train_loader, val_loader, net_trainer)
        return fit_info

    def initialize_trainer(self, trainer_class, auto_pt_check = True, logger = None, **kwargs):
        if trainer_class is None:
            trainer_module = importlib.import_module(self.hyper.trainer_module[0])
            trainer_class = getattr(trainer_module, self.hyper.trainer_module[1])

        if not callable(trainer_class):
            raise TypeError(f'trainer_class must be callable, got {type(trainer_class).__name__}')

        net_trainer = trainer_class(self.hyper, logger, **kwargs)

        load = False
        if auto_pt_check and net_trainer.finish:
            self.load_pretraing_file(file_path=net_trainer.best_file, tag='checkpoint', logger=logger)
            load = True
        return load, net_trainer

    def _xfit(self, train_loader, val_loader, net_trainer):
        net_trainer.loop(self, train_loader, val_loader)
        self.load_pretraing_file(file_path=net_trainer.best_file, tag='checkpoint')
        # epochs_stats = pd.DataFrame(
        #     data={"epoch": range(self.iter),
        #           "lr_list": self.lr_list,
        #           "train_loss": self.train_loss_list,
        #           "val_loss": self.val_loss_list,
        #           "train_acc": self.train_acc_list,
        #           "val_acc": self.val_acc_list}
        # )
        fit_info = net_trainer.epochs_stats
        return fit_info

    def loader_predict(self, data_loader):
        self.eval()
        pre_lab_all = []
        label_all = []
        with tqdm(total=len(data_loader), desc='Batches', mininterval=0.3, colour='blue', leave=False) as bbar:
            for data_batch in data_loader:
                # if hasattr(self.hyper, 'using_snr') and self.hyper.using_snr
                # there will be three elements in data_batch, which are sig_batch, lab_batch, and snr_batch
                sig_batch, lab_batch = data_batch[0], data_batch[1]
                pre_lab = self.predict(sig_batch)
                pre_lab_all.append(pre_lab)
                label_all.append(lab_batch)
                bbar.update(1)

        pre_lab_all = torch.cat(pre_lab_all)
        label_all = torch.cat(label_all)
        acc = accuracy_score(label_all, pre_lab_all)
        return acc, pre_lab_all, label_all

    def predict(self, sample, return_cpu = True):
        """
        Return: prediction label of each sample as torch.tensor.
        """
        logits = self.logits(sample)
        pre_lab = torch.argmax(logits, 1)
        if return_cpu:
            pre_lab = pre_lab.cpu()
        return pre_lab


    def logits(self, sample):
        """
        Return: prediction logits of each sample as torch.tensor.
        """
        sample = sample.to(self.hyper.device)
        logits = self.forward(sample)
        return logits

    def feature_extract(self, x):
        return self.logits(x)


class BaseNetConfig(Opt):
    def __init__(self):
        super().__init__()

        self.arch = 'torch_nn'
        self.trainer_module = ('models/nn/_baseTrainer.py', 'AnnealingTrainer')

        self.hyper = hyper()
        # self.tuner = tuner()
        # self.tuning = tuning()

        self.base_modify()
        self.hyper_modify()
        self.tuning_modify()

        self.task_modify()
        self.ablation_modify()

        self.common_process()

    def common_process(self,):
        if "import_path" in self.dict:
            self.import_path = self.import_path.replace(
            '.py', '').replace('/', '.')
            self.hyper.import_path = self.import_path
        if "trainer_module" in self.dict:
            self.trainer_module = (self.trainer_module[0].replace(
                    '.py', '').replace('/', '.'), self.trainer_module[1])
            self.hyper.trainer_module = self.trainer_module

        self.hyper.class_name = self.class_name
        # if 'gpu' in self.tuner.resource:
        #     num_gpus = torch.cuda.device_count()
        #     trial_perGPU = int(1 // self.tuner.resource['gpu'])
        #     self.tuner.resource['cpu'] = self.tuner.num_cpus
        #     self.tuner.num_cpus = self.tuner.num_cpus * trial_perGPU * num_gpus


    def base_modify(self,):
        pass
    def hyper_modify(self,):
        pass
    def tuning_modify(self):
        pass
    def ablation_modify(self):
        pass
    def task_modify(self):
        pass



class hyper(Opt):
    def __init__(self):
        super().__init__()
        self.epochs = 400
        self.patience = 15
        self.milestone_step = 3
        self.gamma = 0.5
        self.lr = 0.001
        self.pretraining_file = ''
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

# class tuner(Opt):
#     def __init__(self):
#         super().__init__()
#         self.resource = {
#             "gpu": 1  # set this for GPUs
#         } # Parallel nums = min(num_cpus // cpu, num_gpus // gpu), where num_gpus = torch.cuda.device_count(), which means this setting only affects to the worker scheduler of the ray tuner, and the cpus settings does not affect the system resources the runing worker utilizes.

#         self.num_samples = 20 # tuner num trails
#         self.max_training_iteration = 100 # max fitness epochs per trail
#         self.min_training_iteration = 20
#         self.algo = 'tpe'
#         self.num_cpus = os.cpu_count()

# class tuning(Opt):
#     def __init__(self):
#         super().__init__()
#         self.lr = tune.loguniform(1e-4, 1e-2)
#         self.gamma = tune.uniform(0.33,0.99)
#         self.milestone_step = tune.qrandint(1,10,1)
#         self.batch_size = tune.choice([64, 96, 128, 160, 192])