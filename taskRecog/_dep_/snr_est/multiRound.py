
from taskRecog.util import Opt
from copy import deepcopy

import importlib

from taskRecog.softmask.snr_est.SNRpredWrapper import SnrPredMaskTask
# from taskRecog.softmask.logitsWrapper import SoftMaskTask
import os
from taskRecog.util import os_makedirs, os_rmdirs, set_logger
import torch

class MutliRoundMask(Opt):
    def __init__(self, args):
        self.args = args
        self.exp_module = importlib.import_module('{}.{}'.format(
            args.exp_config.replace('/', '.'), args.exp_file))

        self.data_config(args)
        self.exp_name = args.exp_name
        self.model_name = args.model

        if 'exp_dir' in vars(args) and args.exp_dir is not None:
            self.exp_dir = args.exp_dir
        else:
            assert 'exp_dataname_first' in vars(args)
            self.exp_dir = 'yield_results' if args.test == False else 'yield_test'
            self.exp_dir = os.path.join(self.exp_dir, self.data_name, args.exp_name) if args.exp_dataname_first else os.path.join(self.exp_dir, args.exp_name, self.data_name)
            self.exp_name = args.exp_name


    def snrmodel_config(self, args):
        model_config = args.snrmodel.config
        model = importlib.import_module(model_config.import_path)
        model = getattr(model, model_config.class_name)
        model_config.hyper.num_classes = args.num_env
        model_config.hyper.merge(self.data_opts, ['sig_len'])
        if torch.cuda.is_available() and args.cuda:
            model_config.hyper.device = torch.device('cuda:{}'.format(args.gid))
        else:
            model_config.hyper.device = torch.device('cpu')

        model = model(model_config.hyper, self.logger)
        model_pt_file = os.path.join('yield_results/SNR_Estimation', f'C{args.num_env}', self.data_name ,'fit', f'{args.snrmodel.name}', f'checkpoint/{self.data_name}_{args.snrmodel.name}.best.pt')
        model.load_pretraing_file(file_path =  model_pt_file)
        return model


    def data_config(self, args):
        # data_opts = getattr(self.exp_module, args.dataset + '_data')
        data_opts = getattr(self.exp_module, 'Data')
        self.data_opts = data_opts(args)
        self.data_name = self.data_opts.data_name

    def logger_config(self, dir, stage):
        log_path = os.path.join(dir, 'logs',
                                '{}.{}.log'.format(stage,self.data_name))
        log_name = '{}.{}.{}'.format(
            self.exp_name,self.data_name, self.model_name)
        logger = set_logger(log_path, log_name, 20)
        return logger

    def conduct(self,):
        self.args.force_update = False

        self.logger = self.logger_config(self.exp_dir, f'{self.args.exp_name}.Mask.{self.model_name}')
        snrmodel = self.snrmodel_config(self.args)
        snrmodel.eval()

        for i_iter in range(1, self.args.rounds+1):
            iter_args = deepcopy(self.args)
            iter_args.exp_name = self.args.exp_name+f'.iter{i_iter}'
            iter_args.exp_dir = os.path.join(self.exp_dir, f'iter{i_iter}')

            iter_hyper = Opt()
            if i_iter > 1:
                iter_hyper.pretraining_allDatafile = os.path.join(self.exp_dir, f'iter{i_iter -1}/fit/{self.model_name}/checkpoint/{self.data_name}_{self.model_name}.best.pt')

            # add to check!!!!!!!
            iter_hyper.pretraining_file = os.path.join(self.exp_dir, f'iter{i_iter}/fit/{self.model_name}/checkpoint/{self.data_name}_{self.model_name}.best.pt')
            iter_hyper.sampler_file = os.path.join(self.exp_dir, f'iter{i_iter}/fit/{self.model_name}/sampler/snrs_step_mask.pt')
            # SnrPredTest.AMC/env2/RML2016.10a/SnrPredTest.AMC/env2.iter1/fit/awn/checkpoint/RML2016.10a_awn.best.pt
            iter_args.hyper = iter_hyper


            task = SnrPredMaskTask(iter_args, snrmodel)
            ori_acc, predmask_acc, refmask_acc= taskRecog.conduct()

            self.logger.info('Iter: {}\tOri. OA: {:.2f}%\tPred. Mask OA: {:.2f}%\tRef. Mask OA: {:.2f}%'.format(i_iter, ori_acc * 100, predmask_acc * 100, refmask_acc * 100))


    def evaluate(self,):
        self.args.force_update = False

        self.logger = self.logger_config(self.exp_dir, f'{self.args.exp_name}.Mask.{self.model_name}')
        snrmodel = self.snrmodel_config(self.args)
        snrmodel.eval()

        for i_iter in range(1, self.args.rounds+1):
            iter_args = deepcopy(self.args)
            iter_args.exp_name = self.args.exp_name+f'.iter{i_iter}'
            iter_args.exp_dir = os.path.join(self.exp_dir, f'iter{i_iter}')

            iter_hyper = Opt()
            if i_iter > 1:
                iter_hyper.pretraining_allDatafile = os.path.join(self.exp_dir, f'iter{i_iter -1}/fit/{self.model_name}/checkpoint/{self.data_name}_{self.model_name}.best.pt')

            # add to check!!!!!!!
            iter_hyper.pretraining_file = os.path.join(self.exp_dir, f'iter{i_iter}/fit/{self.model_name}/checkpoint/{self.data_name}_{self.model_name}.best.pt')
            iter_hyper.sampler_file = os.path.join(self.exp_dir, f'iter{i_iter}/fit/{self.model_name}/sampler/snrs_step_mask.pt')
            # SnrPredTest.AMC/env2/RML2016.10a/SnrPredTest.AMC/env2.iter1/fit/awn/checkpoint/RML2016.10a_awn.best.pt
            iter_args.hyper = iter_hyper


            task = SnrPredMaskTask(iter_args, snrmodel)
            ori_acc, predmask_acc, refmask_acc= taskRecog.evaluate()

            self.logger.info('Iter: {}\tOri. OA: {:.2f}%\tPred. Mask OA: {:.2f}%\tRef. Mask OA: {:.2f}%'.format(i_iter, ori_acc * 100, predmask_acc * 100, refmask_acc * 100))