import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from tqdm.std import tqdm
import importlib

from task.TaskLoader import Opt

import torch
import numpy as np
import statistics

from tqdm import trange
import shutil
from task.util import os_makedirs, os_rmdirs, set_logger


class Task(Opt):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)

        self.exp_module_path = importlib.import_module('{}.{}'.format(
            args.datafolder.replace('/', '.'), args.dataset))
        
        self.data_config(args)
        self.model_config(args)
        self.exp_config(args)
        
    def data_config(self, args):
        self.data_name = args.dataset
        # data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        data_opts = getattr(self.exp_module_path, 'Data')
        self.data_opts = data_opts(args)
        
    def model_config(self, args):
        self.model_name = args.model

        # load the specifical config firstly, if not exists, load the common config
        if hasattr(self.exp_module_path, self.model_name):
            model_opts = getattr(self.exp_module_path,
                                 args.model)
        else:
            try:
                share_module_path = importlib.import_module('data.base')
                model_opts = getattr(
                    share_module_path, self.model_name + '_default')
            except:
                raise ValueError(
                    'Non-supported model "{}" in the "{}" module, please check the module or the model name'.format(self.model_name, self.exp_module_path))

        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        
        if 'hyper' in vars(args):
            self.model_opts.hyper.update(args.hyper)


    def exp_config(self, args, fit = True):
        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.device = torch.device('cuda:{}'.format(args.gid))
        else:
            self.device = torch.device('cpu')

        if fit and self.model_opts.arch == 'statistic':
            self.device = torch.device('cpu')

        self.exp_dir = 'trial' if args.test == False else 'test'

        if args.exp_name is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.exp_name)

        self.exp_dir = os.path.join(self.exp_dir, args.dataset)
        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model) if args.tag == '' else '{}_{}'.format(args.model, args.tag)

        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        self.rep_times = args.rep_times

        self.cid_list = args.cid

        if self.cid_list == ['all']:
            self.cid_list = list(range(self.rep_times))
        else:
            _temp = [int(c) for c in self.cid_list]
            self.cid_list = _temp

        self.model_fit_dir_list = []
        self.model_eval_dir_list = []
        
        for sid in range(len(self.data_opts.seriesPack)):
            # toDo: change sid to snr tag.
            _H = self.data_opts.seriesPack[sid].H
            _task_dir = os.path.join(self.fit_dir, 'series{}'.format(sid), 'h{}'.format(_H), self.model_name)
            _eval_dir = os.path.join(
                    self.eval_dir, 'series{}'.format(sid), 'h{}'.format(_H), self.model_name)
            if args.test and args.clean:
                os_rmdirs(_task_dir)
            # os_makedirs(_task_dir)
            self.model_fit_dir_list.append(_task_dir)
            self.model_eval_dir_list.append(_eval_dir)
        

        if fit:
            self.model_opts.hyper.device = self.device
            # self.tune = args.tune  # default False
            
    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        return model        
    
    def logger_config(self, dir, stage, cv, sub_count):
        log_path = os.path.join(dir, 'logs',
                                '{}.cv{}.series{}.log'.format(stage, cv, sub_count))
        log_name = '{}.series{}.cv{}.{}'.format(
            self.data_name, sub_count, cv, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger
    
    def conduct(self,):
        
        pass