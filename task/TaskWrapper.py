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
from task.TaskVisual import save_training_process

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
        # data_opts = getattr(self.exp_module_path, args.dataset + '_data')
        data_opts = getattr(self.exp_module_path, 'Data')
        self.data_opts = data_opts(args)
        self.data_name = self.data_opts.data_name
        
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

        self.exp_dir = 'exp_results' if args.test == False else 'exp_tempTest'

        if args.exp_name is not None:
            self.exp_dir = os.path.join(self.exp_dir, args.exp_name)

        self.exp_dir = os.path.join(self.exp_dir, args.dataset)
        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model) if args.tag == '' else '{}_{}'.format(args.model, args.tag)

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name)
        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        
        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)
            
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

        if fit:
            self.model_opts.hyper.device = self.device
            # self.tune = args.tune  # default False
            
    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        return model        
    
    def logger_config(self, dir, stage, cv):
        log_path = os.path.join(dir, 'logs',
                                '{}.{}.cv{}.log'.format(stage,self.data_name, cv))
        log_name = '{}.{}.cv{}'.format(
            self.data_name, self.model_name, cv)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger
    
    def conduct(self,):
        self.pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        
        # os_makedirs(self.model_fit_dir)
        os_makedirs(self.pred_dir)
        
        for i in tqdm(self.cid_list):
            result_file = os.path.join(self.pred_dir, 'results_{}.npy'.format(i))
            if os.path.exists(result_file):
                    continue
            
            self.conduct_iter(self, i , result_file, innerSaving=True)
    
    def conduct_iter(self, i, result_file, innerSaving = True):
        """
        docstring
        """
        try:
            clogger = self.logger_config(
                self.model_fit_dir, 'train', i)
            clogger.critical('*'*80)
            clogger.critical('Dataset: {}\t Model:{} \t Class: {}\t Trail: {}'.format(
                self.data_name, self.model_name, self.data_opts.num_classes, i))
            
            cid_hyper = Opt(self.model_opts.hyper)
            cid_hyper.num_classes = self.data_opts.num_classes
            cid_hyper.model_fit_dir = self.model_fit_dir
            cid_hyper.model_name = self.model_name
            cid_hyper.data_name = self.data_name     
            cid_hyper.cid = i
            
            #toDo: set Tune, and loading the best parameters
            # if self.tune:
                # best_hyper = self.load_tuning()
                # cid_hyper.update(best_hyper)
                # clogger.info("Updating tuning result complete.")
                # clogger.critical('-'*80)

            clogger.critical(
                'For {}th-batch-trainingLoading, loading the datasets {}'.format(i, self.data_name))
            clogger.critical('-'*80)            
            
            model = self.model_import()
            model = model(cid_hyper, clogger)

            clogger.critical('Loading complete.')
            clogger.critical(f'Model: \n{str(model)}')
            
            train_loader, val_loader = self.data_opts.load_fitset()
            epochs_stats =  model.xfit(train_loader, val_loader) 
            # epochs_stats is a dataframe with the following columns at least
            # pd.DataFrame(
            # data={"lr_list": self.lr_list,
            #       "train_loss": self.train_loss_list,
            #       "val_loss": self.val_loss_list,
            #       "train_acc": self.train_acc_list,
            #       "val_acc": self.val_acc_list})
            
            loss_dir = os.path.join(self.model_fit_dir, 'loss_curve')
            lossfig_dir = os.path.join(loss_dir, 'figure')
            
            if set(['val_loss','val_acc', 'train_loss', 'train_acc', 'lr_list']).issubset(epochs_stats.columns):
                save_training_process(epochs_stats, plot_dir=lossfig_dir)
            
            
            
            
            
            
            
            
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()
