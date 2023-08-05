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
from task.TaskVisual import save_training_process, save_confmat, save_snr_acc
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score

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
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        self.model_result_file = os.path.join(self.model_pred_dir, 'results.npz')
        
        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)
            
        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        # self.rep_times = args.rep_times

        # self.cid_list = args.cid

        # if self.cid_list == ['all']:
        #     self.cid_list = list(range(self.rep_times))
        # else:
        #     _temp = [int(c) for c in self.cid_list]
        #     self.cid_list = _temp

        if fit:
            self.model_opts.hyper.device = self.device
            # self.tune = args.tune  # default False
            
    def model_import(self,):
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        return model        
    
    def logger_config(self, dir, stage):
        log_path = os.path.join(dir, 'logs',
                                '{}.{}.log'.format(stage,self.data_name))
        log_name = '{}.{}'.format(
            self.data_name, self.model_name)
        logger = set_logger(log_path, log_name, self.logger_level)
        return logger
    
    def conduct(self, force_update = False):
        # os_makedirs(self.model_fit_dir)
        os_makedirs(self.model_pred_dir)
        # for i in tqdm(self.cid_list):
        if not os.path.exists(self.model_result_file) or force_update:     
            self.conduct_fit(self,self.model_result_file, innerSaving=True)

    def conduct_fit(self, result_file = None, innerSaving = True):
        """
        docstring
        """
        try:
            clogger = self.logger_config(
                self.model_fit_dir, 'train')
            clogger.critical('*'*80)
            clogger.critical('Dataset: {}\t Model:{} \t Class: {}'.format(
                self.data_name, self.model_name, self.data_opts.num_classes))
            
            cid_hyper = Opt(self.model_opts.hyper)
            cid_hyper.num_classes = self.data_opts.num_classes
            cid_hyper.model_fit_dir = self.model_fit_dir
            cid_hyper.model_name = self.model_name
            cid_hyper.data_name = self.data_name     
            # cid_hyper.cid = i
            
            #toDo: set Tune, and loading the best parameters
            # if self.tune:
                # best_hyper = self.load_tuning()
                # cid_hyper.update(best_hyper)
                # clogger.info("Updating tuning result complete.")
                # clogger.critical('-'*80)

            clogger.critical(
                'Loading the datasets {}'.format(self.data_name))
            clogger.critical('-'*80)            
            clogger.critical(f"Signal_train.shape: {list(self.data_opts.train_set[0].shape)}", )
            clogger.critical(f"Signal_val.shape: {list(self.val_set.train_set[0].shape)}")
            clogger.critical(f"Signal_test.shape: {list(self.val_set.test_set[0].shape)}")
            clogger.critical('-'*80)  
            
            
            model = self.model_import()
            model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer

            clogger.critical('Loading Model.')
            clogger.critical(f'Model: \n{str(model)}')
            
            train_loader, val_loader = self.data_opts.load_fitset()
            
            clogger.critical('Loading training set and validation set.')
            clogger.info(f"Train_loader batch: {len(train_loader)}")
            clogger.info(f"Val_loader batch: {len(val_loader)}")
            clogger.critical('>'*40)
            clogger.critical('Start fit.')
            
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
            
            # generate the output of the testset

            pre_lab_all, label_all = self.eval_testset(model)
            
            tgt_result_file = result_file if result_file is not None else self.model_result_file # add to allow the external configuration
            
            clogger.critical('>'*40)
            clogger.critical('End fit.')
            if innerSaving:
                np.savez(tgt_result_file, pred = pre_lab_all, label= label_all)
                clogger.critical('Save result file to the location: {}'.format(tgt_result_file))
            clogger.critical('-'*80)   
        
            return pre_lab_all, label_all, epochs_stats
            
            
        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_testset(self, model):
        with torch.no_grad():
            
            test_sample_list, test_lable_list = self.data_opts.load_testset()

            pre_lab_all = []
            label_all = []
            
            for (Sample, Label) in zip(test_sample_list, test_lable_list):
                pred_i = []
                label_i = []
                for (sample, label) in zip(Sample, Label):
                    pre_lab = model.predict(sample)
                    pred_i.append(pre_lab)
                    label_i.append(label)
                pred_i = np.concatenate(pred_i)
                label_i = np.concatenate(label_i)
                
                pre_lab_all.append(pred_i)
                label_all.append(label_i)
                
        return pre_lab_all, label_all

    def evaluation(self, force_update=True):
        eLogger = set_logger(os.path.join(self.eval_dir, '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
                self.data_name, self.model_name.upper()), self.logger_level)
        
        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        self.eval_acc_dir = os.path.join(self.model_eval_dir, 'accuracy')
        self.eval_plot_dir = os.path.join(self.model_eval_dir, 'figures')
        
        os_makedirs(self.eval_acc_dir)
        os_makedirs(self.eval_plot_dir)
        os_makedirs(self.model_pred_dir)
        
        # for i in self.cid_list: # multiple cross validation in the future version
        
        
        if os.path.exists(self.model_result_file) and force_update is False:
            with np.load(self.model_result_file) as data:
                pre_lab_all, label_all = data['pred'], data['label']
        else:
            pre_lab_all, label_all,_ = self.conduct_fit()
            
        Confmat_Set = np.zeros((len(self.data_opts.num_snrs), len(self.data_opts.num_classes), len(len(self.data_opts.num_classes))), dtype=int)
        Accuracy_list = np.zeros(len(self.data_opts.num_snrs), dtype=float)
        
        for snr_i, (pred_i, label_i) in enumerate(zip(pre_lab_all, label_all)):
            Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i)
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)
            
        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)
        
        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)
        
        eLogger.info(f'overall accuracy is: {acc}')
        eLogger.info(f'macro F1-score is: {F1_score}')
        eLogger.info(f'kappa coefficient is: {kappa}')
            
        save_confmat(Confmat_Set, self.data_opts.num_snrs, self.data_opts.num_classes, self.eval_plot_dir)
        
        
        Accuracy_Mods = save_snr_acc(Accuracy_list, Confmat_Set, self.data_opts.num_snrs, self.data_name, self.data_opts.classes.keys(), self.eval_plot_dir)
        
        tgt_acc_file = os.path.join(self.eval_acc_dir, 'acc.npz')
        np.savez(tgt_acc_file, acc_overall = Accuracy_list, acc_mods= Accuracy_Mods)
        eLogger.info('Save accuracy file to the location: {}'.format(tgt_acc_file))