import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from task.base.TaskWrapper import Task
from task.base.TaskLoader import Opt
from task.TaskParser import get_parser
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from task.util import os_makedirs, os_rmdirs, set_logger, fix_seed
from task.base.TaskVisual import save_training_process, save_confmat, save_snr_acc
from task.maskcut.maskTuner import MaskTuner
from task.maskcut.maskLoader import mask_count

class maskcutTask(Task):
    def __init__(self, args):
        # self.opts = Opt()
        # self.opts.merge(args)
        super().__init__(args)
        
        
    
    def init_inputMask(self, cid_hyper = None):
        for i in range(self.data_opts.info.sig_len):
            if 'inputMask_{}'.format(i) not in cid_hyper.dict:
                raise ValueError('Missing hyper config: "inputMask_{}"!'.format(i))
            
        inputMask = [cid_hyper.dict['inputMask_{}'.format(i)] for i in range(self.data_opts.info.sig_len)]
        
        return inputMask
        
        
    def load_fitset(self, cid_hyper = None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size
        
        self.inputMask = self.init_inputMask(cid_hyper)
        
        train_loader, val_loader = self.data_opts.load_fitset(fit_batch_size = batch_size, mask_opt = self.inputMask)
        return train_loader, val_loader

    def load_testset(self, cid_hyper = None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size
        
        self.inputMask = self.init_inputMask(cid_hyper)
        
        test_sample_list, test_lable_list = self.data_opts.load_testset(test_batch_size = batch_size, mask_opt = self.inputMask)
        return test_sample_list, test_lable_list

    def load_hyper(self, clogger):
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
        if self.tune:
            if 'best_hyper' not in self.dict:
                self.tuning()
                
            best_hyper = self.best_hyper
            cid_hyper.update(best_hyper)
            clogger.info("Updating tuning result complete.")
            clogger.critical('-'*80)
            if self.best_checkpoint_path is not None:
                cid_hyper.pretraining_file = self.best_checkpoint_path
        
        
        self.inputMask = self.init_inputMask(cid_hyper)
        cid_hyper.mask_num = mask_count(self.inputMask)
        
        cid_hyper.sig_len = self.data_opts.info.sig_len - cid_hyper.mask_num
        cid_hyper.mask_ratio = cid_hyper.mask_num / self.data_opts.info.sig_len
        
        clogger.critical("Loading mask operator.")
        clogger.critical('Overall Mask-ratio is: {:.2f}% \t Mask-num is: {} \t Remaining sig_len is: {}'.format(cid_hyper.mask_ratio * 100, cid_hyper.mask_num, cid_hyper.sig_len))
        clogger.critical('-'*80)
        
        
        return cid_hyper

    def load_tuner(self, logger):
        series_tuner = MaskTuner(self.model_opts, logger, self.data_opts)
        return series_tuner


    def eval_testset(self, model, logger, result_file):
        if logger is not None:
            logger.critical('>'*40)
            logger.critical('Evaluation on the testing set.')
            
        with torch.no_grad():
            if self.model_opts.arch == 'torch_nn':
                model.eval()
                       
            test_sample_list, test_lable_list = self.load_testset(model.hyper)
            pre_lab_all = []
            label_all = []
            for (Sample, Label) in tqdm(zip(test_sample_list, test_lable_list), total=len(test_sample_list)):
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
            
            tgt_result_file = result_file if result_file is not None else self.model_result_file # add to allow the external configuration
            
            np.savez(tgt_result_file, pred = pre_lab_all, label= label_all, mask_opt = self.inputMask)
            logger.critical('Save result file to the location: {}'.format(tgt_result_file))
            logger.critical('-'*80)   
            
            # logger.info('Overall Mask-ratio is: {:.2f}% \t Mask-num is: {}'.format(model.hyper.mask_ratio * 100, model.hyper.mask_num))
            
        return pre_lab_all, label_all
    

    def evaluate(self, elogger = None, force_update=False, ave_confMax = False):
        eLogger = set_logger(os.path.join(self.eval_dir, '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
                self.data_name, self.model_name.upper()), self.logger_level) if elogger is None else elogger
        
        if self.data_statue is False:
            self.load_data(logger=eLogger)
        
        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        self.eval_acc_dir = os.path.join(self.model_eval_dir, 'accuracy')
        self.eval_plot_dir = os.path.join(self.model_eval_dir, 'figures')
        
        os_makedirs(self.eval_acc_dir)
        os_makedirs(self.eval_plot_dir)
        os_makedirs(self.model_pred_dir)
        
        # for i in self.cid_list: # multiple cross validation in the future version
        
        if self.fit_statue:
            force_update = False
        
        if os.path.exists(self.model_result_file) and force_update is False:
            with np.load(self.model_result_file) as data:
                pre_lab_all, label_all, self.inputMask = data['pred'], data['label'], data['mask_opt']
        else:
            pre_lab_all, label_all,_ = self.conduct_fit()
        
        
        Confmat_Set = np.zeros((len(self.data_opts.num_snrs), self.data_opts.num_classes, self.data_opts.num_classes), dtype=int)
        Accuracy_list = np.zeros(len(self.data_opts.num_snrs), dtype=float)
        
        for snr_i, (pred_i, label_i) in enumerate(zip(pre_lab_all, label_all)):
            cm_i =  confusion_matrix(label_i, pred_i)
            Confmat_Set[snr_i, :, :] = cm_i
            Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)
    
        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)
        
        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)
        
        mask_num = mask_count(self.inputMask)
        mask_ratio =  mask_num / self.data_opts.info.sig_len 
        eLogger.info('Overall Mask-ratio is: {:.2f}% \t Mask-num is: {} \t Remaining sig_len is: {}'.format(mask_ratio * 100, mask_num, self.data_opts.info.sig_len - mask_num))
        eLogger.info('Overall Accuracy is: {:.2f}%'.format(acc * 100))
        eLogger.info(f'Macro F1-score is: {F1_score:.4f}')
        eLogger.info(f'Kappa Coefficient is: {kappa:.4f}')
        
        if ave_confMax:
            save_confmat(Confmat_Set, self.data_opts.num_snrs, self.data_opts.classes, self.eval_plot_dir)
        
        
        Accuracy_Mods = save_snr_acc(Accuracy_list, Confmat_Set, self.data_opts.num_snrs, self.data_name, self.data_opts.classes.keys(), self.eval_plot_dir)
        
        tgt_acc_file = os.path.join(self.eval_acc_dir, 'acc.npz')
        np.savez(tgt_acc_file, acc_overall = Accuracy_list, acc_mods= Accuracy_Mods)
        eLogger.info('Save accuracy file to the location: {}'.format(tgt_acc_file))
        
        return F1_score, kappa, acc    

if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    
    args.exp_config = 'exp_config/maskcut'
    args.exp_file= '16a'
    args.force_update = True
    args.gid = 0
    
    args.test = True
    args.clean = True
    args.exp_name = 'sub.demo'
    args.model = 'awn'
    
    
    task = maskcutTask(args)
    task.tuning()
    task.conduct()
    # task.evaluate(force_update=True)     