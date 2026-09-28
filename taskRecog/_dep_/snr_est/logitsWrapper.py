import os
import sys

import torch
import numpy as np
from collections import Counter
from tqdm.auto import tqdm, trange
from taskRecog.util import Opt, set_dataloader
from taskRecog.Wrapper import Task
from data.Loader import TaskDataset
from taskRecog.Parser import get_parser

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from taskRecog.util import os_makedirs, os_rmdirs, set_logger, fix_seed
from taskRecog.util import save_training_process, save_confmat, save_snr_acc
from copy import deepcopy


from taskRecog.featureselection.snr_aware.softmaskTuner import maskTuner

class Mask(Opt):


    def __init__(self, init=None):
        super().__init__(init)

        self.local_dir = ''
        self.name = ''


class SoftMaskTask(Task):
    """
    """

    def __init__(self, args):
        super().__init__(args)

    def tuning(self):
        raise ValueError('Non support to tuning!')


    def exp_config(self, args):
        self.masking_tag = False

        cuda_exist = torch.cuda.is_available()
        if cuda_exist and args.cuda:
            self.model_opts.hyper.device = torch.device('cuda:{}'.format(args.gid))
            torch.cuda.set_device(args.gid)
            self.device_id = args.gid
        else:
            self.model_opts.hyper.device = torch.device('cpu')
            self.device_id = -1

        if 'exp_dir' in vars(args) and args.exp_dir is not None:
            self.exp_dir = args.exp_dir
        else:
            self.exp_dir = 'yield_results' if args.test == False else 'yield_test'
            self.exp_dir = os.path.join(self.exp_dir, self.data_name )

        assert 'exp_name' in vars(args)
        self.exp_dir = os.path.join(self.exp_dir, args.exp_name)

        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model) if args.tag == '' else '{}_{}'.format(args.model, args.tag)

        self.sampler = Mask()
        self.sampler.local_dir = os.path.join(self.fit_dir, self.model_name, 'sampler')
        os_makedirs(self.sampler.local_dir)

        self.sampler.save_path = os.path.join(self.sampler.local_dir, f'snrs_step_mask.pt')

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name)
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        self.model_result_file = os.path.join(self.model_pred_dir, 'results.npz')

        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)

        if args.test and args.logger_level != 20:
            self.logger_level = 50  # equal to critical
        else:
            self.logger_level = 20  # equal to info

        self.force_update = args.force_update

    def snr_evoMask(self, snr = 0):
        # step_idxs = []

        tuner_dir = os.path.join(self.model_fit_dir, 'SNR_tuner', f'snr.{snr}')
        os_makedirs(tuner_dir)
        self.model_opts.tuner.dir = tuner_dir
        tLogger = self.logger_config(tuner_dir, 'tuning')

        snr_Tuner = maskTuner(self.model_opts, logger= tLogger, subPack= self.data_opts, snr = snr)

        step_idxs = snr_Tuner.conduct()

        return step_idxs


    def masking(self, logger):
        if os.path.exists(self.sampler.save_path):
        # if False: #toDo: this is set to re-save the sampler results.
            snrs_step_mask = torch.load(self.sampler.save_path)
            logger.info('-'*80)
            logger.info(f'Read the saved snrs_step_mask from {self.sampler.save_path} successfully!')
        else:
            logger.info('*'*80)
            logger.info(f'Process the sampling with {self.model_opts.tuner.algo} for every step on every snr.')
            with tqdm(total = len(self.data_opts.snr_envs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
                snrs_step_mask = []
                for step, snr in enumerate(self.data_opts.snr_envs):

                    step_idxs = self.snr_evoMask(snr)
                    snrs_step_mask.append(step_idxs)

                    pbar.update(1)

            os_makedirs(self.sampler.local_dir)
            torch.save(snrs_step_mask, self.sampler.save_path)
            logger.info(f'Save the snrs_step_mask to {self.sampler.save_path} !')


        self.sampling_tag = True

        # snrs_step_mask
        for snr,  step_idxs in enumerate(snrs_step_mask):
            logger.info(f'SNR: {self.data_opts.snr_envs[snr]}\t Mask step idxs: {step_idxs}')

        return snrs_step_mask


    def sampling_dataset(self, data_set, data_idx):
        Signals, Labels = data_set

        data_sampling_set = Signals.detach().clone()
        from taskRecog.util import snr_slice
        for snr, step_idxs in zip(self.data_opts.snr_envs, self.snrs_mask_idxs):
            sig_i, _, idx_i = snr_slice(data_set, data_idx, self.data_opts.SNR_values, snr)

            for step, value in enumerate(step_idxs):
                sig_i[:,:, step] = sig_i[:,:, step].detach().clone() * value

            data_sampling_set[idx_i] = sig_i


        return data_sampling_set, Labels


    def load_fitset(self, cid_hyper=None):
        batch_size = 64 if cid_hyper is None else cid_hyper.batch_size

        if 'snrs_mask_idxs' not in self.dict:
             raise ValueError('Non sampling stage! Please run sampling function first.')

        train_set = self.sampling_dataset(self.data_opts.train_set, self.data_opts.train_idx)
        val_set = self.sampling_dataset(self.data_opts.val_set, self.data_opts.val_idx)

        train_loader = set_dataloader(batch_size= batch_size, data_set=train_set)
        val_loader = set_dataloader(batch_size= batch_size, data_set=val_set)

        return train_loader, val_loader

    # def load_testset(self, cid_hyper=None):
    #     batch_size = 64 if cid_hyper is None else cid_hyper.batch_size
    #     test_set = self.sampling_dataset(self.data_opts.test_set, self.data_opts.test_idx)
    #     Sample_list = []
    #     Label_list = []

    #     for snr in self.data_opts.snr_envs:
    #         sig_i, lab_i, _ = self.data_opts.snr_slice('test', snr)

    #         num_chunk = int(sig_i.shape[0] / batch_size)

    #         Sample = torch.chunk(sig_i, num_chunk, dim=0)
    #         Label = torch.chunk(lab_i, num_chunk, dim=0)

    #         Sample_list.append(Sample)
    #         Label_list.append(Label)

    #     return Sample_list, Label_list

    def load_hyper(self, clogger):
        clogger.critical('*'*80)
        clogger.critical('Dataset: {}\t Model:{} \t Class: {}'.format(
            self.data_name, self.model_name, self.data_opts.num_classes))

        cid_hyper = Opt(self.model_opts.hyper)
        cid_hyper.num_classes = self.data_opts.num_classes
        cid_hyper.model_fit_dir = self.model_fit_dir
        cid_hyper.model_name = self.model_name
        cid_hyper.data_name = self.data_name

        # if self.sampling_tag:
        #     cid_hyper.sig_len = self.sampler.num_steps

        return cid_hyper

    def conduct(self, force_update=None, ):
        if force_update is not None:
            if force_update in [True, False]:
                self.force_update = force_update
            else:
                raise ValueError(
                    'force_update parameter is incorrect, please set with True or False.')

        if not os.path.exists(self.model_result_file) or self.force_update:
            if os.path.exists(self.model_result_file):
                os_rmdirs(self.model_pred_dir)
            os_makedirs(self.model_pred_dir)

            task_logger = self.logger_config(
                self.model_fit_dir, 'train')
            self.load_data(logger=task_logger)

            self.snrs_mask_idxs = self.masking(logger=task_logger)

            self.conduct_fit(clogger=task_logger,
                             result_file=self.model_result_file)

            self.fit_statue = True

            F1_score, kappa, acc = self.evaluate(elogger=task_logger) # toDo: add eval dir with sampler.num_steps tag
        else:
            F1_score, kappa, acc = self.evaluate()

        return F1_score, kappa, acc


    def snrLogits_predict(self, model, sample, return_cpu = True):
        sample_nums = sample.shape[0]
        snr_nums = len(self.data_opts.snr_envs)
        label_nums = len(self.data_opts.mods)

        logits_matrix = torch.empty(size=(sample_nums, snr_nums, label_nums)).to(model.hyper.device)

        for s, step_idxs in zip(list(range(snr_nums)), self.snrs_mask_idxs):
            _sample = sample.detach().clone()
            for step, value in enumerate(step_idxs):
                _sample[:,:, step] = _sample[:,:, step].detach().clone() * value

            logits_snr = model.logits(_sample)
            logits_matrix[:,s, :] = logits_snr

        max_logits, _= torch.max(logits_matrix, 1)
        pre_lab = torch.argmax(max_logits, 1)
        if return_cpu:
            pre_lab = pre_lab.cpu()
        return pre_lab



    def eval_testset(self, model, logger, result_file):
        if logger is not None:
            logger.critical('>'*40)
            logger.critical('Evaluation on the testing set.')

        with torch.no_grad():

            if self.model_opts.arch == 'torch_nn':
                model.eval()

            #to DO: for each snr, input the subsampling operator, output the logits.
            test_sample_list, test_lable_list = self.load_testset(model.hyper)

            pre_lab_all = []
            label_all = []

            for (Sample, Label) in tqdm(zip(test_sample_list, test_lable_list), total=len(test_sample_list)):
                pred_i = []
                label_i = []
                for (sample, label) in zip(Sample, Label):
                    pre_lab = self.snrLogits_predict(model, sample)
                    pred_i.append(pre_lab)
                    label_i.append(label)
                pred_i = np.concatenate(pred_i)
                label_i = np.concatenate(label_i)

                pre_lab_all.append(pred_i)
                label_all.append(label_i)

            tgt_result_file = result_file if result_file is not None else self.model_result_file # add to allow the external configuration

            np.savez(tgt_result_file, pred = pre_lab_all, label= label_all)
            logger.critical('Save result file to the location: {}'.format(tgt_result_file))
            logger.critical('-'*80)

        return pre_lab_all, label_all

    def evaluate(self, elogger = None, force_update=False, ave_confMax = False):
        eLogger = set_logger(os.path.join(self.eval_dir, '_log_', '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
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
                pre_lab_all, label_all = data['pred'], data['label']
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