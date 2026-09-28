from taskRecog.Wrapper import Task
import os
from taskRecog.util import os_makedirs, os_rmdirs, set_logger, fix_seed
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from taskRecog.util import save_training_process, save_confmat, save_snr_acc
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torch

class SNR_Task(Task):

    def eval_testset(self, model, logger, result_file):
        if logger is not None:
            logger.critical('>'*40)
            logger.critical('Evaluation on the testing set.')

        with torch.no_grad():
            # if self.model_opts.arch == 'torch_nn':
            model.eval()

            test_sample_list, test_lable_list = self.load_testset(model.hyper)

            pre_lab_all = []
            label_all = []

            # loop of mods in test_sample_list
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

            np.savez(tgt_result_file, pred = pre_lab_all, label= label_all)
            logger.critical('Save result file to the location: {}'.format(tgt_result_file))
            logger.critical('-'*80)

        return pre_lab_all, label_all

    def evaluate(self, elogger = None, ave_confMax = False, xfit_stats = False):
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

        if self.fit_statue and self.args.force_update is False:
            force_update = False
        else:
            force_update = True

        if os.path.exists(self.model_result_file) and force_update is False:
            with np.load(self.model_result_file) as data:
                pre_lab_all, label_all = data['pred'], data['label']
        else:
            pre_lab_all, label_all,_ = self.conduct_fit(xfit_stats = xfit_stats)


        # Confmat_Set = np.zeros((len(self.data_opts.num_snrs), self.data_opts.num_classes, self.data_opts.num_classes), dtype=int)
        Accuracy_list = np.zeros(len(self.data_opts.mods), dtype=float)

        mod_list = []
        for mod_i, (pred_i, label_i) in enumerate(zip(pre_lab_all, label_all)):
            # cm_i =  confusion_matrix(label_i, pred_i)
            # Confmat_Set[mod_i, :, :] = cm_i
            Accuracy_list[mod_i] = accuracy_score(label_i, pred_i)
            mod_list.append(self.data_opts.mods_dict[mod_i])

        pre_lab_all = np.concatenate(pre_lab_all)
        label_all = np.concatenate(label_all)

        F1_score = f1_score(label_all, pre_lab_all, average='macro')
        kappa = cohen_kappa_score(label_all, pre_lab_all)
        acc = np.mean(Accuracy_list)

        eLogger.info('Overall Accuracy is: {:.2f}%'.format(acc * 100))
        eLogger.info(f'Macro F1-score is: {F1_score:.4f}')
        eLogger.info(f'Kappa Coefficient is: {kappa:.4f}')

        plt.figure(figsize=(12,6))
        plt.plot(mod_list, Accuracy_list)
        plt.xlabel("Modulation Class")
        plt.ylabel("Overall Accuracy")
        plt.title(f"Overall Accuracy on {self.data_name} dataset")
        plt.yticks(np.linspace(0, 1, 11))
        plt.grid(True, linestyle='--')
        acc_dir = os.path.join(self.eval_plot_dir, 'acc')
        os.makedirs(acc_dir, exist_ok=True)
        plt.savefig(acc_dir + '/' + 'acc.png', dpi=300)
        plt.close()

        # if ave_confMax:
        #     save_confmat(Confmat_Set, self.data_opts.num_snrs, self.data_opts.classes, self.eval_plot_dir)


        # Accuracy_Mods = save_snr_acc(Accuracy_list, Confmat_Set, self.data_opts.num_snrs, self.data_name, self.data_opts.classes.keys(), self.eval_plot_dir)

        # tgt_acc_file = os.path.join(self.eval_acc_dir, 'acc.npz')
        # np.savez(tgt_acc_file, acc_overall = Accuracy_list, acc_mods= Accuracy_Mods)
        # eLogger.info('Save accuracy file to the location: {}'.format(tgt_acc_file))

        return F1_score, kappa, acc