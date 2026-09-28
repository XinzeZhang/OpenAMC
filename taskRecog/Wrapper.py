import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

# from tqdm.std import tqdm
import importlib

from taskRecog.util import Opt
from taskRecog.Parser import get_parser

import torch
import numpy as np
import statistics

from tqdm.auto import tqdm
import shutil
from taskRecog.util import os_makedirs, os_rmdirs, set_logger, fix_seed
from taskRecog.util import save_training_process, save_confmat, save_snr_acc
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from collections.abc import Mapping

from models.nn._baseTrainer import AnnealingTrainer
# from taskRecog.Tuner import HyperTuner
from data import load_data_class

class Task(Opt):
    def __init__(self, args):
        # super().__init__(args)
        self.args = args
        self.seed = args.seed
        fix_seed(args.seed)

        # if 'model_modules' in vars(args):

        if 'model_init_path' not in vars(args) or args.model_init_path == '':
            model_init_path = '{}.{}'.format(args.expconfig_dir, args.data).replace('/', '.')
        else:
            model_init_path = args.model_init_path.replace('.py', '').replace('/', '.')
        self.model_init_path = importlib.import_module(model_init_path)

        self.data_config(args)
        self.exp_config(args)
        # self.model_init(args)
        self.data_statue = False    #status indicating whether the data has been loaded
        # self.fit_statue = False
        # self.tune = False if 'tune' not in vars(args) else args.tune

    def data_config(self, args):
        '''
        Create an instance of the data class (defined in data.RML) and assign it to self.data_opts
        '''
        # data_opts = getattr(self.exp_module, args.dataset + '_data')
        data_opts = load_data_class(args.data)
        self.data_opts = data_opts(args)
        self.data_name = self.data_opts.data_name

    def exp_config(self, args):
        '''
        create experiment directory and set up some experiment configurations
        '''
        self.exp_name = args.exp_name # default is the name of the exp_config file, can be set by args.exp_name
        if 'exp_dir' in vars(args) and args.exp_dir is not None:
            self.exp_dir = args.exp_dir
        else:
            assert 'exp_dataname_first' in vars(args)
            self.exp_dir = 'yield_results' if args.test == False else 'yield_test'
            self.exp_dir = os.path.join(self.exp_dir, self.data_name, self.exp_name) if args.exp_dataname_first else os.path.join(self.exp_dir, self.exp_name, self.data_name)

        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model)
        # if args.tag == '' else '{}_{}'.format(args.model, args.tag)

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name)
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        self.model_result_file = os.path.join(self.model_pred_dir, 'results.npz')

        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)

        self.force_update = args.force_update

        # self.tune = args.tune  # default False

    def logger_config(self, dir, stage, name = None):
        '''
        get a logger created in  util.py -> set_logger()
        '''
        log_path = os.path.join(dir, '{}.{}.log'.format(stage,self.data_name))
        log_name = '{}.{}.{}'.format(
            self.exp_name,self.data_name, self.model_name) if name is None else name
        logger = set_logger(log_path, log_name)
        return logger

    def load_data(self, logger= None):
        """
        load and split data, assign it to self.data_opts
        """
        if logger is None:
            logger = self.logger_config(self.model_fit_dir, 'data')
        logger.critical(
            'Loading the datasets {}'.format(self.data_name))
        self.data_opts.info.seed = self.seed

        # Split the preprocessed data, and pack them to self.train_set, self.val_set, self.test_set, self.test_idx.
        # See taskRecog.Loader for more details.
        self.data_opts.pack_dataset(logger = logger)

        self.data_opts.num_snrs = list(np.unique(self.data_opts.snr_envs))
        # logger.critical('-'*80)
        Dataset_size = self.data_opts.train_set[0].shape[0] + self.data_opts.val_set[0].shape[0] + self.data_opts.test_set[0].shape[0]
        logger.critical('-'*80)
        logger.critical(f'Dataset size: {int(Dataset_size)}\t Class num: {self.data_opts.num_classes}')
        logger.critical(f"Signal_train.shape: {list(self.data_opts.train_set[0].shape)}", )
        logger.critical(f"Signal_val.shape: {list(self.data_opts.val_set[0].shape)}")
        logger.critical(f"Signal_test.shape: {list(self.data_opts.test_set[0].shape)}")
        self.data_statue = True  #already load data

    def load_fitset(self, batch_size = None):
        _batch_size = 64 if batch_size is None else batch_size
        train_loader, val_loader = self.data_opts.load_fitset(fit_batch_size = _batch_size)
        return train_loader, val_loader

    def conduct(self, force_update = None, xfit_stats = True, ave_confMax = False):
        '''
        entry function
        '''
        if force_update is not None:
            if force_update in [True, False]:
                self.args.force_update = force_update
            else:
                raise ValueError('force_update parameter is incorrect, please set with True or False.')

        if not os.path.exists(self.model_result_file) or self.args.force_update:
            if os.path.exists(self.model_result_file):
                os_rmdirs(self.model_pred_dir)
            os_makedirs(self.model_pred_dir)

            task_logger= self.logger_config(
                    self.model_fit_dir, 'train')
            self.load_data(logger=task_logger)

            self.conduct_fit(clogger= task_logger,result_file = self.model_result_file, xfit_stats=xfit_stats)

            F1_score, kappa, acc = self.evaluate(elogger= task_logger, ave_confMax = ave_confMax)
        else:
            F1_score, kappa, acc = self.evaluate(ave_confMax = ave_confMax)

        return F1_score, kappa, acc

    def hyper_config(self, clogger = None):
        '''
        Get the corresponding model's hyperparameters and save it to self.model_opts;\n
        Update the hyperparameters with the self.data_opts and self.args.hyper;\n
        Finally return the correct hyperparameters.
        '''
        if hasattr(self.model_init_path, self.model_name):
            model_opts = getattr(self.model_init_path, self.model_name)
        else:
            raise ValueError(
                'Non-supported model "{}" in the "{}" module, please check the module or the model name'.format(self.model_name, self.model_init_path))


        self.model_opts = model_opts()
        self.model_opts.hyper.merge(opts=self.data_opts.info)
        self.model_opts.hyper.num_classes = self.data_opts.num_classes
        self.model_opts.hyper.model_fit_dir = self.model_fit_dir
        self.model_opts.hyper.model_name = self.model_name
        self.model_opts.hyper.data_name = self.data_name
        # self.model_opts.hyper.arch = self.model_opts.arch

        if 'hyper' in vars(self.args):
            self.model_opts.hyper.update(self.args.hyper)

        cuda_exist = torch.cuda.is_available()
        if cuda_exist and self.args.cuda:
            self.model_opts.hyper.device = torch.device('cuda:{}'.format(self.args.gid))
            torch.cuda.set_device(self.args.gid)
            self.device_id = self.args.gid
        else:
            self.model_opts.hyper.device = torch.device('cpu')
            self.device_id = -1

        model_hyper = Opt(self.model_opts.hyper)
        # if self.tune:
        #     if 'best_hyper' not in self.dict:
        #         self.tuning()

        #     best_hyper = self.best_hyper
        #     model_hyper.update(best_hyper)
        #     if clogger is not None:
        #         clogger.info("Updating tuning result complete.")
        #         clogger.critical('-'*80)
        #     if self.best_checkpoint_path is not None:
        #         model_hyper.pretraining_file = self.best_checkpoint_path

        return model_hyper

    def model_init(self, clogger):
        '''
        Get the corresponding model's hyperparameters (default no tuning) with self.hyper_config()\n
        Then import the model class from the model_opts.import_path. \n
        Finally, use them to create a model instance and return.
        '''
        model_hyper = self.hyper_config(clogger)
        model = importlib.import_module(self.model_opts.import_path)
        model = getattr(model, self.model_opts.class_name)
        model = model(model_hyper, clogger)
        return model

    def model_fit(self, model, clogger, xfit_stats = True):
        '''
        Fit the model with the training set and validation set.
        '''
        train_loader, val_loader = None, None
        if xfit_stats:
            train_loader, val_loader = self.load_fitset(batch_size=model.hyper.batch_size) #get dataloader
            clogger.critical('Loading training set and validation set.')
            clogger.info(f'Fit batch size: {train_loader.batch_size}')
            clogger.info(f"Train_loader batch: {len(train_loader)}")
            clogger.info(f"Val_loader batch: {len(val_loader)}")
            clogger.critical('>'*40)

        # cid_hyper = self.load_hyper(clogger)
        # model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer
        epochs_stats =  model.xfit(train_loader, val_loader, trainer_class=AnnealingTrainer, xfit_stats=xfit_stats, finetune=self.args.finetune, logger = clogger)

        if epochs_stats is not None and set(['val_loss','val_acc', 'train_loss', 'train_acc', 'lr_list']).issubset(epochs_stats.columns):
            loss_dir = os.path.join(self.model_fit_dir, 'loss_curve')
            lossfig_dir = os.path.join(loss_dir, 'figure')
            save_training_process(epochs_stats, plot_dir=lossfig_dir)

        return model, epochs_stats


    def conduct_fit(self, clogger = None, result_file = None, xfit_stats = True):
        '''
        train model using model.xfit()
        draw training process
        predict on the test set and save predict results
        return : predict results , true labels , training process states
        '''
        try:
            if clogger is None:
                clogger = self.logger_config(
                    self.model_fit_dir, 'train')

            model = self.model_init(clogger)
            model, epochs_stats = self.model_fit(model, clogger, xfit_stats=xfit_stats)  #see the xfit method in the models.nn._baseNet class

            # generate the output of the testset
            clogger.critical('>'*40)
            clogger.critical('End fit.')
            pre_lab_all, label_all = self.eval_testset(model, clogger, result_file)

            return pre_lab_all, label_all, epochs_stats
            # epochs_stats is a dataframe with the following columns at least
            # pd.DataFrame(
            # data={"lr_list": self.lr_list,
            #       "train_loss": self.train_loss_list,
            #       "val_loss": self.val_loss_list,
            #       "train_acc": self.train_acc_list,
            #       "val_acc": self.val_acc_list})

        except:
            clogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def eval_testset(self, model, logger, result_file):
        '''
        predict on the test set
        return two list: predict labels and true labels corresponding to each specific SNR value
        save these two list ,respectively named as pred and label
        '''
        if logger is not None:
            logger.critical('>'*40)
            logger.critical('Evaluation on the testing set.')

        with torch.no_grad():

            # if self.model_opts.arch != 'naive':
            model.eval()

            test_sample_list, test_lable_list = self.data_opts.load_testset(test_batch_size = 64)

            pre_lab_all = []
            label_all = []

            # loop of SNRs in test_sample_list
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

    def evaluate(self, elogger = None, xfit_stats=False, ave_confMax = False):
        '''
        For each snr value , calculate accuracy and confusion matrix,save them to Confmat_Set and Accuracy_list
        Calculate overall accuracy , f1_score , kappa_score
        Save confusion matrix heatmap for each snr value
        Save overall accuracy for each snr: Accuracy_list (num_snrs) and accuracy each mod for each snr : Accuracy_Mods  (num_snrs,num_mods)
        '''
        eLogger = set_logger(os.path.join(self.eval_dir, '_log_', '{}.{}.eval.log'.format(self.data_name, self.model_name)), '{}.{}'.format(
                self.data_name, self.model_name.upper())) if elogger is None else elogger

        if self.data_statue is False:
            self.load_data(logger=eLogger)

        self.model_eval_dir = os.path.join(self.eval_dir, self.model_name)
        self.eval_acc_dir = os.path.join(self.model_eval_dir, 'accuracy')
        self.eval_plot_dir = os.path.join(self.model_eval_dir, 'figures')

        os_makedirs(self.eval_acc_dir)
        os_makedirs(self.eval_plot_dir)
        os_makedirs(self.model_pred_dir)

        # for i in self.cid_list: # multiple cross validation in the future version

        # if self.fit_statue and self.args.force_update is False:
        #     force_update = False
        # else:
        #     force_update = True

        if os.path.exists(self.model_result_file) and self.args.force_update is False:
            with np.load(self.model_result_file) as data:
                pre_lab_all, label_all = data['pred'], data['label']
        else:
            pre_lab_all, label_all,_ = self.conduct_fit(xfit_stats = xfit_stats)


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

    # def load_tuner(self, logger):
    #     series_tuner = HyperTuner(self.model_opts, logger, self.data_opts)
    #     return series_tuner

    # def tuning(self):

    #     if self.device_id > 0:
    #         raise ValueError('When using tuning mode, please set args.gid with 0 or using cpu.')

    #     self.tune = True
    #     self.best_checkpoint_path = None

    #     if 'innerTuning' in self.model_opts.dict:
    #         if self.model_opts.innerTuning == True:
    #             self.tune = False

    #     if self.tune:
    #         tuner_dir = os.path.join(self.model_fit_dir, 'tuner')
    #         os_makedirs(tuner_dir)

    #         self.model_opts.tuner.dir = tuner_dir
    #         tLogger = self.logger_config(tuner_dir, 'tuning')

    #         try:
    #             pT_hyper = Opt()
    #             if 'preTuning_model_path' in self.model_opts.tuner.dict:
    #                 pT_path = self.model_opts.tuner.preTuning_model_path
    #                 if os.path.exists(pT_path):
    #                     pT_hyper.merge(torch.load(pT_path))
    #                     self.model_opts.hyper.update(pT_hyper)

    #                     tLogger.critical('-'*80)
    #                     for (arg, value) in pT_hyper.dict.items():
    #                         tLogger.info("PreTuning Results:\t %s - %r", arg, value)
    #                 else:
    #                     raise ValueError('Non-found preTuning results: {}.\nPlease check the preTuning_model: {}'.format(pT_path, self.model_opts.tuner.preTuning_model))
    #             # else:
    #             if len(list(self.model_opts.tuning.dict.keys())) > 0:

    #                 if self.data_statue is False:
    #                     self.load_data(logger=tLogger)

    #                 series_tuner = self.load_tuner(logger=tLogger)
    #                 best_hyper, best_checkpoint_path = series_tuner.conduct() # best_hyper is an Obj
    #             else:
    #                 best_hyper = Opt()


    #             pT_hyper.merge(best_hyper)
    #             self.best_hyper = pT_hyper
    #             self.best_checkpoint_path = best_checkpoint_path

    #             tLogger.critical('-'*80)
    #             tLogger.critical('Tuning complete.')

    #             if isinstance(pT_hyper, Mapping):
    #                 pT_hyper_info = pT_hyper
    #             elif isinstance(pT_hyper, object):
    #                 pT_hyper_info = vars(pT_hyper)
    #             else:
    #                 raise ValueError('Error data type in pT_hyper from: {}'.format(tuner_dir))

    #             for (arg, value) in pT_hyper_info.items():
    #                 tLogger.info("Tuning Results:\t %s - %r", arg, value)

    #         except:
    #             self.tune = False
    #             tLogger.exception(
    #                 '{}\nGot an error on tuning.\n{}'.format('!'*50, '!'*50))
    #             raise SystemExit()

    #     return self.tune

    # def load_tuning(self):
    #     tuner_dir = os.path.join(self.model_fit_dir, 'tuner')
    #     tuner_path = os.path.join(tuner_dir, 'hyperTuning.best.pt')
    #     best_hyper = torch.load(tuner_path)
    #     if not os.path.exists(tuner_path):
    #         raise ValueError(
    #             'Invalid tuner path: {}'.format(tuner_path))
    #     return best_hyper

if __name__ == "__main__":
    args = get_parser()
    args.cuda = True

    args.exp_config = 'exp_config/xinze'
    args.exp_file= 'rml16a'
    # args.exp_name = 'icassp23'
    args.exp_name = 'tuning.mcl'
    args.gid = 0

    args.test = True
    args.clean = True
    args.model = 'awn'


    task = Task(args)
    task.tuning()
    task.conduct()
    # taskRecog.evaluate(force_update=True)