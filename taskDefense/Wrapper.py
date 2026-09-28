import os
# import torch
from taskAttack.Parser import get_group_args
from taskRecog.util import set_dataloader, os_makedirs, os_rmdirs
from tqdm.auto import tqdm
from taskRecog.Wrapper import Task as BaseTask
from taskAttack.Wrapper import Attack as AttackTask
from taskAttack.attackmethods import load_attack_class
from taskAttack.util import plot_snr_acc

class Task(BaseTask, AttackTask):
    """
    Wrapper for the attack task.
    """

    def __init__(self, args, parser):
        super().__init__(args) # Initialize the base class with BaseTask

        self.parser = parser
        self.defense_method = args.defense_method if hasattr(args, 'defense_method') else 'PGD_AT'
        # Initialize the attack algorithm
        # self.attack_algo = self.get_attack_algo(args.algo, args)
        # self.attack_method = args.attack_method if hasattr(args, 'attack_method') else 'FGSM'

    def exp_config(self, args):
        '''
        create experiment directory and set up some experiment configurations
        '''
        self.exp_name = args.exp_name
        if 'exp_dir' in vars(args) and args.exp_dir is not None:
            self.exp_dir = args.exp_dir
        else:
            assert 'exp_dataname_first' in vars(args)
            self.exp_dir = 'yield_results' if args.test == False else 'yield_test'
            self.exp_dir = os.path.join(self.exp_dir, self.data_name, self.exp_name) if args.exp_dataname_first else os.path.join(self.exp_dir, self.exp_name, self.data_name)

        self.fit_dir = os.path.join(self.exp_dir, 'fit')
        self.eval_dir = os.path.join(self.exp_dir, 'eval')

        self.model_name = '{}'.format(args.model)

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name)
        self.model_pred_dir = os.path.join(self.model_fit_dir, 'pred_results')
        self.model_result_file = os.path.join(self.model_pred_dir, 'results.npz')

        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)

    def attack_config(self, alogger):
        """
        Configure the attack method.
        """
        boundinfo = ""
        if self.args.norm == 'l2':
            boundinfo = f"pnr.{self.args.pnr}" if self.args.bound == 'pnr' else f"psr.{self.args.psr}"
        elif self.args.norm == 'linfty':
            boundinfo = f"epsilon.{self.args.epsilon}"
        else:
            raise ValueError(f"Unsupported norm type: {self.args.norm}")

        self.pert_dir = os.path.join(self.model_fit_dir, f'results.{self.args.algo}.{boundinfo}')

        if not os.path.exists(self.pert_dir):
            os_makedirs(self.pert_dir)

        self.exp_snrs = self.data_opts.snr_envs if self.args.snr == ['all'] else [int(s) for s in self.args.snr]

        self.pert_filedict = {}
        for s in self.exp_snrs:
            if s not in self.data_opts.snr_envs:
                raise ValueError(
                    f"Non supported SNR condition '{s}' for Dataset {self.data_name}")
            self.pert_filedict[s] = os.path.join(self.pert_dir, f'result.snr{s}.pt')


        config_dict = get_group_args(self.parser, new_args=self.args, group_name='Attack')
        config_dict = {key: value for key, value in vars(
                config_dict).items()} if config_dict is not None else {}
        config_dict['pert_dir'] = self.pert_dir
        config_dict['data_info'] = self.data_opts.info

        attacker = load_attack_class(self.args.algo)
        self.attacker = attacker(model=self.model, logger=alogger, **config_dict)

        if self.data_statue is False:
            self.load_data(logger=alogger)

        self.test_snrloaders = {}
        for snr in self.exp_snrs:
            sig_i, lab_i, _ = self.data_opts.snr_slice('test', snr)
            dataset = (sig_i, lab_i)
            batch_size = self.args.batch_size if 'batch_size' in vars(
                self.args) else sig_i.size(0)
            snr_data_loader = set_dataloader(
                batch_size=batch_size, data_set=dataset, shuffle=False)
            self.test_snrloaders[snr] = snr_data_loader


        self.model.eval()

    def conduct(self, eval = True, xfit_stats = True, ave_confMax = True, show_variance = False):

        dlogger = self.logger_config(
            dir=self.model_fit_dir, stage='defense', name=f'{self.defense_method}')
        dlogger.info('Conducting defense task with method: {}'.format(self.defense_method))

        # fit the model with defense method, and return the defended model with self.model
        self.conduct_fit(dlogger=dlogger, xfit_stats=xfit_stats)

        attack_logger_name = f"{self.exp_name}.{self.data_name}.{self.args.bound}.{self.model_name}.{self.args.algo}"
        alogger = self.logger_config(
            dir=self.model_fit_dir, stage=f'attack.{self.args.algo}', name=attack_logger_name)


        self.attack_config(alogger)

        snrs_pert_dict = {}
        with tqdm(total=len(self.exp_snrs), desc='SNRs', mininterval=0.3, colour='red') as pbar:
            for snr in self.exp_snrs:
                snr_pert_dict = self.snr_attack(logger=alogger, snr=snr)
                snrs_pert_dict[snr] = snr_pert_dict
                pbar.update(1)

        if eval:
            self.evaluation(snrs_pert_dict, alogger, ave_confMax=ave_confMax, show_variance=show_variance)

        return snrs_pert_dict

    def evaluation(self, snrs_pert_dict, logger, ave_confMax=False, show_variance = False):
        # self.eval_dir = os.path.join(self.pert_dir, 'eval')
        # elogger = self.logger_config(dir=self.eval_dir, stage='evaluation',
        #                              name='{}->{}'.format(self.args.model, self.args.model))
        try:

            logger.critical('Save accuracy file to the location: {}/'.format(
                os.path.join(self.pert_dir, f'{self.model_name}')))
            clean = self.record_result(
                snrs_pert_dict, model=self.model, logger=logger, ave_confMax=ave_confMax, tag='clean')
            adv = self.record_result(
                snrs_pert_dict, model=self.model, logger=logger, ave_confMax=ave_confMax, tag='adv')

            if show_variance:
                from taskAttack.util import plot_delta_variance
                plot_delta_variance(snrs_pert_dict, clean, adv, self.pert_dir, self.data_name)

            for i, snr in enumerate(self.exp_snrs):
                logger.critical(
                    f'SNR {snr}: Clean Acc. {"{:.2f}".format(clean.Acc[i] * 100)}% -> Adv. Acc. {"{:.2f}".format(adv.Acc[i] * 100)}%')

            logger.critical(
                f'Overall: Clean Acc. {"{:.2f}".format(clean.Acc.mean() * 100)}% -> Adv. Acc. {"{:.2f}".format(adv.Acc.mean() * 100)}%')

            acc_list = [(clean.Acc, 'clean'), (adv.Acc, 'adv')]
            plot_snr_acc(acc_list, self.exp_snrs,
                         self.data_name, self.pert_dir)
            return clean, adv
        except Exception:
            logger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def defense_config(self,):
        """
        Configure the defense method.
        """
        from taskDefense.advTraining import load_defense_class

        defense_class = load_defense_class(self.defense_method)

        return defense_class

        # if not hasattr(self.args, 'hyper'):
        #     self.args.hyper = Opt()
        # else:
        #     if isinstance(self.args.hyper, type) and hasattr(self.args.hyper, 'trainer_module'):
        #         assert self.args.hyper.trainer_module == defense_zoo[self.defense_method]

        # self.args.hyper.trainer_module = defense_zoo[self.defense_method]

    def conduct_fit(self, dlogger=None, xfit_stats=True):
        try:
            model = self.model_init(dlogger)
            # if not os.path.exists(self.model_pred_dir): # can be commented out if not needed
            #     os.makedirs(self.model_pred_dir)

            dlogger.critical('>'*40)
            dlogger.info('Conducting fit for attack task with defense method: {}'.format(self.defense_method))
            dlogger.critical('>'*40)
            self.load_data(logger=dlogger)
            self.model, _ = self.model_fit(model, dlogger, xfit_stats=xfit_stats)

        except Exception:
            dlogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def model_fit(self, model, clogger, xfit_stats = True):
        '''
        Fit the model with the training set and validation set.
        '''
        train_loader, val_loader = None, None
        if xfit_stats:
            train_loader, val_loader = self.load_fitset(batch_size=self.args.batch_size) #get dataloader
            clogger.critical('Loading training set and validation set.')
            clogger.info(f'Fit batch size: {train_loader.batch_size}')
            clogger.info(f"Train_loader batch: {len(train_loader)}")
            clogger.info(f"Val_loader batch: {len(val_loader)}")
            clogger.critical('>'*40)

        # cid_hyper = self.load_hyper(clogger)
        # model = model(cid_hyper, clogger) #todo: check model_fit_dir in model and model trainer
        defense_class = self.defense_config()

        defense_attacker_kwargs = dict(PSR=self.args.psr, warmup_epoch = self.args.warmup_epoch) # can be extended with more parameters if needed

        epochs_stats =  model.xfit(train_loader, val_loader, trainer_class = defense_class,xfit_stats=xfit_stats, finetune=self.args.finetune, logger = clogger, **defense_attacker_kwargs)

        # from taskRecog.util import save_training_process
        if epochs_stats is not None and set(['train_loss','val_loss', 'train_nat_acc', 'train_adv_acc','val_nat_acc','val_adv_acc', 'lr_list']).issubset(epochs_stats.columns):
            plot_dir = os.path.join(self.model_fit_dir, 'loss_curve')
            # plot_dir = os.path.join(loss_dir, 'figure')

            from matplotlib import pyplot as plt
            os_makedirs(plot_dir)
            fig = plt.figure(figsize=(18, 4))
            plt.subplot(1, 3, 1)
            plt.plot(epochs_stats.epoch, epochs_stats.train_loss,
                    "r-", label="Train loss")
            plt.plot(epochs_stats.epoch, epochs_stats.val_loss,
                    "b-", label="Val adv loss")
            plt.legend()
            plt.grid()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.subplot(1, 3, 2)
            plt.plot(epochs_stats.epoch, epochs_stats.train_nat_acc, "r-", label="Train nat acc")
            plt.plot(epochs_stats.epoch, epochs_stats.val_nat_acc, "b-", label="Val nat acc")
            plt.plot(epochs_stats.epoch, epochs_stats.train_adv_acc, "r--", label="Train adv acc")
            plt.plot(epochs_stats.epoch, epochs_stats.val_adv_acc, "b--", label="Val adv acc")
            plt.legend()
            plt.grid()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")

            plt.subplot(1, 3, 3)
            plt.plot(epochs_stats.epoch, epochs_stats.lr_list, label = 'Learing rate')
            plt.xlabel("epoch")
            plt.ylabel("lr")
            plt.legend()
            plt.grid()

            fig.savefig(os.path.join(plot_dir, 'loss_acc.png'), dpi=300, bbox_inches='tight')
            plt.close()

        return model, epochs_stats



if __name__ == "__main__":
    from taskDefense.Parser import get_parser

    args, parser = get_parser(parsing=True)

    args.test = True
    args.clean = True
    args.data = 'dr2'
    args.exp_name = 'defense_attack_unit_test'
    args.expconfig_dir = 'exp_config/defense/config'

    args.defense_method = 'pgdat'
    args.model = 'awn'

    args.snr = [0,10]
    args.algo = 'mi'
    args.psr = -10
    args.cuda = True
    args.gid = 2
    args.warmup_epoch = 5


    # uncomment the following lines to set pretraining file, if not set, the model will be trained with the given defense method
    # args.hyper = Opt()
    # args.hyper.pretraining_file = 'data/RML2016.10a/pre_AT_models/RML2016.10a_awn.abest.pt'

    task = Task(args, parser)
    task.conduct(eval=True)