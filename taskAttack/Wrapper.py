import os
import sys
sys.path.append(os.path.abspath('.'))
from taskRecog.util import os_makedirs, os_rmdirs, set_logger, fix_seed
import importlib
from taskRecog.util import Opt, set_dataloader
import torch
import numpy as np
from taskAttack.attackmethods import load_attack_class
# from taskAttack.attackmethods.util import AttackConfig
from tqdm.auto import tqdm
from taskRecog.util import save_confmat
from data import load_data_class
from taskAttack.util import empty_unused_gpu_cache, snrdata_eval, plot_snr_acc, cal_PNR
from taskAttack.Parser import get_parser, get_group_args


class Results(Opt):
    def __init__(self, exp_snrs, num_classes):
        num_snrs = len(exp_snrs)
        self.num_snrs = num_snrs
        self.Confmat_Set = np.zeros(
            (num_snrs, num_classes, num_classes), dtype=int)
        self.Acc = np.zeros(num_snrs, dtype=float)
        self.pe = np.zeros(num_snrs, dtype=float)
        self.psr = np.zeros(num_snrs, dtype=float)
        self.pnr = np.zeros(num_snrs, dtype=float)
        self.snrs_pred = dict()

    def __call__(self, snr_result, i):
        self.Confmat_Set[i] = snr_result['cm']
        self.Acc[i] = snr_result['acc']
        self.pe[i] = snr_result['pe']
        self.psr[i] = snr_result['psr']
        self.pnr[i] = snr_result['pnr']
        if 'pred' in snr_result.keys():
            self.snrs_pred[i] = snr_result['pred']

class Attack(Opt):
    def __init__(self, args, parser):
        # super().__init__(args)
        self.args = args
        self.seed = args.seed
        fix_seed(args.seed)

        # self.exp_module = importlib.import_module('{}.{}'.format(
            # args.exp_config.replace('/', '.'), args.exp_file)) # need improve! to avoid multi-import


        if 'model_init_path' not in vars(args) or args.model_init_path == '':
            model_init_path = '{}.{}'.format(args.expconfig_dir, args.data).replace('/', '.')
        else:
            model_init_path = args.model_init_path.replace('.py', '').replace('/', '.')
        self.model_init_path = importlib.import_module(model_init_path)


        self.data_config(args)
        # self.model_config(args)
        self.exp_config(args)
        self.parser = parser
        self.AlgoArgs = get_group_args(self.parser, new_args=self.args, group_name='Attack')


    def logger_config(self, dir, stage, model_name, rewrite=True):
        log_path = os.path.join(dir, '{}.log'.format(stage))
        log_name = '{}.{}.{}.{}.{}'.format(
            self.exp_name, self.data_name, self.boundinfo, self.args.algo,model_name)
        logger = set_logger(log_path, log_name,
                            self.logger_level, rewrite=rewrite)
        return logger

    def exp_config(self, args):
        self.model_name = args.surrogate_model

        self.exp_name = args.exp_name
        if 'exp_dir' in vars(args) and args.exp_dir is not None:
            self.exp_dir = args.exp_dir
        else:
            assert 'exp_dataname_first' in vars(args)
            self.exp_dir = 'yield_results' if not args.test else 'yield_test'
            self.exp_dir = os.path.join(self.exp_dir, self.data_name, self.exp_name) if args.exp_dataname_first else os.path.join(
                self.exp_dir, self.exp_name, self.data_name)

        self.fit_dir = os.path.join(self.exp_dir, 'surrogate') #toDO: remove the middle surrogate dir. in the next big version.

        self.model_fit_dir = os.path.join(self.fit_dir, self.model_name)
        self.pert_dir = os.path.join(
            self.model_fit_dir, f'{self.args.algo}_results')
        self.pert_filedict = {}

        if args.test and args.clean:
            os_rmdirs(self.model_fit_dir)

        if args.log_level == 'i':
            self.logger_level = 20  # equal to info
        elif args.log_level == 'c':
            self.logger_level = 50  # equal to critical

        self.exp_snrs = self.args.snr

        system_snrs = []
        self.attackset_path = os.path.join('data/postdata/', f'attack.{self.data_name}_dict.pt')
        if self.args.attackset == 'attack':
            if not os.path.exists(self.attackset_path):
                raise FileNotFoundError(f"Attack set file {self.attackset_path} does not exist. Please run the attack set generation script first.")

            common_balanced_samples = torch.load(self.attackset_path, weights_only=False)
            system_snrs = list(common_balanced_samples.keys())

            print(
                f"Using the balanced jointly-correct attack set at "
                f"{self.attackset_path}."
            )
            assert set(system_snrs).issubset(set(self.data_opts.snr_envs)), f"Some of the specified attack set SNRs: {system_snrs} \nare not in the dataset SNRs: {self.data_opts.snr_envs}."
        else:
            system_snrs = self.data_opts.snr_envs


        if len(self.args.snr) == 1:
            if self.args.snr[0] == 'all':
                self.exp_snrs = system_snrs
            else:
                self.exp_snrs = [int(self.args.snr[0])]
        elif len(self.args.snr) == 2:
            snr_start, snr_end = int(self.args.snr[0]), int(self.args.snr[1])
            self.exp_snrs = [s for s in system_snrs if snr_start <= s <= snr_end]
        else:
            raise ValueError('Invalid format for -snr argument. Please provide a single value, a range (two values), or "all".')

        for s in self.exp_snrs:
            if s not in self.data_opts.snr_envs:
                raise ValueError(
                    f"Non supported SNR condition '{s}' for Dataset {self.data_name}")
            self.pert_filedict[s] = os.path.join(
                self.pert_dir, f'result.snr{s}.pt')

        self.attacker = None
        self.boundinfo = dict(
            pnr=self.args.pnr) if self.args.bound == 'pnr' else dict(psr=self.args.psr)

    def data_config(self, args):
        data_opts = load_data_class(args.data)
        self.data_opts = data_opts(args)
        self.data_name = self.data_opts.data_name

    def load_data(self, logger=None):
        logger.info(
            'Loading the datasets {}'.format(self.data_name))
        self.data_opts.info.seed = self.seed
        # self.data_opts.load_rawdata(logger = logger)
        self.test_snrloaders = {}
        if self.args.attackset == 'test':
            self.data_opts.pack_dataset(logger=logger)
            # logger.info('-'*80)
            Dataset_size = self.data_opts.train_set[0].shape[0] + \
                self.data_opts.val_set[0].shape[0] + \
                self.data_opts.test_set[0].shape[0]
            logger.info('-'*80)
            logger.info(
                f'Dataset size: {int(Dataset_size)}\t Class num: {self.data_opts.num_classes}')
            logger.info(
                f"Signal_train.shape: {list(self.data_opts.train_set[0].shape)}", )
            logger.info(
                f"Signal_val.shape: {list(self.data_opts.val_set[0].shape)}")
            logger.info(
                f"Signal_test.shape: {list(self.data_opts.test_set[0].shape)}")

            for snr in self.exp_snrs:
                sig_i, lab_i, _ = self.data_opts.snr_slice('test', snr)
                dataset = (sig_i, lab_i)
                batch_size = self.args.batch_size if 'batch_size' in vars(
                    self.args) else sig_i.size(0)
                snr_data_loader = set_dataloader(
                    batch_size=batch_size, data_set=dataset, shuffle=False)
                self.test_snrloaders[snr] = snr_data_loader
        elif self.args.attackset == 'attack':
            # attackset_path = os.path.join('data/postdata/', f'attack.{self.data_name}_dict.pt')
            common_balanced_samples = torch.load(self.attackset_path, weights_only=False)

            for snr in self.exp_snrs:
                snr_sig, snr_lab, snr_idx = common_balanced_samples[snr]

                dataset = (snr_sig, snr_lab)
                batch_size = self.args.batch_size if 'batch_size' in vars(
                    self.args) else sig_i.size(0)
                snr_data_loader = set_dataloader(
                    batch_size=batch_size, data_set=dataset, shuffle=False)
                self.test_snrloaders[snr] = snr_data_loader

            logger.info("Using the experimental SNRs: {} from the attack set.".format(self.exp_snrs))

        else:
            raise ValueError(f"Invalid attackset {self.args.attackset}. Supported attackset options are 'test' and 'attack'.")

    def conduct(self, eval=True , ave_confMax=False, show_variance = False, algo_configs = None):
        task_logger = self.logger_config(
            dir=self.pert_dir, stage='surrogate', model_name=self.model_name)
        self.load_data(logger=task_logger)

        if not os.path.exists(self.pert_dir):
            os_makedirs(self.pert_dir)

        snrs_pert_dict = {}
        self.attacker_config(
            clogger=task_logger, configs=self.AlgoArgs, algo_configs=algo_configs)
        with tqdm(total=len(self.exp_snrs), desc='SNRs', mininterval=0.3, colour='red', leave=False) as pbar:
            for snr in self.exp_snrs:
                snr_pert_dict = self.snr_attack(logger=task_logger, snr=snr)
                snrs_pert_dict[snr] = snr_pert_dict
                pbar.update(1)

        if eval:
            self.evaluation(snrs_pert_dict, ave_confMax=ave_confMax, show_variance=show_variance)

        empty_unused_gpu_cache()
        return snrs_pert_dict

    def evaluation(self, snrs_pert_dict, ave_confMax=True, show_variance = True):
        self.eval_dir = os.path.join(self.pert_dir, f'target.{self.args.target_model}')
        elogger = self.logger_config(dir=self.eval_dir, stage='evaluation',
                                     model_name='{}->{}'.format(self.args.surrogate_model, self.args.target_model))
        try:
            self.victim = self.model_import(
                elogger, model_name=self.args.target_model, ckp=self.args.target_ckp)
            elogger.info('Save accuracy file to the location: {}/'.format(
                os.path.join(self.pert_dir, f'target.{self.args.target_model}')))
            clean = self.record_result(
                snrs_pert_dict, model=self.victim, logger=elogger, ave_confMax=ave_confMax, tag='clean')
            adv = self.record_result(
                snrs_pert_dict, model=self.victim, logger=elogger, ave_confMax=ave_confMax, tag='adv')

            if show_variance:
                from taskAttack.util import plot_delta_variance
                plot_delta_variance(snrs_pert_dict, clean, adv, self.eval_dir, self.data_name)

            # from taskAttack.util import cal_std_successRate, plot_snr_successRate
            # cal_std_successRate(snrs_pert_dict, clean, adv, self.eval_dir)
            # plot_snr_successRate(snrs_pert_dict, clean, adv, self.eval_dir)

            for i, snr in enumerate(self.exp_snrs):
                elogger.info(
                    f'SNR {snr}: Clean Acc. {"{:.2f}".format(clean.Acc[i] * 100)}% -> Adv. Acc. {"{:.2f}".format(adv.Acc[i] * 100)}%')

            elogger.critical(
                f'Overall: Clean Acc. {"{:.2f}".format(clean.Acc.mean() * 100)}% -> Adv. Acc. {"{:.2f}".format(adv.Acc.mean() * 100)}%')

            acc_list = [(clean.Acc, 'clean'), (adv.Acc, 'adv')]
            plot_snr_acc(acc_list, self.exp_snrs,
                         self.data_name, self.eval_dir)
            return clean, adv
        except Exception:
            elogger.exception(
                '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            raise SystemExit()

    def snr_result(self, model, snr_pert_dict, snr, tag, logger):

        X, Label = [], []

        for _signals, _labels in self.test_snrloaders[snr]:
            X.append(_signals)
            Label.append(_labels)

        X = torch.cat(X, dim=0)
        Label = torch.cat(Label, dim=0)

        batch_size = self.args.batch_size if 'batch_size' in vars(
            self.args) else None

        Pert = snr_pert_dict['pert']

        # clean data
        PSR, PNR = cal_PNR(X, Pert, snr)
        epsilon = torch.norm(Pert.detach().clone().view(Pert.size(0), -1), dim=-1).mean().item()

        if tag != 'clean':
            X = X + Pert
        cm_i, acc, (pred_i, label_i) = snrdata_eval(
            X, Label, model, batch_size)

        result = dict(acc = acc, cm = cm_i, pred = pred_i, label = label_i, pe = epsilon, psr = PSR, pnr = PNR)

        if tag != 'clean':
            pinfo = 'SNR {}: P.E: {:.2g}, PSR {:.1f} <-> PNR {:.1f}'.format(
            snr, epsilon, PSR, PNR)
            logger.info(pinfo)
        return result

    def record_result(self, snrs_pert_dict, model, logger, tag='adv', ave_confMax=False):
        # logger.info(f'\nOn the target.{model.hyper.model_name}, testing with {tag} data')
        model_name = model.hyper.model_name
        if tag == 'adv':
            eval_dir = os.path.join(self.pert_dir, f'target.{model_name}', tag)
        elif tag == 'clean':
            eval_dir = self.model_fit_dir
        else:
            eval_dir = os.path.join(self.pert_dir, f'target.{model_name}', tag)

        os_makedirs(eval_dir)

        tgt_acc_file = os.path.join(eval_dir, f'acc.{tag}.pt')

        if os.path.exists(tgt_acc_file) and snrs_pert_dict is None:
            results_dict = torch.load(tgt_acc_file, weights_only=False)
            results = Opt(init=results_dict)
            logger.info('{} reading snrs_pert_dict results from {}'.format(tag,tgt_acc_file))

        else:
            results = Results(self.exp_snrs, model.hyper.num_classes)
            with tqdm(total=len(self.exp_snrs), desc=f'{tag.capitalize()} Data', mininterval=0.3, colour='red', leave=False) as sbar:
                for i, snr in enumerate(self.exp_snrs):
                    snr_pert_dict = snrs_pert_dict[snr]
                    snr_result = self.snr_result(model,snr_pert_dict, snr, tag, logger)
                    results(snr_result,i)
                    sbar.update(1)

            # F1_score = f1_score(pre_lab_all, pre_lab_all, average='macro')
            # kappa = cohen_kappa_score(label_all, pre_lab_all)
            if ave_confMax:
                save_confmat(results.Confmat_Set, self.exp_snrs,
                            self.data_opts.classes, eval_dir)
            # Accuracy_Mods = save_snr_acc(Acc, Confmat_Set, self.exp_snrs, self.data_name, self.data_opts.classes.keys(), eval_dir)
            torch.save(results.dict, tgt_acc_file)

        return results

    def model_import(self, clogger = None, model_name='awn', ckp = ''):

        if hasattr(self.model_init_path, model_name):
            model_opts = getattr(self.model_init_path, model_name)
        else:
            raise ValueError(
                'Non-supported model "{}" in the "{}" module, please check the module or the model name'.format(model_name, self.model_init_path))

        model_opts = model_opts()
        # Only merge the key-value not in the current hyper.\n
        model_opts.hyper.merge(opts=self.data_opts.info)
        model_opts.hyper.arch_type = model_opts.arch

        cuda_exist = torch.cuda.is_available()
        if cuda_exist and self.args.cuda:
            model_opts.hyper.device = torch.device(
                'cuda:{}'.format(self.args.gid))
            torch.cuda.set_device(self.args.gid)

        else:
            model_opts.hyper.device = torch.device('cpu')

        info = clogger.info if clogger is not None else print

        info('*'*80)
        info('Dataset: {}\t Model:{} \t Class: {}'.format(
            self.data_name, model_name, self.data_opts.num_classes))

        model_hyper = Opt(model_opts.hyper)
        model_hyper.num_classes = self.data_opts.num_classes
        # model_hyper.model_fit_dir = self.model_fit_dir
        model_hyper.model_name = model_name
        model_hyper.data_name = self.data_name

        model = importlib.import_module(model_opts.import_path)
        model = getattr(model, model_opts.class_name)
        model = model(model_hyper, clogger)
        if ckp != '':
            model.load_pretraing_file(file_path=ckp)
        else:
            model.load_pretraing_file(file_path=model_hyper.pretraining_file)
        model.eval()

        return model

    def attacker_config(self, clogger = None, configs=None, algo_configs=None):
        if self.attacker is None:
            # assert clogger is not None
            self.model = self.model_import(
                clogger, model_name=self.args.surrogate_model, ckp=self.args.surrogate_ckp)
            # insert config from the parser
            config_dict = {key: value for key, value in vars(
                configs).items()} if configs is not None else {}
            # insert other config from the wrapper
            config_dict['pert_dir'] = self.pert_dir
            config_dict['data_info'] = self.data_opts.info

            Attacker = load_attack_class(self.args.algo)
            # init attacker with related args
            self.attacker = Attacker(
                model=self.model, logger=clogger, **config_dict)

            if algo_configs is not None:
                config_dict.update(algo_configs)
            self.attacker.update_params(**config_dict)

    def snr_attack(self, logger, snr):
        try:
            pert_file = self.pert_filedict[snr]
            if os.path.exists(pert_file):
                snr_pert_dict = torch.load(self.pert_filedict[snr])
                logger.info(
                    'Read perturbations file from the location: {}'.format(pert_file))
            else:
                snr_pert_dict = self._snr_attack(snr)
                # snr_pert_dict = dict(x=x,pert = pert, label = label)
                torch.save(snr_pert_dict, pert_file)
                logger.info(
                    'Save perturbations file to the location: {}'.format(pert_file))
            return snr_pert_dict


        except RuntimeError as e:
            # logger.exception(
            #     '{}\nGot an error on conduction.\n{}'.format('!'*50, '!'*50))
            # Provide a helpful hint if this is the cudnn RNN backward error
            err_msg = str(e)
            if 'RNN backward' in err_msg:
                hint = ("\n"+ "!"*30 +"\nRuntimeError from RNN backward detected. This often happens when a model containing RNN/LSTM/GRU layers is in eval mode during a backward pass. \n Please ensure the model exposes a boolean attribute `has_rnn=True` if it contains RNN layers.")

                logger.exception(hint)
                hint = ("Model.has_rnn =", getattr(self.model, 'has_rnn', None))
                logger.exception(hint)
            tqdm.write(err_msg, file=sys.stderr)

            # Re-raise so caller can still handle/see the original traceback
            raise SystemExit()

    def _snr_attack(self, snr):
        '''The actual attack codes of attacking the model for an snr-specific data'''
        snr_pert_dict = self.attacker.snr_attack(
                    self.test_snrloaders[snr], snr)
        return snr_pert_dict


if __name__ == "__main__":

    args, parser = get_parser()

    args.log_level = 'i'
    args.snr = [10]
    args.algo = 'mi'
    args.gid = 1
    args.pnr = 0
    args.bound = 'psr'
    args.psr = -12
    args.batch_size = 1024
    args.gid = 2
    args.attackset = 'attack'
    args.data = 'dr2'

    args.surrogate_model = 'awn'
    args.target_model = 'awn'
    args.surrogate_ckp = 'checkpoints/MIMO.Nt4Nr2/nature/MIMO.Nt4Nr2_awn.best.pt'
    args.target_ckp = 'checkpoints/MIMO.Nt4Nr2/nature/MIMO.Nt4Nr2_awn.best.pt'


    args.cuda = True
    args.test = True
    args.clean = True

    task = Attack(args, parser)
    task.conduct()
