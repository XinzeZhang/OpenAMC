
from taskRecog.util import Opt
from copy import deepcopy

import importlib
import os
from taskRecog.util import set_logger
from data import load_data_class

class MutliRoundHardMask(Opt):
    def __init__(self, args):
        self.args = args

        if 'finetune' not in vars(self.args):
            self.args.finetune = False

        # self.exp_module = importlib.import_module('{}.{}'.format(
        #     args.exp_config.replace('/', '.'), args.exp_file))

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


    def data_config(self, args):
        # data_opts = getattr(self.exp_module, args.dataset + '_data')
        data_opts = load_data_class(args.data)
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
        self.logger = self.logger_config(self.exp_dir, f'{self.model_name}')
        # snrmodel = self.snrmodel_config(self.args)
        # snrmodel.eval()

        for i_iter in range(1, self.args.rounds+1):
            iter_args = deepcopy(self.args)
            iter_args.exp_name = self.args.exp_name+f'.iter{i_iter}'
            iter_args.exp_dir = os.path.join(self.exp_dir, f'iter{i_iter}')

            iter_hyper = Opt()
            if i_iter > 1:
                iter_hyper.pretraining_allDatafile = os.path.join(self.exp_dir, f'iter{i_iter -1}/fit/{self.model_name}/checkpoint/{self.data_name}_{self.model_name}.best.pt')
            iter_hyper.pretraining_file = os.path.join(self.exp_dir, f'iter{i_iter}/fit/{self.model_name}/checkpoint/{self.data_name}_{self.model_name}.best.pt')
            iter_hyper.sampler_file = os.path.join(self.exp_dir, f'iter{i_iter}/fit/{self.model_name}/sampler/snrs_step_mask.pt')
            # SnrPredTest.AMC/env2/RML2016.10a/SnrPredTest.AMC/env2.iter1/fit/awn/checkpoint/RML2016.10a_awn.best.pt
            iter_args.hyper = iter_hyper

            task = self.task_config(iter_args) # task = self.task_config(iter_args, snrmodel)
            result_statue= task.conduct()
            if not result_statue:
                raise ValueError('Failed to storage results')
            # else:
            #     print(f'Finishe round {i_iter}')
        # return True
            # score= task.conduct(force_update=self.args.force_update)
            # val_acc= score.val_acc
            # ori_acc = score.ori_acc
            # refmask_acc = score.refmask_acc
            # self.logger.info('Iter: {}\tVal. OA: {:.2f}%\tOri. OA: {:.2f}%\tRef. Mask OA: {:.2f}%'.format(i_iter, val_acc* 100, ori_acc * 100, refmask_acc * 100))
            # self.logger.info('Iter: {}\tOri. OA: {:.2f}%\tPred. Mask OA: {:.2f}%\tRef. Mask OA: {:.2f}%'.format(i_iter, ori_acc * 100, predmask_acc * 100, refmask_acc * 100))

    def evaluate(self,):
        self.args.force_update = False

        self.logger = self.logger_config(self.exp_dir, f'{self.model_name}')

        acc_list = [['Iters','val_acc','ori acc', 'refmask acc']]

        import numpy as np
        acc_array = np.zeros(shape=(3, self.args.rounds))
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


            task = self.task_config(iter_args)
            score= task.evaluate()

            val_acc= score.val_acc
            ori_acc = score.ori_acc
            refmask_acc = score.refmask_acc
            self.logger.info('Iter: {}\tVal. OA: {:.2f}%\tOri. OA: {:.2f}%\tRef. Mask OA: {:.2f}%'.format(i_iter, val_acc* 100, ori_acc * 100, refmask_acc * 100))

            cur_list = []
            cur_list.append('{}'.format(i_iter))
            cur_list.append('{:.2f}'.format(val_acc* 100))
            cur_list.append('{:.2f}'.format(ori_acc* 100))
            cur_list.append('{:.2f}'.format(refmask_acc* 100))
            acc_list.append(cur_list)
            acc_array[0, i_iter-1] = val_acc * 100
            acc_array[1, i_iter-1] = ori_acc * 100
            acc_array[2, i_iter-1] = refmask_acc* 100

        val_array = acc_array[0,:]
        _val = val_array[::-1]
        max_iter = len(_val) - np.argmax(_val) - 1
        max_val = acc_array[0, max_iter]
        max_ori = acc_array[1, max_iter]
        max_refmax = acc_array[2, max_iter]
        max_list = []
        max_list.append('{}'.format(max_iter + 1))
        max_list.append('{:.2f}'.format(max_val))
        max_list.append('{:.2f}'.format(max_ori))
        max_list.append('{:.2f}'.format(max_refmax))
        acc_list.append(max_list)

        acc_list = list(map(list, zip(*acc_list)))
        np.savetxt(os.path.join(self.exp_dir, 'logs', 'acc.{}.{}.csv'.format(self.model_name, self.data_name)), acc_list, delimiter=',', fmt='%s')
        self.logger.info('Selected Iter: {}\tVal. OA: {:.2f}%\tOri. OA: {:.2f}%\tRef. Mask OA: {:.2f}%'.format(max_iter + 1, max_val, max_ori, max_refmax))

    def task_config(self, args):
        if args.IQ:
            if args.ablation == 'non':
                from taskRecog.dataAug.SoftMask.IQAugWrapper import IQHardAugTask as HardAug
            else:
                raise ValueError(f'Non supported ablation arg: {args.ablation}')
        else:
            if args.ablation == 'non':
                from taskRecog.dataAug.SoftMask.AugWrapper import HardAugTask as HardAug
            else:
                raise ValueError(f'Non supported ablation arg: {args.ablation}')
        # elif args.ablation == 'prob':
        #     from taskRecog.dataAug.SoftMask.ablations.ProbHardAugWrapper import WeightAugMaskTask as HardAug
        # elif args.ablation == 'rand':
        #     from taskRecog.dataAug.SoftMask.ablations.RandHardAugWrapper import WeightAugMaskTask as HardAug
        # elif args.ablation == 'ratio':
        #     from taskRecog.dataAug.SoftMask.ablations.RatioHardAugWrapper import WeightAugMaskTask as HardAug
        task = HardAug(args)
        return task


class MutliRoundSoftMask(MutliRoundHardMask):
    def task_config(self,args):
        if args.IQ:
            if args.ablation == 'non':
                from taskRecog.dataAug.SoftMask.IQAugWrapper import IQSoftAugTask as SoftAug
            else:
                raise ValueError(f'Non supported ablation arg: {args.ablation}')
        else:
            if args.ablation == 'non':
                from taskRecog.dataAug.SoftMask.AugWrapper import SoftAugTask as SoftAug
            else:
                raise ValueError(f'Non supported ablation arg: {args.ablation}')

        # elif args.ablation == 'prob':
        #     from taskRecog.dataAug.SoftMask.ablations.ProbSoftAugWrapper import WeightAugMaskTask as SoftAug
        # elif args.ablation == 'rand':
        #     from taskRecog.dataAug.SoftMask.ablations.RandSoftAugWrapper import WeightAugMaskTask as SoftAug
        # elif args.ablation == 'ratio':
        #     from taskRecog.dataAug.SoftMask.ablations.RatioSoftAugWrapper import WeightAugMaskTask as SoftAug
        task = SoftAug(args)
        return task