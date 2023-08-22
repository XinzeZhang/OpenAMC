import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import torch
import torch.utils.data as Data

import ray
from task.base.TaskLoader import Opt
# from ray.tune.suggest.bohb import TuneBOHB
# from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining 
# https://arxiv.org/pdf/1711.09846.pdf.
from ray.tune.schedulers.pb2 import PB2 
# pip install GPy sklearn
# https://arxiv.org/abs/2002.02518 (NIPS 2020)
from ray import tune
from ray.air import session, FailureConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
# from ray.air.checkpoint import Checkpoint

from ray.tune.search.nevergrad import NevergradSearch
from ray.tune.search.ax import AxSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import ExperimentAnalysis

import nevergrad as ng

import importlib

from task.util import os_rmdirs,set_logger,set_fitset
from task.base.TaskTuner import HyperTuner
from task.maskcut.maskLoader import mask_cutout, mask_count
from random import sample
import numpy as np

def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)

class MaskTuner(HyperTuner):
    def __init__(self, opts = None, logger= None, subPack = None):
        super().__init__(opts, logger, subPack)

    def gen_p2e(self,):
        config = {}
        for key in self.tuning.dict.keys():
            config[key] = self.hyper.dict[key]
            
        self.points_to_evaluate = []
        
        idx = [i for i in range(self.hyper.sig_len)]
        # tags = ['inputMask_{}'.format(i) for i in range(self.hyper.sig_len)]
        
        for mr in [0.02, 0.03, 0.04, 0.05]:
            num = int(self.hyper.sig_len * mr)
            for t in range(4):
                selected_idx = sample(idx, num)
                new_config = config.copy()
                for s_idx in selected_idx:
                    new_config['inputMask_{}'.format(s_idx)] = 0
                
                self.points_to_evaluate.append(new_config)

                

    def _conduct(self,):
        
        func_data = Opt()
        # func_data.logger = self.logger
        func_data.merge(self,['hyper', 'import_path', 'class_name', 'trainer_module'])
        func_data.merge(self.subPack, ['train_set', 'val_set', 'sig_len'])
        odd_check = True if 'odd_check' in self.tuner.dict and self.tuner.odd_check else False
        func_data.odd_check = odd_check
        
        # ray.init(num_cpus=self.tuner.num_cpus)
        os.environ['RAY_COLOR_PREFIX'] = '1'
        ray.init()
        sched = ASHAScheduler(time_attr='training_iteration', max_t=self.tuner.max_training_iteration, grace_period= self.tuner.min_training_iteration) if self.using_sched else None
        # self.tuner.num_samples = 80
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(maskTuningCell, data=func_data), 
                resources=self.resource),
            param_space=self.tuning.dict,
            tune_config=
            tune.TuneConfig(
            search_alg=self.algo_func,
            metric='best_val_acc',
            mode="max",
            num_samples=self.tuner.num_samples,
            scheduler=sched,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
            chdir_to_trial_dir = False,
            # name=self.algo_name,
            # resources_per_trial=self.resource,
            # verbose=1,
            # raise_on_failed_trial = False
            ),
            run_config=RunConfig(
                name=self.algo_name,
                storage_path=self.tuner.dir,
                verbose=1,
                failure_config=FailureConfig(max_failures=self.tuner.num_samples // 2),
                stop={'training_iteration':self.tuner.max_training_iteration,
                      'stop_counter': self.hyper.patience},
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute ='val_acc',
                    checkpoint_score_order='max',
                    checkpoint_frequency= 1
                )      
            )
        )
        
        results = tuner.fit() 

        df = results.get_dataframe()
        df.to_csv(os.path.join(self.tuner.dir, '{}.trial.csv'.format(self.algo_name)))
        ray.shutdown()
        
        # best_result = results.get_best_result(self.metric, 'max', scope='all')
        # self.best_config.merge(best_result.config)
        # self.best_result = best_result.metrics
        # self.best_checkpoint_path = os.path.join(best_result.checkpoint.path, 'model.pth')
      
    

class maskTuningCell(tune.Trainable):
    '''
    Trainable class of tuning stochastic neural network
    see https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#the-train-function
    '''
    def setup(self, config, data=None):
        self.train_data = data.train_set
        self.valid_data = data.val_set
        self.base_hyper = data.hyper
        self.import_path = data.import_path
        self.class_name = data.class_name
        # self.logger = data.logger
        self.odd_check = data.odd_check
        
        logger_path = os.path.join(self.logdir, '{}.log'.format(self.class_name))
        self.logger = set_logger(logger_path, f'{self.class_name}.tuning', rewrite=False)
        
        
        _hyper = Opt()
        _hyper.merge(self.base_hyper)
        _hyper.update(config) # Using ignore_unk will be very risky
        
        
        self.sample_hyper = _hyper
        self.num_workers = 0 if 'num_workers' not in _hyper.dict else  _hyper.num_workers
        
        for i in range(_hyper.sig_len):
            if 'inputMask_{}'.format(i) not in _hyper.dict:
                raise ValueError('Missing hyper config: "inputMask_{}"!'.format(i))
            
        self.inputMask = [_hyper.dict['inputMask_{}'.format(i)] for i in range(data.sig_len)]
        
        self.sample_hyper.mask_num = mask_count(self.inputMask)
        self.sample_hyper.sig_len = data.sig_len - self.sample_hyper.mask_num
        self.sample_hyper.mask_ratio = self.sample_hyper.mask_num / data.sig_len  
        
        self.mask_constrain()
        
        train_set = mask_cutout(data.train_set, self.inputMask)
        val_set = mask_cutout(data.val_set, self.inputMask)
        
        train_loader, val_loader = set_fitset(batch_size=_hyper.batch_size, train_set=train_set, val_set=val_set)

        self.logger.critical("Loading mask operator.")
        self.logger.critical('Overall Mask-ratio is: {:.2f}% \t Mask-num is: {} \t Remaining sig_len is: {}'.format(self.sample_hyper.mask_ratio * 100, self.sample_hyper.mask_num, self.sample_hyper.sig_len))
        self.logger.critical('-'*80)
        
        self.logger.critical('Loading training set and validation set.')
        self.logger.info(f"Train_loader batch: {len(train_loader)}")
        self.logger.info(f"Val_loader batch: {len(val_loader)}")
        self.logger.critical('>'*40)
        
        _model = importlib.import_module(self.import_path)
        _model = getattr(_model, self.class_name)
        sample_model = _model(self.sample_hyper, self.logger)
        
        self.logger.critical('Loading Model.')
        self.logger.critical(f'Model: \n{str(sample_model)}')
        self.logger.info(">>> Total params: {:.2f}M".format(
                    sum(p.numel() for p in list(sample_model.parameters())) / 1000000.0))
        self.logger.critical('Start fit.')
            
        trainer = importlib.import_module(data.trainer_module[0])
        trainer = getattr(trainer, data.trainer_module[1])
        
        self.trainer = trainer(sample_model, train_loader, val_loader, self.base_hyper, self.logger)
        # self.trainer.cfg.patience = self.trainer.cfg.epochs # Actually, in TuningCell, patience does not work as the same as in trainer.
        self.trainer.before_train()
    
    def mask_constrain(self, max_ratio = 0.5):
        '''If mask_ratio > max_ratio, random select mask idx to set as 1 to make mask_ratio be decresed to max_ratio
        '''
        total_sig_len = self.sample_hyper.mask_num + self.sample_hyper.sig_len
        
        if self.sample_hyper.mask_num == 0:
            rev_ids = sample([i for i in range(total_sig_len)], 2)
            for id in rev_ids:
                self.inputMask[id] = 0
            
            self.sample_hyper.mask_num += 2
            self.sample_hyper.sig_len -= 2
            self.sample_hyper.mask_ratio = self.sample_hyper.mask_num / total_sig_len 
        
        if self.odd_check:
            if self.sample_hyper.mask_num % 2 != 0:
                pso_ids = [i for i,v in enumerate(self.inputMask) if v == 1]
                rev_id = sample(pso_ids, 1)[0]
                self.inputMask[rev_id] = 0
                self.sample_hyper.dict['inputMask_{}'.format(rev_id)] = 0
                
                self.sample_hyper.mask_num += 1
                self.sample_hyper.sig_len -= 1
                self.sample_hyper.mask_ratio = self.sample_hyper.mask_num / total_sig_len 
        
        max_num = int(total_sig_len * max_ratio)
        if self.sample_hyper.mask_ratio > max_ratio:
            neg_ids = [i for i,v in enumerate(self.inputMask) if v == 0]
            fill_num = self.sample_hyper.mask_num - max_num
            rev_ids = np.random.choice(neg_ids, size = fill_num, replace = False).tolist()
            
            for id in rev_ids:
                self.inputMask[id] = 1
                self.sample_hyper.dict['inputMask_{}'.format(id)] = 1
            
            # update sample_hyper
            self.inputMask = [self.sample_hyper.dict['inputMask_{}'.format(i)] for i in range(total_sig_len)]
            
            self.sample_hyper.mask_num = mask_count(self.inputMask)
            self.sample_hyper.sig_len = total_sig_len - self.sample_hyper.mask_num
            self.sample_hyper.mask_ratio = self.sample_hyper.mask_num / total_sig_len  
        
        
            
    
    def step(self,):
        self.trainer.before_train_step()
        _, t_acc = self.trainer.run_train_step(ray=True)
        self.trainer.after_train_step()
        self.trainer.before_val_step()
        _, v_acc = self.trainer.run_val_step(ray=True)
        self.trainer.after_val_step(checkpoint = False)
        self.trainer.iter += 1      
        
        best_acc =  self.trainer.best_monitor
        
        return {
            'mask_ratio': self.sample_hyper.mask_ratio,
            'tra_acc': t_acc,
            'val_acc': v_acc,
            'best_val_acc': best_acc * 100, 
            'stop_counter': self.trainer.early_stopping.counter,
        }
    
    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.trainer.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.trainer.model.load_state_dict(torch.load(checkpoint_path))        
        

   