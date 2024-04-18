import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.path.pardir, os.path.pardir))
from exp_config._baseSetting import AMC_Net_base, AWN_base, mcldnn_base, vtcnn2_base, dualnet_base, resnet_base, cldnn_base, pcnn_base
from ray import tune
import torch
import numpy as np
import pickle
from task.base.TaskLoader import TaskDataset
from task.base.TaskWrapper import Task
from task.TaskParser import get_parser
from data.RML import RML2016_10a_Data

class Data(RML2016_10a_Data):
    def __init__(self, opts):
        super().__init__(opts)
        


class amcnet(AMC_Net_base):
    def task_modify(self):
        self.hyper.extend_channel = 36
        
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_AMC_Net.best.pt'


class awn(AWN_base):
    def task_modify(self):
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_AWN.best.pt'

class dualnet(dualnet_base):
    def task_modify(self):
        self.hyper.epochs = 200
        self.hyper.patience = 15
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_DualNet.best.pt'




class mcl(mcldnn_base):
    def task_modify(self):
        # self.hyper.batch_size = 64
        self.hyper.epochs = 200
        self.hyper.patience = 20
        # self.hyper.gamma = 0.5502
        # self.hyper.lr = 0.0015
        self.hyper.milestone_step = 2
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_MCLDNN.best.pt'

        self.tuner.num_samples = 40
        self.tuner.training_iteration = self.hyper.epochs
        # self.tuner.num_cpus = 32 * 3
        self.tuner.resource = {"gpu": 0.5}
        # self.tuner.points_to_evaluate=[{
        #     'lr':0.001,
        #     'gamma':0.8,
        #     'milestone_step':5,
        #     'batch_size': 400
        # }]
        # self.tuner.using_sched = False
        self.tuning.lr = tune.loguniform(1e-4, 2e-3)
        self.tuning.gamma = tune.uniform(0.5, 0.99)
        self.tuning.milestone_step = tune.qrandint(1, 10, 1)
        self.tuning.batch_size = tune.choice([64, 96, 128, 160, 192])

# resnet_base


class res(resnet_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1
        self.hyper.patience = 20
        # self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_ResNet.best.pt'
        # self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_ResNet.best.bak.pth'
        
        self.tuner.num_samples = 40
        self.tuner.resource = {"gpu": 0.5}
        self.tuning.lr = tune.loguniform(1e-4, 2e-3)
        self.tuning.gamma = tune.uniform(0.5, 0.99)
        self.tuning.milestone_step = tune.qrandint(1, 10, 1)
        self.tuning.batch_size = tune.qrandint(64, 1024, 64)

class cldnn(cldnn_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1
        self.hyper.patience = 20
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_CLDNN.best.pt'
        
        self.tuner.num_samples = 60
        self.tuner.resource = {"gpu": 0.33}
        self.tuning.lr = tune.loguniform(1e-4, 2e-3)
        self.tuning.gamma = tune.uniform(0.5, 0.99)
        self.tuning.milestone_step = tune.qrandint(1, 5, 1)
        self.tuning.batch_size = tune.qrandint(64, 1024, 64)

class vtcnn(vtcnn2_base):
    def task_modify(self):
        self.hyper.epochs = 100
        self.hyper.patience = 10
        self.hyper.gamma = 0.5
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_VTCNN.best.pt'

        self.tuner.resource = {"gpu": 0.5}  # set this for GPUs
            

class pcnn(pcnn_base):
    def task_modify(self):
        self.hyper.epochs = 400
        self.hyper.batch_size = 64
        self.hyper.gamma = 0.5
        self.hyper.patience = 20
        self.hyper.milestone_step = 1
        self.hyper.pretraining_file = 'data/RML2016.10a/pretrain_models/RML2016.10a_PCNN.best.pt'
        
if __name__ == "__main__":
    args = get_parser()
    args.cuda = True
    # args.gid = 2

    args.exp_config = os.path.dirname(sys.argv[0]).replace(os.getcwd()+'/', '')
    args.exp_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    args.exp_name = 'Baselines'
    # args.exp_name = 'cldnn.test'
    # args.exp_name = 'pcnn.test'
    # args.force_update = True # if need to rerun the model fiting or reload the pretraining_file (if os.path.exit(hyper.pretraining_file) to get the results, uncomment this line.)

    args.test = True
    args.clean = True
    args.model = 'awn'

    task = Task(args)
    # task.tuning()
    task.conduct()
    # task.evaluate()

    # for model in ['amcnet', 'awn', 'cldnn', 'dualnet', 'pcnn', 'res']:
    #     args.model = model
    #     task = Task(args)
    #     task.evaluate()