from task.TaskLoader import Opt
from ray import tune


class nn_base(Opt):
    def __init__(self):
        super().__init__()
        
        self.arch = 'torch_nn'
        
        self.hyper = Opt()
        self.tuner = Opt()
        self.tuning = Opt()

        self.hyper_init()
        self.tuner_init()

        self.base_modify()
        self.hyper_modify()

        self.tuning_modify()
        self.task_modify()
        self.ablation_modify()
        
        self.common_process()

    def hyper_init(self,):
        self.hyper.epochs = 100
        self.hyper.patience = 10
        self.hyper.milestone_step = 3
        self.hyper.gamma = 0.5
        self.hyper.lr = 0.001

    def tuner_init(self,):
        # total cpu cores for tuning
        self.trainer_path = 'models/_comTrainer.py'
        self.trainer_name = 'Trainer'
        self.tuner.resource = {
            "cpu": 10,
            "gpu": 1  # set this for GPUs
        }
        # gpu cards per trial in tune
        # tuner search times
        self.tuner.num_samples = 20
        # fitness epoch per iter
        self.tuner.epochPerIter = 50
        # self.tuner.algo = 'rand'

    def base_modify(self,):
        pass

    def hyper_modify(self,):
        pass

    def tuning_modify(self):
        pass

    def ablation_modify(self):
        pass

    def task_modify(self):
        pass


    def common_process(self,):
        if "import_path" in self.dict:
            self.import_path = self.import_path.replace(
            '.py', '').replace('/', '.')
        if "trainer_path" in self.dict:
            self.trainer_path = self.trainer_path.replace(
                    '.py', '').replace('/', '.')            



# class mlp_base(nn_base):
#     def base_modify(self):
#         self.import_path = 'models/training/MLP.py'
#         self.class_name = 'mlp'
    
#     def hyper_init(self):
#         self.hyper.hidden_size = 400
#         self.hyper.epochs = 1000
#         self.hyper.learning_rate = 0.01
#         self.hyper.step_lr = 20

class AMC_Net_base(nn_base):
    def base_modify(self):
        self.import_path = 'models/AMC_Net.py'
        self.class_name = 'AMC_Net'
    def hyper_modify(self):        
        self.hyper.gamma = 0.1

        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 512
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]
        
class AWN_base(nn_base):
    def base_modify(self) -> None:
        self.import_path = 'models/AWN.py'
        self.class_name = 'AWN'
        self.trainer_path = 'models/AWN.py'
        self.trainer_name = 'AWN_Trainer'
        
    def hyper_modify(self):        
        self.hyper.batch_size = 128
        self.hyper.gamma = 0.5
        
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.kernel_size = 3
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320

    

class mcldnn_base(nn_base):
    def base_modify(self) -> None:
        self.import_path = 'models/mcldnn.py'
        self.class_name = 'MCLDNN'
        
class vtcnn2_base(nn_base):
    def base_modify(self) -> None:
        self.import_path = 'models/VT_CNN2.py'
        self.class_name = 'VTCNN'
        
class cldnn_base(nn_base):
    def base_modify(self) -> None:
        self.import_path = 'models/CLDNN.py'
        self.class_name = 'CLDNN'