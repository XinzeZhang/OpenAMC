# import os,sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

# from models.nn._baseNet import BaseNetConfig

from models.nn.AMC_Net import AMCNet_config
from models.nn.AWN import AWN_config
from models.nn.CLDNN import CLDNN_config
from models.nn.MCLDNN import MCLDNN_config
from models.nn.Dual_Net import DualNet_config
from models.nn.VT_CNN import VTCNN_config
from models.nn.ResNet import ResNet_config
from models.nn.MsmcNet import MsmcNet_config

class amcnet(AMCNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_amcnet.best.pt'

class awn(AWN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_awn.best.pt'

class res(ResNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_res.best.pt'

class dualnet(DualNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_dualnet.best.pt'

class mcl(MCLDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_mcl.best.pt'

class cldnn(CLDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_cldnn.best.pt'

class vtcnn(VTCNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_vtcnn.best.pt'

from models.nn.MCDformer import MCDformer_config
class mcd(MCDformer_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_mcd.best.pt'
        self.hyper.epochs = 200
        self.hyper.num_heads = 2
        self.hyper.lr = 0.0036
        self.hyper.gamma = 0.3160
        self.hyper.milestone_step = 9
        self.hyper.batch_size = 128

from models.nn.CTDNN import CTDNN_config
class ctdnn(CTDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_ctdnn.best.pt'

class msmc(MsmcNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10b/nature/RML2016.10b_msmc.best.pt'

class awn_ftml(AWN_config):
    def task_modify(self):
        # self.hyper.pretraining_file = 'yield_test/SignalAug.ICASSP25/RML2016.10a/ftml/fit/awn/checkpoint/RML2016.10a_awn.best.pt'
        self.hyper.pretraining_file = 'yield_results/SignalAug.ICASSP25/RML2016.10b/ftml/fit/awn/checkpoint/RML2016.10b_awn.best.pt'
