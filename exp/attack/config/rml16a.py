# import os,sys
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from models.nn.AMC_Net import AMCNet_config
from models.nn.AWN import AWN_config
from models.nn.CLDNN import CLDNN_config
from models.nn.MCLDNN import MCLDNN_config
from models.nn.Dual_Net import DualNet_config
from models.nn.VGG import VGG16_config
from models.nn.VT_CNN import VTCNN_config
from models.nn.ResNet import ResNet_config
from models.nn.MCDformer import MCDformer_config
from models.nn.CTDNN import CTDNN_config
from models.nn.MsmcNet import MsmcNet_config


class amcnet(AMCNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_amcnet.best.pt'

class awn(AWN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_awn.best.pt'
        # self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_awn.PGD_AT.pt'
        # self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_awn.best.pt'

class cldnn(CLDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_cldnn.best.pt'

class ctdnn(CTDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_ctdnn.best.pt'

class msmc(MsmcNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_msmc.best.pt'

class dualnet(DualNet_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_dualnet.best.pt'


class mcd(MCDformer_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_mcd.best.pt'

class mcl(MCLDNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_mcl.best.pt'


class res(ResNet_config):
    def task_modify(self):
        self.hyper.pretraining_file ='checkpoints/RML2016.10a/nature/RML2016.10a_res.best.pt'

class vgg(VGG16_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_vgg.best.pt'

class vtcnn(VTCNN_config):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/nature/RML2016.10a_vtcnn.best.pt'


# under PGD_AT
class amcnet_amd(amcnet):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_amcnet.best.pt'

class awn_amd(awn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_awn.best.pt'

class cldnn_amd(cldnn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_cldnn.best.pt'

class ctdnn_amd(ctdnn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_ctdnn.best.pt'

class dualnet_amd(dualnet):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_dualnet.best.pt'

class mcd_amd(mcd):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_mcd.best.pt'

class mcl_amd(mcl):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_mcl.best.pt'

class res_amd(res):
    def task_modify(self):
        self.hyper.pretraining_file ='checkpoints/RML2016.10a/AMD/RML2016.10a_res.best.pt'

class vgg_amd(vgg):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_vgg.best.pt'

class vtcnn_amd(vtcnn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/AMD/RML2016.10a_vtcnn.best.pt'


# under checkpoints/RML2016.10a/advTraining.AMD
class amcnet_xamd(amcnet):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_amcnet.best.pt'

class awn_xamd(awn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_awn.best.pt'

class cldnn_xamd(cldnn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_cldnn.best.pt'

class ctdnn_xamd(ctdnn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_ctdnn.best.pt'

class dualnet_xamd(dualnet):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_dualnet.best.pt'

class mcd_xamd(mcd):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_mcd.best.pt'

class mcl_xamd(mcl):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_mcl.best.pt'

class res_xamd(res):
    def task_modify(self):
        self.hyper.pretraining_file ='checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_res.best.pt'

class vgg_xamd(vgg):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_vgg.best.pt'

class vtcnn_xamd(vtcnn):
    def task_modify(self):
        self.hyper.pretraining_file = 'checkpoints/RML2016.10a/advTraining.AMD/RML2016.10a_vtcnn.best.pt'




# class awn_ftml(AWN_config):
#     def task_modify(self):
#         self.hyper.pretraining_file = 'data/RML2016.10a/test_model/awn_ftml_best/model.pth'

# from models.nn.ftml_defend_net import AWN_w_config
# class awn_w(AWN_w_config):
#     def task_modify(self):
#         self.hyper.pretraining_file = 'data/RML2016.10a/test_model/awn_w_best/model.pth'
