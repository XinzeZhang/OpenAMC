import torch
import torch.nn as nn
from models.nn.AWN import AWN as AWN_base
from models.nn.AWN import AWN_config
from models.nn.MCDformer import TinyMLP
from models.nn._baseNet import BaseNetConfig, BaseNet

from models.nn.ResNet import ResNet_config,Subsampling_ResNet
from models.nn.VT_CNN import VTCNN_config,VTCNN
from models.nn.CLDNN import CLDNN_config,CLDNN
from models.nn.CTDNN import CTDNN, CTDNN_config
from models.nn.MCLDNN import MCLDNN, MCLDNN_config

class FTML_config(BaseNetConfig):
    def hyper_modify(self):
        self.hyper.k_s = 10
        self.hyper.k_n = 40
        self.hyper.ftml_beta = 1.0
        self.hyper.ftml_alpha = 0.6
        self.hyper.ftml = True

class AWN_FRE_config(FTML_config, AWN_config):
    def base_modify(self):
        super().base_modify()
        self.import_path = 'taskDefense/advTraining/FRE/defense_net.py'
        self.class_name = 'AWN_FRE'

class ResNet_FRE_config(FTML_config, ResNet_config):
    def base_modify(self):
        super().base_modify()
        self.import_path = 'taskDefense/advTraining/FRE/defense_net.py'
        self.class_name = 'ResNet_FRE'

class VTCNN_FRE_config(FTML_config, VTCNN2_config):
    def base_modify(self):
        super().base_modify()
        self.import_path = 'taskDefense/advTraining/FRE/defense_net.py'
        self.class_name = 'VTCNN_FRE'

class CLDNN_FRE_config(FTML_config, CLDNN_config):
    def base_modify(self):
        super().base_modify()
        self.import_path = 'taskDefense/advTraining/FRE/defense_net.py'
        self.class_name = 'CLDNN_FRE'

class CTDNN_FRE_config(FTML_config, CTDNN_config):
    def base_modify(self):
        super().base_modify()
        self.import_path = 'taskDefense/advTraining/FRE/defense_net.py'
        self.class_name = 'CTDNN_FRE'

class MCLDNN_FRE_config(FTML_config, MCLDNN_config):
    def base_modify(self):
        super().base_modify()
        self.import_path = 'taskDefense/advTraining/FRE/defense_net.py'
        self.class_name = 'MCLDNN_FRE'

class ACM2(nn.Module):
    def __init__(self, sig_len):
        super(ACM2, self).__init__()
        self.Im = TinyMLP(sig_len)
        # Define the layers of ACM2 here
        # Example: self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x_init = x.detach().clone()
        r = x[:,0,:] + 1j*x[:,1,:]
        r = torch.fft.fft(r,dim=-1)
        x_fft = torch.zeros_like(x)
        x_fft[:,0,:] = torch.real(r)
        x_fft[:,1,:] = torch.imag(r)
        h = self.Im(x_fft)
        x_h = torch.mul(h[:,0,:], x_fft[:,0,:]) + 1j * torch.mul(h[:,1,:], x_fft[:,1,:])
        x_h = torch.fft.ifft(x_h, dim=-1)
        r_ = torch.zeros_like(x)
        r_[:,0,:] = torch.real(x_h)
        r_[:,1,:] = torch.imag(x_h)
        x = r_ + x_init
        # Implement the forward pass for ACM2
        # Example: x = self.conv1(x)
        return x  # Return the processed signal


class FRE_Base(BaseNet):
    def __init__(self, hyper=None, logger=None):
        super().__init__(hyper, logger)

    def initialize_classifier(self):
        self.classifier = AWN_base(self.hyper, self.logger)
        self.classifier.initialize_arch()

    def initialize_arch(self):
        self.encoder = ACM2(self.hyper.sig_len)
        self.encoder.to(self.hyper.device)
        self.initialize_classifier()
        # self.store_defend_net_weights()
        # self.store_initial_awn_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        x = self.classifier(x)
        return x

    def feature_extract(self, x):
        x = self.encoder(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x

    def logits(self, sample):
        sample = sample.to(self.hyper.device)
        logits = self.forward(sample)
        return logits


    def store_encoder_weights(self):
        """Store current encoder weights for comparison"""
        self.initial_encoder_weights = {}
        for name, param in self.encoder.named_parameters():
            self.initial_encoder_weights[name] = param.data.clone().detach()

    def store_initial_classifier_weights(self):
        """Store initial classifier weights for comparison"""
        self.initial_classifier_weights = {}
        for name, param in self.classifier.named_parameters():
            self.initial_classifier_weights[name] = param.data.clone().detach()

    def check_weights_unchanged(self):
        """Check if weights remain unchanged"""
        if self.initial_encoder_weights is None or self.initial_classifier_weights is None:
            return False

        unchanged = True
        for name, param in self.classifier.named_parameters():
            if not torch.equal(param.data, self.initial_classifier_weights[name]):
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"classifier parameter {name} has changed")
                unchanged = False

        if unchanged and hasattr(self, 'logger') and self.logger:
            self.logger.info("All classifier weights remain unchanged")

        for name, param in self.encoder.named_parameters():
            if not torch.equal(param.data, self.initial_encoder_weights[name]):
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"encoder parameter {name} has changed")
                unchanged = False

        return unchanged

class AWN_FRE(FRE_Base):
    def initialize_classifier(self):
        self.classifier = AWN_base(self.hyper, self.logger)
        self.classifier.initialize_arch()
        # self.store_defend_net_weights()
        # self.store_initial_awn_weights()

    def logits(self, sample):
        sample = sample.to(self.hyper.device)
        logits, _ = self.forward(sample)
        return logits


class ResNet_FRE(FRE_Base):
    def initialize_classifier(self):
        self.classifier = Subsampling_ResNet(self.hyper, self.logger)
        self.classifier.initialize_arch()

class VTCNN_FRE(FRE_Base):
    def initialize_classifier(self):
        self.classifier = VTCNN(self.hyper, self.logger)
        self.classifier.initialize_arch()

class CLDNN_FRE(FRE_Base):
    def initialize_classifier(self):
        self.classifier = CLDNN(self.hyper, self.logger)
        self.classifier.initialize_arch()
        self.has_rnn = True  # Indicate that this model uses RNN layers

class CTDNN_FRE(FRE_Base):
    def initialize_classifier(self):
        self.classifier = CTDNN(self.hyper, self.logger)
        self.classifier.initialize_arch()

class MCLDNN_FRE(FRE_Base):
    def initialize_classifier(self):
        self.classifier = MCLDNN(self.hyper, self.logger)
        self.classifier.initialize_arch()
        self.has_rnn = True  # Indicate that this model uses RNN layers