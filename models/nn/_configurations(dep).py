from models.nn._baseNet import BaseNetConfig

class MCDformer_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/MCDformer.py'
        self.class_name = 'MCDformer'

        self.hyper.gamma = 0.1
        self.hyper.extend_channel = 36
        self.hyper.latent_dim = 256
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [1, 36, 64, 128, 256]

class CTDNN_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/CTDNN.py'
        self.class_name = 'CTDNN'

        self.hyper.gamma = 0.5
        self.hyper.batch_size = 400
        self.hyper.lr = 1e-3
        self.hyper.epochs = 100
        self.hyper.encoder_emb = 256
        self.hyper.num_heads = 2
        self.hyper.mlp_ratio = 2
        self.hyper.depth = 2
        self.hyper.patch_size = (2,8)

class AMCNet_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/AMC_Net.py'
        self.class_name = 'AMC_Net'

        self.hyper.gamma = 0.1
        self.hyper.extend_channel = 36
        self.hyper.num_heads = 2
        self.hyper.conv_chan_list = [36, 64, 128, 256]


class AWN_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/AWN.py'
        self.class_name = 'AWN'
        # self.trainer_module = (self.import_path, 'AWN_Trainer')

        self.hyper.batch_size = 128
        self.hyper.gamma = 0.5
        self.hyper.num_level = 1
        self.hyper.regu_details = 0.01
        self.hyper.regu_approx = 0.01
        self.hyper.in_channels = 64
        self.hyper.latent_dim = 320


class MCLDNN_config(BaseNetConfig):
    '''Refer to the paper:  'A Spatiotemporal Multi-Channel Learning Framework for Automatic Modulation Recognition.
    '''

    def base_modify(self):
        self.import_path = 'models/nn/MCLDNN.py'
        self.class_name = 'MCLDNN'
        # self.trainer_module = (self.import_path, 'MCLDNN_Trainer')
        self.arch = 'crnn'

        self.hyper.batch_size = 400
        self.hyper.patience = 20
        self.hyper.milestone_step = 2
        self.hyper.gamma = 0.8


class VTCNN2_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/VT_CNN2.py'
        self.class_name = 'VTCNN'

        self.hyper.epochs = 100
        self.hyper.gamma = 0.5


class CLDNN_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/CLDNN.py'
        self.class_name = 'CLDNN'
        self.arch = 'crnn'

        self.hyper.epochs = 100
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1


class PCNN_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/CLDNN.py'
        self.class_name = 'PCNN'

        self.hyper.epochs = 400
        self.hyper.batch_size = 64
        self.hyper.gamma = 0.5
        self.hyper.milestone_step = 1


class DualNet_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/Dual_Net.py'
        self.class_name = 'DualNet'
        self.arch = 'crnn'

        self.hyper.epochs = 200


class ResNet_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = 'models/nn/ResNet.py'
        self.class_name = 'Subsampling_ResNet'

        self.hyper.epochs = 100
        self.hyper.batch_size = 1024
        self.hyper.milestone_step = 1
