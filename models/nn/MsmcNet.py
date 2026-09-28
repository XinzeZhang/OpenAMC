import torch
import torch.nn as nn
from models.nn._baseNet import BaseNet, BaseNetConfig


class MsmcNet_config(BaseNetConfig):
    def base_modify(self):
        self.import_path = "models/nn/MsmcNet.py"
        self.class_name = "MsmcNet"

        self.hyper.epochs = 100
        self.hyper.gamma = 0.5
        # self.hyper.num_classes = 11


class MsmcNet(BaseNet):
    """
    Wang, Y., Bai, J., Xiao, Z., Zhou, H., Jiao, L., 2022. MsmcNet: a modular few-shot learning framework for signal modulation classification. IEEE Trans. Signal Process. 70, 3789–3801. https://doi.org/10.1109/TSP.2022.3191783
    """

    def initialize_arch(self):
        # num_classes = self.hyper.num_classes
        num_classes = self.hyper.num_classes
        self.Block_1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(2, 5), stride=(2, 1), padding=(1, 2)),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.Block_2 = nn.Sequential(
            nn.Conv2d(30, 25, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        self.Block_3 = nn.Sequential(
            nn.Conv2d(25, 15, kernel_size=(1, 5), stride=1, padding=0),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        if self.hyper.sig_len < 2048:
            self.Block_4 = nn.Sequential(
                nn.Conv2d(15, 15, kernel_size=(1, 5), stride=1, padding=0),
                nn.BatchNorm2d(15),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        else:
            target_len = 128
            k = self.hyper.sig_len // target_len // 2
            self.Block_4 = nn.Sequential(
                nn.Conv2d(15, 15, kernel_size=(1, 5), stride=1, padding=0),
                nn.BatchNorm2d(15),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(15, 15, kernel_size=(1, k), stride=1, padding=0),
                nn.ReLU(),
                nn.BatchNorm2d(15),
                nn.Dropout(0.5),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Conv2d(15, 15, kernel_size=(1, k), stride=(1,k), groups=15, bias=False),
                nn.BatchNorm2d(15),
                nn.ReLU(),
                nn.Dropout(0.5),
            )




        self.flatten = nn.Flatten()
        self.flatten_features = self._infer_flatten_features()
        self.fc = nn.Linear(self.flatten_features, num_classes)
        self.to(self.hyper.device)
        # self.fc1 = nn.Linear(75, 128)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(128, num_classes)

    def _infer_flatten_features(self):
        if not hasattr(self.hyper, 'sig_len'):
            raise ValueError('MsmcNet requires hyper.sig_len to infer the classifier input dimension.')

        with torch.no_grad():
            sample = torch.zeros(1, 1, 2, self.hyper.sig_len)
            sample = self.Block_1(sample)
            sample = self.Block_2(sample)
            sample = self.Block_3(sample)
            sample = self.Block_4(sample)
        return sample.numel()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.Block_1(x)
        x = self.Block_2(x)
        x = self.Block_3(x)
        x = self.Block_4(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x

    def get_logits_and_intermediate_features(self, x):
        features = []

        x = torch.unsqueeze(x, 1)
        x = self.Block_1(x)
        features.append(x)
        x = self.Block_2(x)
        features.append(x)
        x = self.Block_3(x)
        features.append(x)
        x = self.Block_4(x)
        features.append(x)
        x = self.flatten(x)
        logit = self.fc(x)

        return logit, features


if __name__ == "__main__":
    """ 测试网络结构构建是否构建正确，并打印每层参数 """
    from torchinfo import summary

    hyper = MsmcNet_config().hyper
    hyper.num_classes = 11
    hyper.sig_len = 1024
    model = MsmcNet(hyper=hyper)
    # model.cuda()
    # print(model)
    # # 统计网络参数及输出大小
    # summary(model, (2, hyper.sig_len), batch_dim=0)
