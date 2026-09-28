import torch
import torch.nn.functional as F
from taskAttack.util import bound_pert
from taskAttack.attackmethods.gradient.fgsm import IFGSM

class PNGD(IFGSM):
    '''Nasr, M.C.B., de Araujo-Filho, P.F., Kaddoum, G., Mourad, A., 2024. Projected natural gradient method: unveiling low-power perturbation vulnerabilities in deep-learning-based automatic modulation classification. IEEE Internet Things J. 11, 37032–37044. https://doi.org/10.1109/JIOT.2024.3439440

    '''
    def set_specific_params(self,):
        self.lambda_param = 0.8  # λ参数
        self.eta = 2  # η参数


    def get_fisher_information_matrix(self, data, delta):
        """
        根据论文公式计算Fisher信息矩阵的对角近似
        F_diag = E[(∂ln(f)/∂x_i)²]

        Args:
            data: 原始数据 [batch_size, 2, 128]
            delta: 当前扰动 [batch_size, 2, 128]

        Returns:
            fim: Fisher信息矩阵对角项 [batch_size, 2, 128]
        """
        # 当前对抗样本
        x_adv = (data + delta).detach()

        def single_sample_log_f(xi):
            logits = self.model.logits(xi.unsqueeze(0))
            log_probs = F.log_softmax(logits, dim=1)
            cls = logits.argmax(dim=1, keepdim=True)
            return log_probs.gather(1, cls).squeeze()

        try:
            per_sample_grad = torch.func.grad(single_sample_log_f)
            grads = torch.func.vmap(per_sample_grad)(x_adv)
            return grads.detach().pow(2).mean(dim=0)
        except (RuntimeError, NotImplementedError):
            batch_size = x_adv.shape[0]
            fisher_diag = torch.zeros_like(x_adv[0])

            for i in range(batch_size):
                xi = x_adv[i:i+1].clone().requires_grad_(True)
                logits = self.model.logits(xi)
                log_probs = F.log_softmax(logits, dim=1)
                cls = logits.argmax(dim=1, keepdim=True)
                log_f = log_probs.gather(1, cls).squeeze()

                grads = torch.autograd.grad(
                    outputs=log_f,
                    inputs=xi,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                fisher_diag += grads[0].detach().pow(2)

            return fisher_diag / batch_size

    def get_grad(self, loss, delta, **kwargs):
        """
        重写梯度计算方法，使用预处理自然梯度

        参数：
            loss: 计算得到的损失
            delta: 当前扰动

        返回:
            preconditioned_grad: 预处理后的自然梯度
        """
        # 计算原始梯度
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        # 获取Fisher信息矩阵
        data = kwargs.get('data', None)
        if data is None:
            return grad  # 如果没有提供数据，则返回原始梯度

        fim = self.get_fisher_information_matrix(data, delta)

        # 应用公式(7)：∇̃xJ(x, y) = (λIn − F⊙In)^η ∇xJ(x, y)
        # 其中⊙表示元素级乘法，^η表示元素级幂运算
        # identity = torch.ones_like(fim)
        precond_factor = torch.clamp(self.lambda_param - fim, min=1e-8).pow(self.eta)
        # (self.lambda_param * identity - fim * identity) ** self.eta
        preconditioned_grad = precond_factor * grad

        return preconditioned_grad

    def forward(self, data, label, **kwargs):
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # 目标标签

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化对抗性扰动
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # 获取模型输出
            logits = self.get_logits(self.transform(data + delta, momentum=momentum))

            # 计算损失
            loss = self.get_loss(logits, label)

            # 计算预处理自然梯度
            grad = self.get_grad(loss, delta, data=data)

            # 计算动量
            momentum = self.get_momentum(grad, momentum)

            # 更新对抗性扰动
            delta = self.update_delta(delta, momentum, self.alpha)

        return delta.detach()


if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser

    from data import data_zoo


    args, parser = get_parser()

    args.psr = -20
    args.data = 'dr2'
    data_name = data_zoo[args.data]['data_name']

    args.algo = 'pngd'
    args.gid = 0

    # model ='awn'
    # args.surrogate_model = model
    # args.target_model = model
    # model_ckp = f'checkpoints/{data_name}/nature/{data_name}_{model}.best.pt'
    # args.surrogate_ckp = model_ckp
    # args.target_ckp = model_ckp

    args.cuda = True
    args.test = True
    args.clean = True
    args.batch_size = 200 # 12000 per snr

    args.snr = [0]
    model_list = ['awn']
    for model in model_list:
        args.surrogate_model = model
        args.target_model = model

        model_ckp = f'checkpoints/{data_name}/nature/{data_name}_{model}.best.pt'
        args.surrogate_ckp = model_ckp
        args.target_ckp = model_ckp

        task = Attack(args, parser)
        task.conduct()