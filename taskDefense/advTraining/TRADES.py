import torch
import torch.nn as nn
from models.nn._baseTrainer import EarlyStopping
import torch.nn.functional as F
from taskAttack.attackmethods.gradient.fgsm import PGD
from taskDefense.advTraining.PGD_AT import PGD_AT
from taskAttack.util import bound_pert
from tqdm.auto import tqdm as real_tqdm
from taskRecog.util import logit_acc


class KL_PGD(PGD):
    """
    Refer to the attack algorithm in the paper "TRADES: TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization"\n
    This class implements the PGD attack with KL divergence loss, which is used in TRADES.\n
    It inherits from the PGD class and overrides the forward method to use KL divergence as the loss function.\n
    https://github.com/yaodongyu/TRADES/blob/master/trades.py
    """

    def __init__(
        self,
        model,
        logger,
        epsilon=16 / 255,
        alpha=1.6 / 255,
        epoch=10,
        targeted=False,
        random_start=True,
        norm="l2",
        loss="KL",
        device=None,
        **kwargs,
    ):
        super().__init__(
            model,
            logger,
            epsilon,
            alpha,
            epoch,
            targeted,
            random_start,
            norm,
            loss,
            device,
        )
        self.epoch = epoch
        self.alpha = epsilon / epoch
        self.decay = 0

    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == "KL":
            return nn.KLDivLoss(reduction="mean").to(self.device)
        else:
            raise Exception("Unsupported loss {}".format(loss))

    def init_delta(self, data):
        delta = torch.zeros_like(data).to(self.device)
        delta = 0.001 * delta.normal_(mean=0, std=1)
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        return delta.detach().requires_grad_(True)

    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, 2, T): tensor for input signal
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]  # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device, non_blocking=True)
        label = label.clone().detach().to(self.device, non_blocking=True)

        logits_natural = self.get_logits(data)
        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            x_adv = data + delta
            logits_adv = self.get_logits(self.transform(x_adv, momentum=momentum))

            are_close = torch.allclose(logits_natural, logits_adv, rtol=1e-5, atol=1e-8)
            if are_close:
                self.logger.warning("Adversarial attack failed - logits unchanged")
            a = F.log_softmax(logits_adv, dim=1)
            b = F.softmax(logits_natural, dim=1)

            # Calculate the loss
            loss = self.get_loss(a, b)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, momentum, self.alpha * 2)

        delta = bound_pert(delta, self.epsilon, self.norm)
        return delta.detach()

        # if self.bound != 'epsilon':
        #     from taskAttack.util import signal_energy, signal_db2pow
        #     sig_energy = signal_energy(signals).mean()
        #     # sig_energy = sig_energy
        #     # print('Signal Power --> Max:{:.2E}\tMin:{:.2E}\tAvg:{:.2E}\tStd:{:.2E}'.format(sig_power.max(),sig_power.min(),sig_power.mean(),sig_power.std()))
        #     if self.bound == 'pnr':
        #         PSR, PNR = self.pnr - snr, self.pnr
        #     elif self.bound == 'psr':
        #         PSR, PNR = self.psr, self.psr + snr
        #     gain = signal_db2pow(PSR)
        #     pert_norm = torch.sqrt(sig_energy * gain)
        #     self.epsilon = pert_norm
        #     self.alpha = self.epsilon / self.epoch * 2
        #     self.logger.critical('Update pert. bound (dB) with PSR {} <->  PNR {} - SNR {} <-> Epsilon {:.2g} <-> Alpha {:.2g}'.format(PSR, PNR, snr, self.epsilon, self.alpha))


class TRADES(PGD_AT):
    def before_train(self):
        super().before_train()
        # 使用标准交叉熵损失
        # self.criterion = nn.CrossEntropyLoss().to(self.hyper.device)
        # self.cel_pointwise = nn.CrossEntropyLoss(
        # reduction='none').to(self.hyper.device)  #return loss for each sample
        # KL散度损失用于TRADES的正则化项
        self.kll = nn.KLDivLoss(reduction="mean").to(self.hyper.device)
        # self.attacker = PGD(model=self.model, logger=logger)
        # 不早停
        # self.early_stopping = EarlyStopping(self.logger, patience=1000)
        # TRADES的正则化参数，可以根据需要调整
        self.beta = 1.0  # 权衡自然准确性和鲁棒性的参数

    def cal_loss_acc(self, sig_batch, lab_batch):
        # TRADES损失计算
        device = self.hyper.device


        # ori_loss = self.cal_ori_loss(X, y)
        logits_natural, ori_loss = self.sig_logits_loss(sig_batch, lab_batch)
        # logits_natural = self.model.logits(X)
        # 对抗样本生成
        # 在TRADES中，我们是针对KL散度而不是交叉熵损失生成对抗样本
        # 因此需要修改攻击方式
        # 这里我们使用已有的PGD攻击器，但注意：实际上TRADES的攻击目标是最大化KL散度
        # 这里为简化，我们仍使用原有PGD攻击器生成对抗样本

        X_detached = sig_batch.detach().clone().requires_grad_(True).to(device)
        y_detached = lab_batch.detach().clone().to(device)

        # 获取对抗扰动
        self.update_psr_epsilon(X_detached)
        delta = self.attacker(X_detached, y_detached)
        delta = delta.detach()
        self.model.train()  # 确保模型在训练模式下
        if torch.isnan(delta).any():
            # self.logger.error("NaN detected in delta! Setting NaNs to 0")
            delta = delta.nan_to_num()

        # 生成对抗样本
        adv_X = X_detached + delta.to(device)

        # 计算对抗样本的输出
        logits_adv = self.model.logits(adv_X)

        # TRADES损失：自然损失 + beta * KL散度损失
        # KL散度损失：对抗样本的预测与自然样本的预测之间的差异
        a = F.log_softmax(logits_adv, dim=1)
        b = F.softmax(logits_natural, dim=1)
        loss_robust = self.kll(a, b)

        # 总损失

        adv_loss = ori_loss + self.beta * loss_robust

        # 准确率计算保持不变

        nat_acc = logit_acc(logits_natural, lab_batch)
        adv_acc = logit_acc(logits_adv, lab_batch)

        return adv_loss, ori_loss, adv_acc, nat_acc

    def run_train_batch(self, data_batch, warmup=False):
        sig_batch, lab_batch = data_batch[0].to(self.hyper.device, non_blocking=True), data_batch[1].to(self.hyper.device, non_blocking=True)
        if warmup:
            nat_logits, nat_loss = self.sig_logits_loss(sig_batch, lab_batch)
            nat_acc = logit_acc(nat_logits, lab_batch)
            self.train_nat_acc.update(nat_acc)
            self.optimizer.zero_grad()
            nat_loss.backward()
            self.train_loss.update(0.0)
            self.train_adv_acc.update(0.0)
        else:
            self.optimizer.zero_grad()

            self.attacker = KL_PGD(
                model=self.model, logger=self.logger, **self.attacker_opts.dict
            )
            adv_loss, nat_loss, adv_acc, nat_acc = self.cal_loss_acc(sig_batch, lab_batch)
            self.train_loss.update(adv_loss.item())
            self.train_adv_acc.update(adv_acc)
            self.train_nat_acc.update(nat_acc)

            adv_loss.backward()

        self.optimizer.step()

        metric_dict = {
            "train_loss_adv": self.train_loss.avg,
            "train_acc_adv": self.train_adv_acc.avg,
            "train_acc_nat": self.train_nat_acc.avg,
        }
        return metric_dict

    def warmup_train_step(self, data_batch):
        metric_dict = self.run_train_batch(data_batch, warmup=True)
        return metric_dict

    def run_val_step(self):
        """
        run validation step, return average adversarial loss and accuracy of adversarial validation examples at this epoch
        """
        self.attacker = KL_PGD(
            model=self.model, logger=self.logger, **self.attacker_opts.dict
        )
        with real_tqdm(
            total=len(self.val_loader),
            desc=f"Epoch {self.iter}/{self.hyper.epochs}",
            postfix=dict,
            mininterval=0.3,
            colour="blue",leave=False
        ) as pbar:
            for step, data_batch in enumerate(self.val_loader):
                sig_batch, lab_batch = (
                    data_batch[0].to(self.hyper.device, non_blocking=True),
                    data_batch[1].to(self.hyper.device, non_blocking=True),
                )

                self.update_psr_epsilon(sig_batch)
                delta = self.attacker(sig_batch, lab_batch)
                if torch.isnan(delta).any():
                    self.logger.error("NaN detected in delta! Setting NaNs to 0")
                    delta = delta.nan_to_num()
                with torch.no_grad():
                    adv_X = sig_batch + delta.to(self.hyper.device)
                    adv_logits, adv_loss = self.sig_logits_loss(adv_X, lab_batch)
                    adv_acc = logit_acc(adv_logits, lab_batch)
                    nat_acc = self.sig_acc(sig_batch, lab_batch)

                self.val_loss.update(adv_loss.item())
                self.val_adv_acc.update(adv_acc)
                self.val_nat_acc.update(nat_acc)

                metric_dict = {
                    "val_loss_adv": self.val_loss.avg,
                    "val_acc_adv": self.val_adv_acc.avg,
                    "val_acc_nat": self.val_nat_acc.avg,
                }

                pbar.set_postfix(**metric_dict)
                pbar.update(1)

        return metric_dict

# uncomment the following lines to make unit test for trades adversarial training, and set the model and dataset in the main function below.
# from models.nn.AWN import AWN_config as awn
if __name__ == "__main__":
    from taskDefense.Parser import get_parser
    from taskDefense.Wrapper import Task

    args, parser = get_parser(parsing=True)

    args.psr = -10
    args.cuda = True
    args.warmup_epoch = 2

    args.snr=[10]
    args.gid = 2
    # args.exp_name = f'defense.psr{args.psr}.warm{args.warmup_epoch}.{args.defense_method}'
    args.data = 'dr2'
    args.batch_size = 4000
    args.test = True
    args.defense_method = 'trades'

    args.model = 'awn'
    try:
        task = Task(args, parser)
        task.conduct(eval=True, ave_confMax=False, show_variance=False)
    except Exception as e:
        print(f"{e}")