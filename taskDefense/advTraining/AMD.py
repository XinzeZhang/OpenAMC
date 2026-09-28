import torch
import torch.nn as nn

from taskDefense.advTraining.PGD_AT import PGD_AT, PGD
from models.nn.VGG import VGG16, VGG16_config
from models.nn.MCLDNN import MCLDNN, MCLDNN_config
from taskRecog.util import logit_acc

class Adversarial_Multi_Distillation(PGD_AT):
    '''Chen, Z., Wang, Z., Xu, D., Zhu, J., Shen, W., Zheng, S., Xuan, Q., Yang, X., 2024. Learn to defend: adversarial multi-distillation for automatic modulation recognition models. IEEE Transactions on Information Forensics and Security 1–1. https://doi.org/10.1109/TIFS.2024.3361172
    '''
    def before_train(self):
        super().before_train()
        # 加载教师模型
        self.lambda1 = 1.0  # Lkd1 的权重
        self.lambda2 = 1.0  # Lnat 的权重
        self.lambda3 = 1.0  # Ladv 的权重
        self.lambda4 = 1.0  # Lkd2 的权重

        # 定义MSE损失函数用于知识蒸馏
        self.mse_loss = nn.MSELoss()

        self.adv_teacher = self.load_vgg16()
        self.acc_teacher = self.load_mcldnn()
        # self.load_adv_teacher()

    def cal_loss_acc(self, sig_batch, lab_batch):
        """
        计算AMD框架的总损失
        """
        # self.check_model_weights_for_nan()
        device = self.hyper.device
        # 1. 对于自然样本的预测
        logits_nat, Lnat = self.sig_logits_loss(sig_batch, lab_batch)

        logits_nat = self.model.logits(sig_batch)
        with torch.no_grad():
            logits_T_acc = self.acc_teacher.logits(sig_batch)

        # Lnat: 自然样本的分类损失
        # Lnat = self.cal_ori_loss(sig_batch, lab_batch)
        # if torch.isnan(Lnat).any() or torch.isinf(Lnat).any():
        #     self.logger.error("NaN/Inf in Lnat! Setting to 0")
        #     Lnat = torch.tensor(0.0, device=device, requires_grad=True)

        # Check for NaN/Inf in inputs before MSE loss
        # if torch.isnan(logits_nat).any() or torch.isnan(logits_T_acc).any():
        #     self.logger.error("NaN detected in logits_nat or logits_T_acc before Lkd1!")
        #     logits_nat = torch.nan_to_num(logits_nat, nan=0.0, posinf=1e6, neginf=-1e6)
        #     logits_T_acc = torch.nan_to_num(logits_T_acc, nan=0.0, posinf=1e6, neginf=-1e6)

        # Lkd1: 自然样本的知识蒸馏损失
        Lkd1 = self.mse_loss(logits_nat, logits_T_acc)
        # if torch.isnan(Lkd1).any() or torch.isinf(Lkd1).any():
        #     self.logger.error("NaN/Inf in Lkd1! Setting to 0")
        #     Lkd1 = torch.tensor(0.0, device=device, requires_grad=True)

        X_detached = sig_batch.detach().clone().requires_grad_(True)
        y_detached = lab_batch.detach().clone()
        self.update_psr_epsilon(X_detached)
        delta = self.attacker(X_detached, y_detached)
        delta = delta.detach()
        self.model.train()

        if torch.isnan(delta).any():
            # self.logger.error("NaN detected in delta! Setting NaNs to 0")
            delta = delta.nan_to_num()

        # self.model.train()
        adv_sig_batch = X_detached + delta.to(device)

        # 2. 对于对抗样本的预测
        adv_sig_batch = adv_sig_batch.detach()
        logits_adv = self.model.logits(adv_sig_batch)
        # Check for NaN/Inf in adversarial logits
        # if torch.isnan(logits_adv).any() or torch.isinf(logits_adv).any():
        #     self.logger.error("NaN/Inf detected in logits_adv!")
        #     logits_adv = torch.nan_to_num(logits_adv, nan=0.0, posinf=1e6, neginf=-1e6)

        with torch.no_grad():
            logits_T_adv = self.adv_teacher.logits(adv_sig_batch)
            # if torch.isnan(logits_T_adv).any() or torch.isinf(logits_T_adv).any():
            #     self.logger.error("NaN/Inf detected in logits_T_adv!")
            #     logits_T_adv = torch.nan_to_num(logits_T_adv, nan=0.0, posinf=1e6, neginf=-1e6)


        # 计算四个损失项

        logits_adv, Ladv = self.sig_logits_loss(adv_sig_batch, lab_batch)
        # Ladv: 对抗样本的分类损失
        # if torch.isnan(Ladv).any() or torch.isinf(Ladv).any():
        #     self.logger.error("NaN/Inf in Ladv! Setting to 0")
        #     Ladv = torch.tensor(0.0, device=device, requires_grad=True)

        # Lkd2: 对抗样本的知识蒸馏损失
        # Check teacher adversarial logits
        Lkd2 = self.mse_loss(logits_adv, logits_T_adv)
        # if torch.isnan(Lkd2).any() or torch.isinf(Lkd2).any():
        #     self.logger.error("NaN/Inf in Lkd2! Setting to 0")
        #     Lkd2 = torch.tensor(0.0, device=device, requires_grad=True)

        # self.logger.info(f"Lkd1: {Lkd1.item()}, Lnat: {Lnat.item()}, Ladv: {Ladv.item()}, Lkd2: {Lkd2.item()}")

        # 计算总损失
        total_loss = self.lambda1 * Lkd1 + self.lambda2 * Lnat + self.lambda3 * Ladv + self.lambda4 * Lkd2

        # 计算准确率
        adv_acc = logit_acc(logits_adv, lab_batch)
        nat_acc = logit_acc(logits_nat, lab_batch)

        return total_loss, Lnat, adv_acc, nat_acc


    def run_train_batch(self, data_batch, warmup=False):
        sig_batch, lab_batch = data_batch[0].to(self.hyper.device,  non_blocking=True), data_batch[1].to(self.hyper.device,  non_blocking=True)
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
            self.attacker = PGD(model=self.model, logger=self.logger, **self.attacker_opts.dict)

            adv_loss, nat_loss, adv_acc, nat_acc = self.cal_loss_acc(sig_batch, lab_batch)
            self.train_loss.update(adv_loss.item())
            self.train_adv_acc.update(adv_acc)
            self.train_nat_acc.update(nat_acc)

            with torch.autograd.detect_anomaly():
                adv_loss.backward()

        self.optimizer.step()

        metric_dict = {
            "train_loss_adv": self.train_loss.avg,
            "train_acc_adv": self.train_adv_acc.avg,
            "train_acc_nat": self.train_nat_acc.avg,
        }
        return metric_dict


    def load_vgg16(self):
        config = VGG16_config()
        config.hyper.merge(self.model.hyper)

        model = VGG16(hyper=config.hyper, logger=self.logger)
        state_path = f'checkpoints/{self.hyper.data_name}/nature/{self.hyper.data_name}_vgg.best.pt'
        state_dict = torch.load(state_path, map_location=config.hyper.device)

        # 如果加载的是完整的checkpoint（包含optimizer状态等）
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        # 如果直接保存的是模型权重
        else:
            model.load_state_dict(state_dict)

        model.eval()
        return model

    def load_mcldnn(self):
        config = MCLDNN_config()
        config.hyper.merge(self.model.hyper)

        model = MCLDNN(hyper=config.hyper, logger=self.logger)
        state_path = f'checkpoints/{self.hyper.data_name}/nature/{self.hyper.data_name}_mcl.best.pt'
        state_dict = torch.load(state_path, map_location=config.hyper.device)

        # 如果加载的是完整的checkpoint（包含optimizer状态等）
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        # 如果直接保存的是模型权重
        else:
            model.load_state_dict(state_dict)

        model.eval()
        return model

    def check_model_weights_for_nan(self):
        """
        Check if any module in self.model has NaN weights
        Returns: (has_nan: bool, nan_modules: dict)
        """
        has_nan = False
        nan_modules = {}

        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:  # Only check modules with parameters
                module_has_nan = False
                module_nan_info = {}

                for param_name, param in module.named_parameters(recurse=False):
                    if torch.isnan(param).any():
                        has_nan = True
                        module_has_nan = True
                        nan_count = torch.isnan(param).sum().item()
                        total_params = param.numel()

                        module_nan_info[param_name] = {
                            'nan_count': nan_count,
                            'total_params': total_params,
                            'percentage': (nan_count / total_params) * 100,
                            'shape': list(param.shape)
                        }

                        self.logger.warning(f"NaN detected in {name}.{param_name}!")
                        self.logger.warning(f"  Shape: {param.shape}")
                        self.logger.warning(f"  NaN count: {nan_count}/{total_params} ({(nan_count/total_params)*100:.2f}%)")

                if module_has_nan:
                    nan_modules[name] = module_nan_info

        if has_nan:
            self.logger.error(f"Total modules with NaN: {len(nan_modules)}")
        # else:
        #     self.logger.info("No NaN weights detected in model")

        return has_nan, nan_modules


# uncomment the following lines to make unit test for adversarial training, and set the model and dataset in the main function below.
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
    args.bs = 65
    args.test = True
    args.defense_method = 'amd'

    args.model = 'awn'
    try:
        task = Task(args, parser)
        task.conduct(eval=True, ave_confMax=False, show_variance=False)
    except Exception as e:
        print(f"{e}")