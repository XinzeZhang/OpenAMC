import torch
import torch.nn.functional as F
from taskAttack.util import bound_pert

from taskAttack.attackmethods._baseAttackAlgo import BaseAttackAlgo

from tqdm import tqdm

class MDAM(BaseAttackAlgo):
    """
    Bai, J., Ge, C., Xiao, Z., Jiang, H., Li, T., Zhou, H., Jiao, L., 2025. A multiscale discriminative attack method for automatic modulation classification. IEEE Trans. Inf. Forensics Security 20, 294–308. https://doi.org/10.1109/TIFS.2024.3515802\n
    Currently, this method only surpports attack on AWN, MCL, MSMC, CTDNN. For other models, the intermediate features remain a func to be extracted, which leads to attack failure. We will fix this issue in the future.
    """
    def set_specific_params(self,):
        self.lambda_threshold = 0.5
        self.mu = 1.0
        self.epoch = 10

    def compute_madl_loss(self, features_clean, features_adv, logits = None, label = None):
        """
        计算Multi-layer Activation Disruption Loss - version corresponding to the original paper
        """
        total_loss = 0

        for i in range(len(features_clean)):
            Vi_clean = features_clean[i]  # 原始特征
            Vi_adv = features_adv[i]      # 对抗特征

            # 打印维度信息用于调试
            # self.logger.debug(f"Layer {i} - Clean features shape: {Vi_clean.shape}, Adv features shape: {Vi_adv.shape}")
            # print(f"Layer {i} - Clean features shape: {Vi_clean.shape}, Adv features shape: {Vi_adv.shape}")

            # 计算支持激活集合
            support_mask = self.compute_support_activations(Vi_clean)

            # 处理不同维度的特征
            Vi_adv_norm = torch.pow(Vi_adv, 2)
            dims_to_sum = tuple(range(1, len(Vi_adv.shape)))

            # 对支持激活和非支持激活分别加权求和
            support_sum = (Vi_adv_norm * support_mask.float()).sum(dim=dims_to_sum).pow(0.5)
            non_support_sum = (Vi_adv_norm * (~support_mask).float()).sum(dim=dims_to_sum).pow(0.5)

            # 计算层损失 L(Vi) - 公式(5)
            epsilon_small = 1e-8
            layer_loss = torch.log(support_sum + epsilon_small) - torch.log(non_support_sum + epsilon_small)

            total_loss += layer_loss.mean()

        return total_loss


    def single_step_attack(self, data, delta, momentum_dict, features_clean, label = None, per_sample = False):
        adv_data = data + delta
        logits, adv_features = self.model.get_logits_and_intermediate_features(adv_data)
        madl_loss = self.compute_madl_loss(features_clean, adv_features, logits, label)

        temp_momentum_dict = {}

        delta_t = torch.zeros_like(delta)
        # pred_idx = logits.argmax(dim=1, keepdim=True)
        # target_score = logits.gather(1, pred_idx).sum()

        delta_list = []
        for i, adv_feature_i in enumerate(adv_features):
            _grad = torch.autograd.grad(madl_loss, adv_feature_i, retain_graph=True,
                                     create_graph=False, allow_unused=True)[0]

            grad = _grad.nan_to_num().detach().clone()
            grad_norm = torch.norm(grad.view(_grad.size(0), -1), dim=-1)
            grad_norm = grad_norm.view(-1, *([1] * (grad.dim() - 1)))
            scaled_grad = grad / (grad_norm + 1e-12)


            temp_momentum_dict[i] = scaled_grad + momentum_dict[i] * self.mu

            delta_i = temp_momentum_dict[i].detach().clone()

            if self.norm == 'Linf':
                delta_i = self.alpha * torch.sign(delta_i)
            else:
                delta_i = delta_i.view(delta_i.size(0), -1).renorm(p=2, dim=0, maxnorm = self.alpha).view_as(temp_momentum_dict[i])  # L2归一化

            delta_list.append(delta_i)

        # adv_j = adv_feature_i[0:1]
        # adv_n = adv_data[0:1]
        # logits, adv_features = self.model.get_logits_and_intermediate_features(adv_n)
        batch_size = adv_data.shape[0]

        temp_feature_grad_list =[]

        for i, adv_feature_i in enumerate(adv_features):
            temp_feature_grad = torch.ones_like(adv_feature_i).to(self.device)
            temp_feature_grad_list.append(temp_feature_grad)


        # 如果不使用逐样本计算，则直接计算整个batch的梯度；如果使用逐样本计算，则对每个样本单独计算梯度并存储.
        #  each sample’s selected log-probability is independent of the other samples, so instead of computing the gradient per-sample, we can compute the gradient for the entire batch separately. with slice j equal to that sample’s gradient. This allows us to compute the gradients for the entire batch at once and then slice out the relevant gradients for each sample. This is more efficient than computing the gradients separately for each sample, especially when using GPU acceleration.
        if not per_sample:
            log_probs = F.log_softmax(logits, dim=1)
            cls = logits.argmax(dim=1, keepdim=True)
            log_f = log_probs.gather(1, cls).sum()
            temp_feature_grad_list = []

            for adv_feature_i in adv_features:
                feature_grad = torch.autograd.grad(
                    log_f,
                    adv_feature_i,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                temp_feature_grad_list.append(feature_grad.detach().clone())
        else:
            for j in tqdm(range(batch_size), desc="GradCAM++", leave=False):
                adv_j = adv_data[j:j+1]
                logits, adv_features_j = self.model.get_logits_and_intermediate_features(adv_j)
                log_probs = F.log_softmax(logits, dim=1)
                cls = logits.argmax(dim=1, keepdim=True)
                log_f = log_probs.gather(1, cls).squeeze()
                for i, adv_feature_i_j in enumerate(adv_features_j):
                    feature_grad = torch.autograd.grad(log_f, adv_feature_i_j, retain_graph=True, create_graph=False)[0]
                    temp_feature_grad_list[i][j:j+1] = feature_grad.detach().clone()



        for i, adv_feature_i in enumerate(adv_features):
            feature_grad_i = temp_feature_grad_list[i]

            eta_i = self.gradcam_plus_plus(feature= adv_feature_i, feature_grad=feature_grad_i, feature_delta=delta_list[i], input=adv_data)

            delta_t += eta_i

        momentum = delta_t

        return momentum, temp_momentum_dict



    def gradcam_plus_plus(self, feature, feature_grad, feature_delta, input):

        if len(feature.shape) == 2:
            # (batch_size, features) -> (batch_size, 1, features)
            feature = feature.unsqueeze(1)
            feature_grad = feature_grad.unsqueeze(1)
            feature_delta = feature_delta.unsqueeze(1)
        elif len(feature.shape) > 3:
            # (batch_size, channels, **) -> (batch_size, channels, sum(**))
            batch_size, channels = feature.shape[0], feature.shape[1]
            feature = feature.view(batch_size, channels, -1)
            feature_grad = feature_grad.view(batch_size, channels, -1)
            feature_delta = feature_delta.view(batch_size, channels, -1)

        grads_power_2 = feature_grad.pow(2)
        grads_power_3 = grads_power_2 * feature_grad
        sum_grads = (grads_power_3 * feature).sum(dim=(1, 2), keepdim=True)
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_grads
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * 1e-7)
        alpha = alpha_num / alpha_denom
        weights = (alpha * F.relu(feature_grad)).sum(dim=(1, 2), keepdim=True)
        saliency_map = (weights * feature)

        smap_flat = saliency_map.view(saliency_map.shape[0], -1)
        smap_min = smap_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        smap_max = smap_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)

        normalized_smap = (saliency_map - smap_min) / (smap_max - smap_min + 1e-8)

        binary_mask = (normalized_smap >= self.lambda_threshold).float()
        eta_i_j = feature_delta * binary_mask

        batch_size, channels, length = input.shape
        delta_t_i_j = F.interpolate(eta_i_j, size=length, mode='linear', align_corners=False) #shape [batch_size, channels, length]
        delta_t_i_j = delta_t_i_j.mean(dim=1, keepdim=True)  # 对通道维度求平均，得到形状 [batch_size, 1, length]
        delta_t_i_j = delta_t_i_j.expand(-1, channels, -1)

        return delta_t_i_j



    def forward(self, data, label, **kwargs):
        """
        MDAM攻击的前向传播过程 - 进一步修正版本
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1]

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # 初始化每层的动量

        # 获取原始信号的中间层特征（只计算一次）

        momentum_dict = {}
        with torch.no_grad():
            _, features_clean = self.model.get_logits_and_intermediate_features(data)
            # 只取前K层

            # 打印特征维度信息用于调试
            for i, feat in enumerate(features_clean):

                momentum_dict[i] = torch.zeros_like(feat)
                # self.logger.debug(f"Clean feature {i} shape: {feat.shape}")

        delta = self.init_delta(data)

        for t in tqdm(range(self.epoch), desc="Multi-Step-Attack", leave=False):
        # for t in range(self.epoch):
            # 计算对抗样本
            momentum, momentum_dict = self.single_step_attack(data, delta, momentum_dict, features_clean, label)
            delta = self.update_delta(delta, momentum, self.alpha)

        return delta.detach()

    def compute_support_activations(self, features):
        """
        计算支持当前预测的高激活值集合 Pi
        """
        # 处理不同维度的特征
        if len(features.shape) == 2:
            # (batch_size, features)
            B = features.mean(dim=1, keepdim=True)
        elif len(features.shape) == 3:
            # (batch_size, channels, length)
            B = features.mean(dim=(1, 2), keepdim=True)
        elif len(features.shape) == 4:
            # (batch_size, channels, height, width)
            B = features.mean(dim=(1, 2, 3), keepdim=True)
        else:
            # 默认情况：沿除第一维外的所有维度求平均
            dims = tuple(range(1, len(features.shape)))
            B = features.mean(dim=dims, keepdim=True)

        # 创建支持激活的掩码 Pi
        support_mask = features > B

        return support_mask

class MDAMce(MDAM):
    """
    This version adds the cross-entropy loss to the original MDAM loss. However, it is found to be as ineffective as MDAM methods in our experiments.
    """
    def compute_madl_loss(self, features_clean, features_adv, logits, label):
        """
        计算Multi-layer Activation Disruption Loss - 修正版本
        """
        total_loss = super().compute_madl_loss(features_clean, features_adv, logits, label)
        total_loss = total_loss / len(features_clean)
        loss = self.get_loss(logits, label)
        total_loss += loss

        return total_loss

if __name__ == "__main__":
    from taskAttack.Wrapper import Attack
    from taskAttack.Parser import get_parser
    from data import data_zoo

    args, parser = get_parser()

    args.algo = "mdam"
    args.gid = 0
    model = "awn"
    args.surrogate_model = model
    args.target_model = model

    args.psr = -20
    args.data = "dr2"
    data_name = data_zoo[args.data]["data_name"]
    model_ckp = f"checkpoints/{data_name}/nature/{data_name}_{model}.best.pt"
    args.surrogate_ckp = model_ckp
    args.target_ckp = model_ckp

    args.cuda = True
    args.test = True
    args.clean = True
    args.batch_size = 200  # 12000 per snr

    args.snr = [0]
    task = Attack(args, parser)
    task.conduct(ave_confMax=False, show_variance=False)