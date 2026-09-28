# import numpy as np
# import pandas as pd
# from taskRecog.util import Opt
import gc
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
from taskRecog.util import set_dataloader, fix_seed, temporary_seed, os_makedirs
import os
from matplotlib import pyplot as plt

class _PrintLogger:
    def __init__(self, silence = False):
        self.silence = silence

    def _emit(self, level, message, *args, **kwargs):
        if args:
            try:
                message = message % args
            except TypeError:
                message = str(message).format(*args, **kwargs)

        if not self.silence:
            print(f"[{level.upper()}] {message}")

    def __getattr__(self, name):
        if name in {'debug', 'info', 'warning', 'error', 'critical'}:
            return lambda message, *args, **kwargs: self._emit(name, message, *args, **kwargs)
        raise AttributeError(f"'_PrintLogger' object has no attribute '{name}'")

def empty_unused_gpu_cache(device=None):
    if not torch.cuda.is_available():
        return False

    gc.collect()
    if device is None:
        torch.cuda.empty_cache()
    else:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    if hasattr(torch.cuda, 'ipc_collect'):
        torch.cuda.ipc_collect()
    return True

def plot_snr_acc(Accuracy_list, num_snrs, data_name, plot_dir):
    for acc, tag in Accuracy_list:
        plt.plot(num_snrs, acc, label=tag)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy on {data_name} dataset")
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    # acc_dir = os.path.join(plot_dir, 'acc')
    os.makedirs(plot_dir, exist_ok=True)
    plt.legend()
    plt.savefig(plot_dir + '/' + 'snr_acc.png', dpi=300)
    plt.close()

def plot_delta_variance(snrs_pert_dict, clean, adv, eval_dir, data_name):
    exp_snrs = list(snrs_pert_dict.keys())
    for i, snr in enumerate(exp_snrs):
        snr_pert_dict = snrs_pert_dict[snr]
        truth = snr_pert_dict['label']

        # truth = clean.snrs_label[i]
        clean_pred = clean.snrs_pred[i]
        idx_cor = torch.where(clean_pred == truth)[0]


        selected_truth = truth[idx_cor]
        selected_pred = adv.snrs_pred[i][idx_cor]
        selected_delta = snr_pert_dict['pert'][idx_cor]
        selected_x = snr_pert_dict['x'][idx_cor]

        idx_succcess = torch.where(selected_pred != selected_truth)[0]
        idx_fail = torch.where(selected_pred == selected_truth)[0]
        # delta_succcess, x_success = selected_delta[idx_succcess], selected_x[idx_succcess]
        # delta_fail, x_fail = selected_delta[idx_fail], selected_x[idx_fail]

        success_axis_x_delta = []
        success_axis_y_delta  = []
        success_axis_x_delta_div_X = []
        success_axis_y_delta_div_X  = []
        fail_axis_x_delta  = []
        fail_axis_y_delta  = []
        fail_axis_x_delta_div_X  = []
        fail_axis_y_delta_div_X  = []

        for i in range(idx_succcess.size(0)):
            std, m = torch.std_mean(selected_delta[idx_succcess[i]])
            success_axis_x_delta.append(m.item())
            success_axis_y_delta.append(std.item())

            std, m = torch.std_mean(selected_delta[idx_succcess[i]] / selected_x[idx_succcess[i]])
            success_axis_x_delta_div_X.append(m.item())
            success_axis_y_delta_div_X.append(std.item())

        for j in range(idx_fail.size(0)):
            std, m = torch.std_mean(selected_delta[idx_fail[j]])
            fail_axis_x_delta.append(m.item())
            fail_axis_y_delta.append(std.item())

            std, m = torch.std_mean(selected_delta[idx_fail[j]] / selected_x[idx_fail[j]])
            fail_axis_x_delta_div_X.append(m.item())
            fail_axis_y_delta_div_X.append(std.item())

        mark_size=8
        fig = plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.scatter(success_axis_x_delta,success_axis_y_delta,label='attack-succes',c='red', marker='*', alpha=0.61,linewidth=0, s=mark_size)
        plt.scatter(fail_axis_x_delta,fail_axis_y_delta,label='attack-error',c='blue', marker='.', alpha=0.61, linewidth=0, s=mark_size)
        plt.xlabel("mean")
        plt.ylabel("std")
        #plt.yticks(np.linspace(0, 1, 11))
        plt.legend()
        plt.grid()

        plt.subplot(1,2,2)
        plt.scatter(success_axis_x_delta_div_X,success_axis_y_delta_div_X,label='attack-succes',c='red', marker='*',alpha=0.61,linewidth=0, s=mark_size)
        plt.scatter(fail_axis_x_delta_div_X,fail_axis_y_delta_div_X,label='attack-error',c='blue', marker='*', alpha=0.61, linewidth=0, s=mark_size)
        plt.xlabel("mean")
        plt.ylabel("std")
        # plt.title(f"delta result_analysis on {self.data_name} dataset on snr{snr}")
        #plt.yticks(np.linspace(0, 1, 11))
        plt.legend()
        plt.grid()

        fig.suptitle(f"delta result_analysis on {data_name} dataset on snr {snr}\nSuccess Num.: {len(success_axis_x_delta)}, Failed Num. :{len(fail_axis_x_delta)}")
        plot_dir = os.path.join(eval_dir, 'delta_variance')
        os_makedirs(plot_dir)
        plt.savefig(os.path.join(plot_dir, f'delta.snr_{snr}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# def cal_std_successRate(snrs_pert_dict, clean, adv, eval_dir):
#     exp_snrs = list(snrs_pert_dict.keys())
#     snr_list=[]
#     std_list=[]
#     successRate_list=[]
#     for i, snr in enumerate(exp_snrs):
#         snr_pert_dict = snrs_pert_dict[snr]
#         truth = snr_pert_dict['label']

#         clean_pred = clean.snrs_pred[i]
#         idx_cor = torch.where(clean_pred == truth)[0]

#         selected_truth = truth[idx_cor]
#         selected_pred = adv.snrs_pred[i][idx_cor]
#         selected_delta = snr_pert_dict['pert'][idx_cor]
#         selected_x = snr_pert_dict['x'][idx_cor]

#         idx_succcess = torch.where(selected_pred != selected_truth)[0]
#         idx_fail = torch.where(selected_pred == selected_truth)[0]
#         success_rate = len(idx_succcess) / len(selected_truth)

#         std_per_sample = torch.sqrt(torch.var(selected_delta.view(selected_delta.size(0), -1), dim=1))
#         std_value = std_per_sample.mean().item()

#         snr_list.append(snr)
#         std_list.append(std_value)
#         successRate_list.append(success_rate)

#     save_dir = os.path.join(eval_dir, 'std_successRate')
#     os_makedirs(save_dir)

#     np.savez(os.path.join(save_dir, f'snr_std_successRate.npz'),
#              snr=np.array(snr_list),
#              std=np.array(std_list),
#              success_rate=np.array(successRate_list))
def cal_std_successRate(snrs_pert_dict, clean, adv, eval_dir):
    exp_snrs = list(snrs_pert_dict.keys())
    snr_list = []
    std_list = []
    successRate_list = []
    balanced_indices_list = []  # 用于存储各个SNR下的均衡索引

    for i, snr in enumerate(exp_snrs):
        snr_pert_dict = snrs_pert_dict[snr]
        truth = snr_pert_dict['label']

        clean_pred = clean.snrs_pred[i]
        idx_cor = torch.where(clean_pred == truth)[0]

        selected_truth = truth[idx_cor]

        # 按类别统计正确分类的样本
        unique_classes = torch.unique(selected_truth)
        class_indices = {}
        for cls in unique_classes:
            class_indices[cls.item()] = torch.where(selected_truth == cls)[0]

        # 找出样本数量最小的类
        min_samples = min([len(indices) for indices in class_indices.values()])

        # 为每个类随机选择相同数量的样本
        balanced_idx = []
        for cls, indices in class_indices.items():
            # 如果当前类的样本数量超过最小数量，随机选择min_samples个样本
            if len(indices) > min_samples:
                selected_indices = indices[torch.randperm(len(indices))[:min_samples]]
            else:
                selected_indices = indices
            balanced_idx.append(selected_indices)

        # 合并所有类的索引
        balanced_idx = torch.cat(balanced_idx, dim=0)
        # 获取原始索引
        balanced_original_idx = idx_cor[balanced_idx]
        balanced_indices_list.append(balanced_original_idx.cpu().numpy())

        # 使用均衡的索引
        balanced_truth = selected_truth[balanced_idx]
        balanced_pred = adv.snrs_pred[i][idx_cor][balanced_idx]
        balanced_delta = snr_pert_dict['pert'][idx_cor][balanced_idx]
        balanced_x = snr_pert_dict['x'][idx_cor][balanced_idx]

        idx_succcess = torch.where(balanced_pred != balanced_truth)[0]
        success_rate = len(idx_succcess) / len(balanced_truth)

        std_per_sample = torch.sqrt(torch.var(balanced_delta.view(balanced_delta.size(0), -1), dim=1))
        std_value = std_per_sample.mean().item()

        snr_list.append(snr)
        std_list.append(std_value)
        successRate_list.append(success_rate)

    save_dir = os.path.join(eval_dir, 'std_successRate')
    os_makedirs(save_dir)

    np.savez(os.path.join(save_dir, f'snr_std_successRate.npz'),
             snr=np.array(snr_list),
             std=np.array(std_list),
             success_rate=np.array(successRate_list),
             balanced_indices=np.array(balanced_indices_list, dtype=object))

def plot_snr_successRate(snrs_pert_dict, clean, adv, eval_dir):
    exp_snrs = list(snrs_pert_dict.keys())
    snr_list=[]
    successRate_list=[]
    for i, snr in enumerate(exp_snrs):
        snr_pert_dict = snrs_pert_dict[snr]
        truth = snr_pert_dict['label']

        clean_pred = clean.snrs_pred[i]
        idx_cor = torch.where(clean_pred == truth)[0]

        selected_truth = truth[idx_cor]
        selected_pred = adv.snrs_pred[i][idx_cor]

        idx_succcess = torch.where(selected_pred != selected_truth)[0]
        idx_fail = torch.where(selected_pred == selected_truth)[0]
        success_rate = len(idx_succcess) / len(selected_truth)

        snr_list.append(snr)
        successRate_list.append(success_rate)

    save_dir = os.path.join(eval_dir, 'snr_successRate')
    os_makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, successRate_list, marker='o')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Attack Success Rate')
    plt.title(f'SNR vs Attack Success Rate')
    plt.grid(True)

    plt.xticks(snr_list)

    save_path = os.path.join(save_dir, f'snr_vs_successRate.png')
    plt.savefig(save_path)
    plt.close()


def clamp(x, x_min, x_max):
    '''E.g. delta = clamp(delta, self.data_min-data, self.data_max-data) \n
    This line only have meanling in image domain, which is equal to delta + data = clamp(delta + data , self.data_min, self.data_max).\n
    However, it should not be operated in raido domain.'''
    return torch.min(torch.max(x, x_min), x_max)

def bound_pert(delta, epsilon, norm = 'l2'):
    device = delta.device
    _delta = torch.zeros_like(delta).to(device)
    grad = torch.ones_like(delta) * delta
    grad = grad.to(device)
    grad = grad / (grad.abs().mean(dim=(1, 2), keepdim=True))
    if norm == 'linfty':
        _delta = torch.clamp(_delta + epsilon * grad.sign(), -epsilon, epsilon)
    else:
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=-1)
        grad_norm = grad_norm.view(-1, 1, 1)  # to debug
        scaled_grad = grad / (grad_norm + 1e-12)  # to debug
        _delta = (_delta + scaled_grad * epsilon).view(delta.size(0), -
                                                    1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(delta)

    _delta = _delta.nan_to_num()
    return _delta


def signal_power(_x):
    '''Input x: (sig_num, 2, sig_len)\n \n For each signal, power = energy / (2 * sig_len) if sig in Complex domain
    Refer to https://github.com/radioML/dataset/blob/4ecf612cfbc5bfc80eb8b0dbe63ed685d0a73c44/analyze_stats.py#L6-L10
    '''
    # X_Power = torch.norm(X.view(X.size(0), -1), dim=-1).pow(2) / X.view(X.size(0), -1).size(1) #This line equals to the following line    return torch.mean(torch.square(x.view(x.size(0), -1) / x.size(1)),dim=-1)
    assert len(_x.size()) == 3
    x = _x.detach().clone()
    power = signal_energy(x) / (_x.size(1) * _x.size(-1))
    return power

def signal_energy(_x):
    '''For each signal, energy = (real_part^2 + imag_part^2) '''
    assert len(_x.size()) == 3
    x = _x.detach().clone()
    return torch.square(torch.norm(x.view(x.size(0), -1), dim=-1))

def signal_db2pow(x):
    return 10**(x/10)

def signal_pow2db(x):
    if torch.is_tensor(x):
        return 10 * torch.log10(x)
    else:
        return 10 * np.log10(x)

def cal_PNR(signal, pert, snr):
    '''
    Return: PSR, PNR (dB)
    '''
    sig_power = signal_energy(signal)
    pert_power = signal_energy(pert)
    valid_mask = pert_power > 0 # This is to avoid log10(0) which will cause -inf. The samples with zero perturbation power (nan-to-0 in L2-norm attack) will be ignored in PSR and PNR calculation. This is reasonable because they do not contribute to the attack success.
    power_ratio = pert_power[valid_mask] / sig_power[valid_mask]
    PSR = signal_pow2db(power_ratio).mean().item()
    PNR = PSR + snr
    return PSR, PNR

def pathloss_shadowing_effect(K=1, d = 10, d_r =1, gamma=2.7, sigma=8, size=(2,128), seed = 2024, cutoff = 1):
    '''Return a simple channel as defined in Goldsmith A. Wireless communications. Cambridge university press. Note that, this channel is defined in DB domain.\n
    P_r / P_t [db] = 10 * log_10(K) - 10 * gamma * log_10(d / d_r) - psi[db], psi[db] ~ N(0, sigma)\n
    Return torch shape: size =(2,sig_len).\n
    The default parameters are from Kim et al. Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers. IEEE TWC 2023.'''
    with temporary_seed(seed):
        shadowing = torch.empty(size=size)
        torch.nn.init.trunc_normal_(shadowing,mean=0, std=sigma,a=-sigma * cutoff, b=sigma * cutoff)
        pathloss = signal_pow2db(K) - gamma * signal_pow2db(d / d_r)
        combine = pathloss - shadowing
        P_r2t = signal_db2pow(combine)

    g_r2t = torch.sqrt(P_r2t)
    # We don sqrt as in https://github.com/farismismar/eesc7v86-fall22/blob/ee3cd4ccc34c347ece21705bbba52d78a57f3a91/main.py#L406. The reason is the effect works on Power. Thus, the signal values should be affected with sqrt().
    return g_r2t

def pathloss_shadowing_channel(X, K=1, distance=10, d_r =1, gamma=2.7, sigma=8, dynamic = False, seed = 2024, unit = False):
    '''Return a simple channel as defined in Goldsmith A. Wireless communications. Cambridge university press. Note that, this channel is defined in DB domain.\n
    Return torch shape: size =(N, 2, sig_len).\n
    The default parameters are from Kim et al. Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers. IEEE TWC 2023.'''
    # shadowing = torch.tensor(np.random.normal(size = (2, sig_len), loc = 0, scale = sigma))

    size = X.size() if dynamic else (X.size(1), X.size(2))
    gain = pathloss_shadowing_effect(K, distance, d_r,gamma, sigma, size, seed)
    if unit:
        gain = unit_gain(gain)
        # print(gain.max(), gain.min())
    new = X * gain
    return new, gain

def rayleigh_effect(sigma = 1, size=(2,128), seed = 2024, conf = 0.1):
    # https://www.yyearth.com/index.php?aid=171
    with temporary_seed(seed):
        u = torch.empty(size=size)
        u = u.uniform_(0 + conf,1 - conf)
        u = torch.rand(size=size)
        g = -2.0 * torch.log(u)
        gain = sigma * torch.sqrt(g)
    return gain

def unit_gain(gain):
    assert len(gain.size()) >= 2
    if len(gain.size()) == 2:
        epsilon = torch.norm(torch.ones_like(gain)).item()
        scaled_gain =  gain / (torch.norm(gain) + 1e-12) * epsilon
        gain = scaled_gain.view(1, -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(gain)
    else:
        epsilon = torch.norm(torch.ones_like(gain[0])).item()
        scaled_gain =  gain / (torch.norm(gain.view(gain.size(0), -1), dim=-1).view(-1,1,1) + 1e-12) * epsilon
        gain = scaled_gain.view(gain.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(gain)
    return gain

def rayleigh_channel(X, sigma = 1, dynamic = False, seed = 2024, unit = False):
    size = X.size() if dynamic else (X.size(1), X.size(2))
    gain = rayleigh_effect(sigma, size, seed)
    if unit:
        gain = unit_gain(gain)
        # print(gain.max(), gain.min())
    new = X * gain
    return new, gain

def timedelay_channel(X, T = 0):
    _X = torch.roll(X.detach().clone(), shifts= T, dims=-1)
    return _X


# def AWGN_channel(X, sigma = 0.00005, cutoff = 1):
#     '''
#     Refer to https://github.com/kirtyvedula/jcm-awgn-imp/blob/be4a1cacf2433cea1ddd3331b806bda08b4e1ab3/channels.py#L17-L23
#     '''
#     # X_Power = signal_power(X)
#     # X_DB = signal_pow2db(X_Power)
#     # Noise_DB = X_DB + SNR
#     # Noise_Power = signal_db2pow(Noise_DB)
#     Noise = torch.zeros_like(X)
#     torch.nn.init.trunc_normal_(Noise,mean = 0, std = sigma,a =-sigma * cutoff, b=sigma * cutoff)
#     _X = X + Noise
#     return _X, Noise

def AWGN_channel(X, snr = 10):
    '''
    Refer to https://github.com/kirtyvedula/jcm-awgn-imp/blob/be4a1cacf2433cea1ddd3331b806bda08b4e1ab3/channels.py#L17-L23
    '''
    SNR = 10 ** (snr / 10) # dB to power
    Noise = torch.randn(X.size(), device=X.device) / ((2 * SNR) ** 0.5)
### Older implementation
    X_Power = signal_power(X)
    X_DB = signal_pow2db(X_Power)
    Noise_DB = X_DB - SNR
    Noise_Power = signal_db2pow(Noise_DB)
    Noise = torch.zeros_like(X)
    for i in range(X.size(0)):
        Noise[i].normal_(mean=0, std=torch.sqrt(Noise_Power[i]))
    _X = X + Noise
    return _X, Noise


def snrdata_eval(Input, Label, model, batch_size = None):
    # num_snrs = len(Input)
    # Confmat_Set = np.zeros((num_snrs, model.hyper.num_classes, model.hyper.num_classes), dtype=int)
    # Accuracy_list = np.zeros(num_snrs, dtype=float)

    # pre_lab_all = []
    # data = zip(Input, Label)
    # with tqdm(total=len(Input), desc='SNRs', mininterval=0.3, colour='red', leave=False) as pbar:
    #     for snr_i, (X_i, label_i) in enumerate(data):
    data_loader = set_dataloader(batch_size= batch_size,data_set=(Input, Label), shuffle=False)
    _,pred_i, label_i = model.loader_predict(data_loader)
    # pred_i = model.predict(X_i)
    cm_i =  confusion_matrix(label_i, pred_i)
    # Confmat_Set[ :, :] = cm_i
    acc = accuracy_score(label_i, pred_i)
    # pre_lab_all.append(pred_i)
    # pbar.update(1)

    # pre_lab_all = np.concatenate(pre_lab_all)
    # label_all = np.concatenate(Label)

    # F1_score = f1_score(label_i, pred_i, average='macro')
    # kappa = cohen_kappa_score(label_i, pred_i)
    # acc = np.mean(Accuracy_list)

    # logger.info('Acc. : {:.2f}%\tF1: {:.2f}\tKappa: {:.2f}'.format(acc * 100, F1_score, kappa))
    # logger.info(f'Macro F1-score is: {F1_score:.4f}')
    # logger.info(f'Kappa Coefficient is: {kappa:.4f}')

    return cm_i, acc, (pred_i, label_i)

if __name__ == "__main__":

    # gain = pathloss_shadowing_channel()
    # print(g_r2t)
    from data.RML import RML2016_10a_Data

    data = RML2016_10a_Data()
    data.pack_dataset()
    snr = 0
    sig_i, lab_i, idx_i = data.snr_slice('train', snr)
    sig_i = sig_i[0]

    X = torch.tensor(sig_i).float()

    print(AWGN_channel(X))