# Liu, X., Yang, Y., He, K., Hopcroft, J.E., 2025. Parameter interpolation adversarial training for robust image classification. IEEE Trans. Inf. Forensics Security 20, 1613–1623. https://doi.org/10.1109/TIFS.2025.3533925
import torch
from torch import optim
import torch.nn.functional as F
from taskAttack.attackmethods.gradient.fgsm import PGD
from taskDefense.advTraining.PGD_AT import PGD_AT
from tqdm.auto import tqdm as real_tqdm
from taskRecog.util import logit_acc
import copy

def NMSE_loss(adv_logits, natural_logits, target,beta=4):
    adv_logits_norm=torch.norm(adv_logits,dim=1,keepdim=True)
    adv_logits_norm_num=adv_logits/adv_logits_norm
    natural_logits_norm=torch.norm(natural_logits,dim=1,keepdim=True)
    natural_logits_norm_num = natural_logits / natural_logits_norm
    predict=F.softmax(natural_logits,dim=1)
    NMSEweight = 1 - predict.gather(1, target.unsqueeze(1)).squeeze(1)
    loss=(((adv_logits_norm_num-natural_logits_norm_num)*(adv_logits_norm_num-natural_logits_norm_num)).sum(dim=1,keepdim=False)*NMSEweight).mean()
    loss=beta*loss
    return loss


class PIAT(PGD_AT):
    '''Liu, X., Yang, Y., He, K., Hopcroft, J.E., 2025. Parameter interpolation adversarial training for robust image classification.'''
    def before_train(self):
        super().before_train()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.hyper.lr, momentum=0.9, weight_decay=7e-4)

        self.piat_step = 0
        self.piat_alpha = 0.99
        self.buffer_ema = True
        self.model_state={
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        decay = min(self.piat_alpha, (self.piat_step + 1) / (self.piat_step + 10))
        # state = self.model.state_dict()
        old_state = model.state_dict()
        now_state = self.model.state_dict()
        for name in self.param_keys:
            self.model_state[name].copy_((1-decay) * now_state[name] + decay*old_state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.model_state[name].copy_((1-decay) * now_state[name] + decay * old_state[name])
            else:
                self.model_state[name].copy_(now_state[name])
        self.piat_step += 1

    def update_parameter(self):
        self.model.load_state_dict(self.model_state)

    def run_train_batch(self, data_batch, warmup=False):
        sig_batch, lab_batch = data_batch[0].to(self.hyper.device, non_blocking=True), data_batch[1].to(self.hyper.device, non_blocking=True)
        nat_logits, nat_loss = self.sig_logits_loss(sig_batch, lab_batch)
        nat_acc = logit_acc(nat_logits, lab_batch)
        self.train_nat_acc.update(nat_acc)
        if warmup:
            self.optimizer.zero_grad()
            nat_loss.backward()
            self.train_loss.update(0.0)
            self.train_adv_acc.update(0.0)
        else:
            self.attacker=PGD(model=self.model,logger=self.logger, **self.attacker_opts.dict)

            X_detached = sig_batch.detach().clone().requires_grad_(True)
            y_detached = lab_batch.detach().clone()
            self.update_psr_epsilon(X_detached)
            delta = self.attacker(X_detached,y_detached)
            self.model.train()
            if torch.isnan(delta).any():
                self.logger.error("NaN detected in delta! Setting NaNs to 0")
                delta = delta.nan_to_num()
            adv_X = X_detached + delta.to(self.hyper.device)
            adv_logits, adv_loss = self.sig_logits_loss(adv_X, y_detached)
            loss1=NMSE_loss(adv_logits,nat_logits,lab_batch,beta=4)
            adv_acc = logit_acc(adv_logits, lab_batch)

            adv_loss = nat_loss + loss1
            self.train_loss.update(adv_loss.item())
            self.train_adv_acc.update(adv_acc)
            self.optimizer.zero_grad()

            adv_loss.backward()

        self.optimizer.step()

        metric_dict = {'train_loss_adv': self.train_loss.avg,
                            'train_acc_adv': self.train_adv_acc.avg,
                            'train_acc_nat': self.train_nat_acc.avg
                            }

        return metric_dict


    def run_train_step(self):
        is_warmup = self.iter <= self.warmup_epoch


        train_step = self.warmup_train_step if is_warmup else self.run_train_batch
        desc = (
            f'Warmup Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}'
            if is_warmup
            else f'Adversarial Training Epoch {self.iter}/{self.hyper.epochs} {self.hyper.model_name} on {self.hyper.data_name}'
        )
        metric_dict = {}

        if not is_warmup:
            copy_model=copy.deepcopy(self.model)

        with real_tqdm(total=len(self.train_loader),
                desc=desc,
                postfix=dict,
                mininterval=0.3,
                dynamic_ncols=True, colour='red', leave=False) as pbar:
            for i, data_batch in enumerate(self.train_loader):
                metric_dict = train_step(data_batch)
                self.scheduler.step(self.iter - 1 + i / len(self.train_loader))
                pbar.set_postfix(**metric_dict)
                pbar.update(1)

        if not is_warmup:
            self.update_params(copy_model)
            self.update_parameter()

        return metric_dict


# uncomment the following lines to make unit test for adversarial training, and set the model and dataset in the main function below.
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
    args.defense_method = 'piat'

    args.model = 'awn'
    try:
        task = Task(args, parser)
        task.conduct(eval=True, ave_confMax=False, show_variance=False)
    except Exception as e:
        print(f"{e}")