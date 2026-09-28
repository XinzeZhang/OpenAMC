import torch.nn as nn

from taskDefense.advTraining.PGD_AT import PGD, PGD_AT, logit_acc
from taskAttack.attackmethods.gradient.FEIG import FEIG

class EH_AT(PGD_AT):
    """"""
    def before_train(self):
        super().before_train()

        self.loss_alpha = 0.2
        self.loss_beta = 0.5

        self.mse = nn.MSELoss().to(self.hyper.device)

    def run_train_batch(self, data_batch, warmup = False):
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

            self.feig = FEIG(model=self.model, logger=self.logger, **self.attacker_opts.dict)

            pgd_delta = self.attacker(X_detached,y_detached)
            pgd_delta = pgd_delta.nan_to_num()
            feig_delta = self.feig(X_detached, y_detached)
            feig_delta = feig_delta.nan_to_num()

            x_star = X_detached + pgd_delta
            x_prime = X_detached + feig_delta

            self.model.train()

            pgd_logits, pgd_loss = self.sig_logits_loss(x_star, lab_batch)
            feig_logits, feig_loss = self.sig_logits_loss(x_prime, lab_batch)

            mse_loss = self.mse(pgd_logits, feig_logits)

            adv_loss = self.loss_alpha * nat_loss + (1 - self.loss_alpha - self.loss_beta) * pgd_loss  + self.loss_beta * feig_loss + mse_loss

            adv_acc = logit_acc(pgd_logits, lab_batch)

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

if __name__ == "__main__":
    from taskDefense.Parser import get_parser
    from taskDefense.Wrapper import Task


    args, parser = get_parser(parsing=True)

    args.data = 'dr4'
    args.model = 'awn'
    args.test = True
    args.clean = True
    # import os
    # from pathlib import Path
    # project_root = os.getcwd()
    # args.model_init_path = str(Path(__file__).resolve().relative_to(project_root))
    # args.gid = 1
    args.defense_method = 'ehat'
    # args.exp_name = 'defense_{method}_unitTest'.format(method=args.defense_method)
    # args.model = 'awn'
    args.snr = [0,10]
    args.algo = 'mi'
    args.psr = -20
    args.cuda = True
    # uncomment the following lines to set pretraining file, if not set, the model will be trained with the given defense method
    args.gid = 0
    # args.hyper = Opt()

    task = Task(args, parser)
    task.conduct(eval=True,ave_confMax=False, show_variance=False)
