"""Fading-Invariant Method (FIM) for over-the-air AMC attacks.

FIM learns a Neural Inverse Model (NIM) from perturbation/channel-inverse pairs
and applies transmit-power estimation before evaluation under random channel
effects. The method was published at ICASSP 2025.
"""

import os
from copy import deepcopy

import torch

from models.nn._baseNet import hyper
from taskAttack.Parser import get_group_args
from taskAttack.Wrapper import Results
from taskAttack.channelAttack.Wrapper import Attack
from taskAttack.channelAttack.neural_inverse_model import load_nim_model_class
from taskAttack.util import bound_pert
from taskRecog.util import save_training_process, set_dataloader, set_fitset


class FIMAttack(Attack):
    """Apply FIM to perturbations produced by any registered base attack."""

    def __init__(self, args, parser=None):
        super().__init__(args, parser)
        self.FIMArgs = get_group_args(
            self.parser, new_args=self.args, group_name="FIM"
        )
        self.nim_input_scale = None
        self.nim_target_scale = None

    def evaluation(
        self,
        snrs_pert_dict,
        method="fim",
        ave_confMax=False,
        show_variance=False,
    ):
        self.eval_dir = os.path.join(
            self.pert_dir, f"target.{self.args.target_model}"
        )
        logger = self.logger_config(
            dir=self.eval_dir,
            stage="evaluation",
            model_name=f"{self.args.surrogate_model}->{self.args.target_model}",
        )
        self.victim = self.model_import(logger, model_name=self.args.target_model)
        clean = super().record_result(
            snrs_pert_dict, self.victim, logger, "clean", ave_confMax
        )
        adv_rece = self.rece_result(
            snrs_pert_dict, self.victim, logger, "adv.rece", ave_confMax
        )
        adv_trans = self.rece_result(
            snrs_pert_dict, self.victim, logger, "adv.trans", ave_confMax
        )
        adv_fim = self.rece_result(
            snrs_pert_dict, self.victim, logger, f"adv.{method}", ave_confMax
        )
        logger.critical("Acc: Clean -> Adv.R(w/o CE) <- Adv.T(w/ CE) -> Adv.FIM")
        for index, snr in enumerate(self.exp_snrs):
            logger.critical(
                "SNR %s: %.2f%% -> %.2f%% <- %.2f%% -> %.2f%%",
                snr,
                clean.Acc[index] * 100,
                adv_rece.Acc[index] * 100,
                adv_trans.Acc[index] * 100,
                adv_fim.Acc[index] * 100,
            )
        return {
            "clean": clean,
            "advrece": adv_rece,
            "advtrans": adv_trans,
            "advfim": adv_fim,
        }

    def load_nim_model(self, snr):
        self.nim_fit_dir = os.path.join(
            self.pert_dir, f"NIM.{self.FIMArgs.nim_model}", f"snr{snr}"
        )
        logger = self.logger_config(
            self.nim_fit_dir, stage="NIM", model_name=self.FIMArgs.nim_model
        )
        self.nim_logger = logger
        nim_hyper = hyper()
        nim_hyper.patience = self.FIMArgs.nim_patience
        nim_hyper.epochs = self.FIMArgs.nim_epochs
        nim_hyper.device = torch.device(
            f"cuda:{self.args.gid}"
            if torch.cuda.is_available() and self.args.cuda
            else "cpu"
        )
        if nim_hyper.device.type == "cuda":
            torch.cuda.set_device(self.args.gid)
        nim_hyper.merge(opts=self.data_opts.info)
        nim_hyper.trainer_module = (
            "taskAttack.channelAttack.neural_inverse_model",
            "ChannelTrainer",
        )
        nim_hyper.model_fit_dir = self.nim_fit_dir
        nim_hyper.model_name = self.FIMArgs.nim_model
        return load_nim_model_class(self.FIMArgs.nim_model)(nim_hyper, logger)

    def build_nim_dataset(self, snr_pert_dict):
        """Build Algorithm 1's (delta, delta / h) training pairs."""
        perturbation = snr_pert_dict["pert"]
        pilot = torch.ones_like(perturbation)
        inputs, targets = [], []
        for index in range(self.FIMArgs.nim_num_channels):
            pilot_after, channel_gain, _ = self.channel_tranform(
                sig=pilot,
                snr_seed=self.FIMArgs.nim_seed + index,
                channel_seed=self.FIMArgs.nim_seed + index,
                dynamic=self.ChannelArgs.dynamic,
                margin_sample=True,
            )
            gain = channel_gain["gain"]
            safe_gain = torch.where(gain.abs() < 1e-12, gain.sign() * 1e-12, gain)
            safe_gain = torch.where(safe_gain == 0, torch.full_like(gain, 1e-12), safe_gain)
            inverse = perturbation / safe_gain
            amplification = 10 * torch.norm(pilot) / torch.norm(pilot_after).clamp_min(1e-12)
            limit = perturbation.abs().amax() * amplification
            inputs.append(perturbation)
            targets.append(inverse.clamp(min=-limit, max=limit))
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        self.nim_input_scale = inputs.abs().amax().clamp_min(1e-12)
        self.nim_target_scale = targets.abs().amax().clamp_min(1e-12)
        return inputs / self.nim_input_scale, targets / self.nim_target_scale

    def _fit_nim(self, snr, snr_pert_dict):
        self.nim_model = self.load_nim_model(snr)
        training_set = self.build_nim_dataset(snr_pert_dict)
        train_loader, _ = set_fitset(
            train_set=training_set,
            val_set=training_set,
            batch_size=self.FIMArgs.nim_batch_size,
        )
        stats = self.nim_model.xfit(
            train_loader,
            train_loader,
            xfit_stats=False,
            auto_pt_check=True,
            logger=self.nim_logger,
        )
        if stats is not None:
            save_training_process(
                stats, plot_dir=os.path.join(self.nim_fit_dir, "loss_curve", "figure")
            )

    def _nim_transform(self, perturbation):
        inputs = perturbation / self.nim_input_scale
        loader = set_dataloader(
            data_set=(inputs, torch.zeros_like(inputs)), shuffle=False
        )
        self.nim_model.eval()
        transformed, _ = self.nim_model.loader_predict(loader)
        return transformed * self.nim_target_scale

    def snr_result(self, model, snr_pert_dict, snr, tag, logger):
        if tag == "adv.fim":
            self._fit_nim(snr, snr_pert_dict)
        if tag in {"clean", "adv.rece"}:
            return super().snr_result(model, snr_pert_dict, snr, tag, logger)

        channel_results = Results(
            list(range(self.ChannelArgs.num_channel)), model.hyper.num_classes
        )
        if self.ChannelArgs.rebound:
            self.amp_epsilon = self.get_amp(snr_pert_dict)

        transformed_perturbations = []
        for index in range(self.ChannelArgs.num_channel):
            sample = deepcopy(snr_pert_dict)
            if tag == "adv.fim":
                sample["pert"] = self._nim_transform(sample["pert"])
            perturbation = sample["pert"]
            if self.ChannelArgs.rebound:
                perturbation = bound_pert(
                    perturbation, self.amp_epsilon, self.args.norm
                )
            perturbation, _, channel_info = self.channel_tranform(
                sig=perturbation,
                snr_seed=self.ChannelArgs.channel_seed + index,
                channel_seed=self.ChannelArgs.channel_seed + index,
                dynamic=self.ChannelArgs.dynamic,
                margin_sample=index > 0,
            )
            logger.info("SNR %s: %s", snr, channel_info)
            sample["pert"] = perturbation
            transformed_perturbations.append(perturbation)
            result = super(Attack, self).snr_result(model, sample, snr, tag, logger)
            channel_results(result, index)

        output_dir = os.path.join(self.eval_dir, tag)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            transformed_perturbations,
            os.path.join(output_dir, f"new_pert.snr{snr}.pt"),
        )
        return {
            "acc": channel_results.Acc.mean(),
            "cm": channel_results.Confmat_Set.mean(axis=0),
            "pe": channel_results.pe.mean(),
            "psr": channel_results.psr.mean(),
            "pnr": channel_results.pnr.mean(),
        }
