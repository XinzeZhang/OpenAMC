import importlib

attack_zoo = dict(
    # FIM foundation attacks and channel-independent baselines.
    fgm=("taskAttack.attackmethods.gradient.fgsm", "FGSM"),
    pgd=("taskAttack.attackmethods.gradient.fgsm", "PGD"),
    pca=("taskAttack.attackmethods.universal.pca", "PCA_UAP"),
    vae=("taskAttack.attackmethods.universal.vae", "VAE_UAP"),

    # TFI/MVRG and the ten baselines reported in their papers.
    bim=("taskAttack.attackmethods.gradient.fgsm", "IFGSM"),
    mi=("taskAttack.attackmethods.gradient.fgsm", "MIFGSM"),
    ni=("taskAttack.attackmethods.gradient.fgsm", "NIFGSM"),
    vtmi=("taskAttack.attackmethods.gradient.fgsm", "VMIFGSM"),
    vtni=("taskAttack.attackmethods.gradient.fgsm", "VNIFGSM"),
    sfaa=("taskAttack.attackmethods.gradient.SFAA", "SFAA"),
    feig=("taskAttack.attackmethods.gradient.FEIG", "FEIG"),
    fciaa=("taskAttack.attackmethods.gradient.FCIAA", "FCIAA"),
    pngd=("taskAttack.attackmethods.gradient.PNGD", "PNGD"),
    mdam=("taskAttack.attackmethods.gradient.MDAM", "MDAM"),
    tfi=("taskAttack.attackmethods.gradient.tfi", "TFI"),
    mvrg=("taskAttack.attackmethods.gradient.mvrg", "MVRG"),
)


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise ValueError(
            "Unsupported attack algorithm {!r}. Available: {}".format(
                attack_name, sorted(attack_zoo)
            )
        )
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class

__version__ = '1.0.0'
