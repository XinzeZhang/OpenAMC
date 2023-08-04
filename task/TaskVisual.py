import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def save_training_process(epochs_stats, plot_dir):
    fig1 = plt.figure(1)
    plt.plot(epochs_stats.epoch, epochs_stats.lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate")
    plt.grid()
    fig1.savefig(os.path.join(plot_dir, 'lr.svg'), format='svg', dpi=150)
    plt.close()

    fig2 = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_stats.epoch, epochs_stats.train_loss,
             "ro-", label="Train loss")
    plt.plot(epochs_stats.epoch, epochs_stats.val_loss,
             "bs-", label="Val loss")
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs_stats.epoch, epochs_stats.train_acc,
             "ro-", label="Train acc")
    plt.plot(epochs_stats.epoch, epochs_stats.val_acc,
             "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.grid()
    fig2.savefig(os.path.join(plot_dir,'loss_acc.svg'), format='svg', dpi=150)
    plt.show()
    plt.close()
