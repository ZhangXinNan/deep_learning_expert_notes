

import matplotlib
import matplotlib.pyplot as plt


def plot_accuracy(train_acc_list, val_acc_list, fig_path):
    plt.plot(train_acc_list)
    plt.plot(val_acc_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['acc-train','acc-val'], loc='upper left')
    plt.savefig(fig_path)
    plt.clf()


def plot_loss(train_loss_list, val_loss_list, fig_path):
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss-train','loss-val'], loc='upper left')
    plt.savefig(fig_path)
    plt.clf()




