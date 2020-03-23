import itertools
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch

def randSelectBatch(input, num):
    id_all = torch.randperm(input.size(0)).cuda()
    id = id_all[:num]
    return id, input[id]

def plot_confusion_matrix(path, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    num_classlabels = cm.sum(axis=1) # count the number of true labels for all the classes
    np.putmask(num_classlabels, num_classlabels == 0, 1) # avoid zero division

    if normalize:
        cm = cm.astype('float') / num_classlabels[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ' (%)', fontsize=20)
    cbar = plt.colorbar(shrink=0.945)
    cbar.ax.tick_params(labelsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    factor = 100 if normalize else 1
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*factor, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    # plt.tight_layout()
    plt.ylabel('Ground Truth', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=20)

    plt.savefig(path)
