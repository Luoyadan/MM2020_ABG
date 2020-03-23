import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


import seaborn as sns
import torch
import numpy as np

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)

def visualize_TSNE(source_feat, target_feat, path):

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


    X = np.concatenate([source_feat.cpu().numpy(), target_feat.cpu().numpy()])
    y_source = np.zeros((source_feat.size()[0], ))
    y_target = np.ones((target_feat.size()[0], ))
    y = np.concatenate([y_source, y_target])

    digits_proj = TSNE().fit_transform(X)

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("Set1", 3))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(digits_proj[:, 0], digits_proj[:, 1], lw=0, s=30,
                    c=palette[y.astype(np.int)], alpha=0.5)
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    ax.axis('off')
    # ax.axis('tight')
    plt.savefig(path)

    txts = []
    # We add the labels for each digit.
    # txts = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
    #                        "calculator", "desk_chair","desk_lamp","desktop_computer","file_cabinet","unk"]
    # for i in range(len(txts)):
    #     # Position of each label.
    #     xtext, ytext = np.median(digits_proj[y == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, txts[i], fontsize=12)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

