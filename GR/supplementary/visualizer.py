"""
A collection of functions used for visualizing results
Created on 29.04.24
by: jokkus
"""

import time

import torch
import numpy as np
import gymnasium as gym
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

import matplotlib.pyplot as plt
from matplotlib import colormaps

from world_model.model_net import ModelNetwork


def visualize_RL():
    # create env
    env = gym.make("HalfCheetah-v4", render_mode="human")

    # Sets an initial state, gets observations
    obs, _ = env.reset()
    obs = torch.tensor(obs)
    obs = obs.type(torch.FloatTensor)
    print(obs.type())

    callback = BaseCallback()
    rollout_buffer = RolloutBuffer()

    # load models
    # world_model = ModelNetwork.load_from_checkpoint("logs/model_test_run_1/test/version_0/checkpoints/epoch=24-step=23450.ckpt")

    rl_model = PPO.load(
        "logs/rl_test/rl_model_20240508-154616/checkpoints/checkpoint_model_100000_steps.zip"
    )
    rl_model.collect_rollouts(env, callback, rollout_buffer, 10)

    # eval mode? should set to training mode when using dropout, otherwise the model is run without dropout
    # x = obs + action

    # render 1200 times
    # for _ in range(1000):
    #     # render env
    #     env.render()
    #
    #     obs, _ = env.reset()
    #     action = rl_model(obs)
    #     # random action                 # TODO: make AI take action
    #     env.step(action)
    #     obs, _, _, _, _ = env.step(action)
    #     x = torch.concat(obs, action)
    # env.close()


def visualize_PCA(array=None, dims=None, save_path=None, title="Plot"):
    """
    TODO    +   make title include PCA
    :param array:
    :param dims:
    :param save_path:
    :param title:
    :return:
    """
    # Preparation for colourmap, simply going from first entry to last entry
    c = np.zeros(len(array))
    for i in range(len(array)):
        c[i] = i

    #  2D PCA & plotting
    if dims is None:
        # Pyplot preparation
        fig, axs = plt.subplots(2, 1, layout="constrained")
        fig.suptitle(title)

        # Conducting the 2D Principal component analysis
        pca = PCA(n_components=2)
        pca_result_2d = pca.fit_transform(array)
        # Conducting the 3D Principal component analysis
        pca = PCA(n_components=3)
        pca_result_3d = pca.fit_transform(array)

        # Plot the 2D plot
        ax = axs[1]
        ax.scatter(
            pca_result_2d[:, 0], pca_result_2d[:, 1], c=c, cmap=colormaps["plasma"]
        )

        # Plot the 3D plot
        axs[0].remove()
        ax = fig.add_subplot(211, projection="3d")
        ax.scatter(
            pca_result_3d[:, 0],
            pca_result_3d[:, 1],
            pca_result_3d[:, 2],
            c=c,
            cmap=colormaps["plasma"],
        )

    #  2D PCA & plotting
    elif dims == 2:
        # Pyplot preparation
        fig = plt.figure()
        fig.suptitle(title)
        # Conducting the 2D Principal component analysis
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(array)

        ax = fig.add_subplot()
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=c, cmap=colormaps["plasma"])

    #  3D PCA & plotting
    elif dims == 3:
        # Pyplot preparation
        fig = plt.figure()
        fig.suptitle(title)
        # Conducting the 3D Principal component analysis
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(array)

        ax = fig.add_subplot()
        ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            pca_result[:, 2],
            c=c,
            cmap=colormaps["plasma"],
        )

    else:
        print("not implemented")  # not planning to add 1D or 4D plotting
        return None

    # Show pyplot figure
    plt.show()

    if save_path is not None:
        fig.savefig(f"{save_path}PCA_{title}.png", bbox_inches="tight")
    pass


def visualize_tSNE(
    array=None, dims=None, save_path=None, title="Plot", time_calculations=True
):
    if dims is None:
        visualize_tSNE(array=array, dims=3, save_path=save_path, title=title, time_calculations=time_calculations)  # 3D
        dims = 2    # calculate 2D

    # Preparation for colourmap, simply going from first entry to last entry
    c = np.zeros(len(array))
    for i in range(len(array)):
        c[i] = i

    if time_calculations:  # method for timing the calculation of tSNE
        time_start = time.time()

    # Conducting the t-distributed stochastic neighbor embedding
    tsne = TSNE(n_components=dims, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(array)

    if time_calculations:  # method for timing the calculation of tSNE
        print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

    # Pyplot
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot()
    ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=c,
        cmap=colormaps["plasma"],
    )

    # Show pyplot figure
    plt.show()

    if save_path is not None:
        fig.savefig(f"{save_path}tSNE_{title}.png", bbox_inches="tight")
    pass
