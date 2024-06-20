"""
A collection of functions used for visualizing results
Created on 29.04.24
by: jokkus
"""
import os
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

# from world_model.model_net import ModelNetwork
from supplementary.settings import PROJECT_ENV


def visualize_RL():  # TODO: implement this functionality, TODO2: decide if needed
    # create env
    env = gym.make(PROJECT_ENV, render_mode="human")

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


def create_plots(array=None, dims=None, title="plot", save_path=None, full_save=False, custom_scaling=None,
                 additional_info=True):
    # TODO make both PCA & tSNE available, combine PCA & tSNE if too large
    # TODO set axis scaling (logarithmic / Asinh), aut detect points outside scaling, and generate addition plot fitting
    # TODO implement verbose

    # *** Handling information not provided ***
    if array is None:
        print("No array provided")
        return False

    array_size_lim = 100000  # current limit is 100.000, since it took long to calculate tSNE for larger
    array_size = array.shape[0]
    if array_size >= array_size_lim:
        print(f"Array size is larger than {array_size_lim}, PCA will be used before tSNE")
        # TODO implement
        #  set variables to activate this, decide which values to use

    if dims is None:
        dims = [2]
        print(f"dims not provided, automatically set to {dims[0]}")
    elif dims is int:
        dims = [dims]  # converting to list
    else:
        print(
            f"Datatype of \'dims\' is not compatible, should be \'int\' or \'tuple of ints\', "
            f"but {type(dims)} was provided")

    if save_path is None:
        print("save_path empty, no plots will be saved")

    if full_save is False:
        print(f"full_save not selected, only .png file will be saved")

    if custom_scaling is None or custom_scaling is False:
        print(f"custom_scaling not provided, automatic scaling will be applied")

    # *** Pyplot setup ***
    # * Colour-map initialization * Indexing based on age of inputs
    c = np.zeros(len(array))
    for i in range(len(array)):
        c[i] = i

    # * Fig setup *
    # fig, axs = plt.subplots(ncols=2, nrows=2) #, sharex=True, sharey=True
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title)
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], autoscale_on=True)

    # Changing scaling dependent on whether it is actions or observations
    # TODO: reimplement but make dynamic to size, just keep equal for all created
    # if custom_scaling == "actions":
    #     ax.set_autoscale_on(False)
    #     ax.set_xlim((-10, 10))
    #     ax.set_ylim((-10, 10))
    # elif custom_scaling == "observations":
    #     ax.set_autoscale_on(False)
    #     ax.set_xlim((-30, 30))
    #     ax.set_ylim((-30, 30))

    for dim in dims:
        if dim == 1 or dim == 2 or dim == 3:
            # Conducting the principal components analysis
            pca = PCA(n_components=dim)
            pc_result = pca.fit_transform(array)

            ax_l = fig.add_subplot(1, 2, 1)
            ax_l.set(adjustable='box', aspect='equal')
            ax_l.set_title("PCA")
            ax_l.scatter(pc_result[:, 0], pc_result[:, 1], c=c, cmap=colormaps["plasma"])

            # Conducting the t-distributed stochastic neighbor embedding
            tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(array)

            ax_r = fig.add_subplot(1, 2, 2)
            ax_r.set(adjustable='box', aspect='equal')
            ax_r.set_title("tSNE")
            mappable = ax_r.scatter(tsne_results[:, 0], tsne_results[:, 1], c=c, cmap=colormaps["plasma"])

            # TODO
            #  full save
            #  Calculate PCA before tSNE
            # TODO detect if ax.dataLim > x/ylim, and generate new plot

            # Colorbar set-up
            cbar = fig.colorbar(mappable, ax=[ax_l, ax_r], shrink=1, anchor=(0.95, 0))
            cbar.set_ticks(ticks=[0, 2048], labels=["Oldest", "Newest"])  # TODO: make ticks not hard-coded

            # Additional_info set-up
            if additional_info:
                # ax_text = fig.add_subplot(2, 1, 3)
                fig.text(0.05, 0.1, f"Mean:{array.mean():.8f}  std:{array.std():.8f}  Var:{array.var():.8f}",
                         fontsize=15)

            fig.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.8)

            plt.show()

            fig.savefig(f"{save_path}plots_{title}.png", bbox_inches="tight")
            pass
        else:
            print(f"Dimension provided is not humanly understandable, dimension is {dim}")
    pass


def visualize_PCA(array=None, dims=None, save_path=None, title="Plot", full_save=False):
    """
    Method to visualize an array using PCA downsampling
    :param array: The array with data to be downsampled & visualized
    :param dims: Dimensionality of output. eithr 2D or 3D
    :param save_path: Path to the folder in which the images will be saved
    :param title: Title of the plot
    :param full_save: Boolean determining if the plot should be saved using several filetypes
    :return: None
    """

    if save_path is None:
        print("save_path empty, no plots will be saved")

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
        save_path = f"{save_path}_plots"
        # Make folder
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}PCA_{title}.png", bbox_inches="tight")
        if full_save:  # save other relevant filetypes
            fig.savefig(f"{save_path}PCA_{title}", bbox_inches="tight", format="pdf")
            fig.savefig(f"{save_path}PCA_{title}", bbox_inches="tight", format="svg")
    pass


def visualize_tSNE(
        array=None,
        dims=None,
        save_path=None,
        title="Plot",
        time_calculations=True,
        full_save=False,
):
    """
    Method to visualize an array using PCA downsampling
    :param array: The array with data to be downsampled & visualized
    :param dims: Dimensionality of output. eithr 2D or 3D
    :param save_path: Path to the folder in which the images will be saved
    :param title: Title of the plot
    :param time_calculations: Boolean determining if the calculations should be timed
    :param full_save: Boolean determining if the plot should be saved using several filetypes
    :return: None
    """
    if save_path is None:
        print("Please provide a path to save the images")
        return False
    if dims is None:
        visualize_tSNE(
            array=array,
            dims=3,
            save_path=save_path,
            title=title,
            time_calculations=time_calculations,
        )  # 3D
        dims = 2  # calculate 2D

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
        save_path = f"{save_path}"
        # Make folder
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}tSNE_{title}.png", bbox_inches="tight")
        if full_save:  # save other relevant filetypes
            fig.savefig(f"{save_path}PCA_{title}.pdf", bbox_inches="tight", format="pdf")
            fig.savefig(f"{save_path}PCA_{title}.svg", bbox_inches="tight", format="svg")
    pass
