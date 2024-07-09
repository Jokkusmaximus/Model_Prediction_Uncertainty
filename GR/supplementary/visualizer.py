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
from matplotlib.gridspec import GridSpec

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


def create_4_plots(array=None, title="plot", save_path=None, full_save=False, xy_lims_pca=None, xy_lims_tsne=None,
                   additional_info=False, additional_PCA=False):
    # TODO combine PCA & tSNE if too large
    # TOD aut detect points outside scaling, and generate addition plot fitting: done in visualize_tSNE

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

    # if dims is None:  # TODO decide if 3D is useful
    #     dims = [2]
    #     print(f"dims not provided, automatically set to {dims[0]}")
    # elif dims is int:
    #     dims = [dims]  # converting to list
    # else:
    #     print(
    #         f"Datatype of \'dims\' is not compatible, should be \'int\' or \'tuple of ints\', "
    #         f"but {type(dims)} was provided")
    dim = 2

    if save_path is None:
        print("save_path empty, no plots will be saved")

    if full_save is False:
        print(f"full_save not selected, only .png file will be saved")

    ax_PCA_xlim, ax_PCA_ylim = xy_lims_pca
    ax_tSNE_xlim, ax_tSNE_ylim = xy_lims_tsne
    xy_axis_scaler = 1.2

    # *** Pyplot setup ***
    # * Colour-map initialization * Indexing based on age of inputs
    c = np.zeros(len(array))
    for i in range(len(array)):
        c[i] = i

    # * Fig setup *
    heights = [1, 1]
    widths = [1]
    if additional_PCA:
        fig = plt.figure(figsize=(10, 7), layout="constrained")
        widths = [1, 1]
        gs = GridSpec(2, 2, figure=fig, width_ratios=widths, height_ratios=heights)
    else:
        fig = plt.figure(figsize=(5, 7), layout="constrained")
        gs = GridSpec(2, 1, figure=fig, width_ratios=widths, height_ratios=heights)
    fig.suptitle(title)
    gs.tight_layout(fig, rect=2)

    # Conducting the t-distributed stochastic neighbor embedding
    tsne = TSNE(n_components=dim, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(array)
    # adjusted x&y lims
    ax_tSNE_adj = fig.add_subplot(gs[0, 0])
    ax_tSNE_adj.set(adjustable='box', aspect='equal')
    ax_tSNE_adj.set_title("tSNE")
    ax_tSNE_adj.scatter(tsne_results[:, 0], tsne_results[:, 1], c=c, cmap=colormaps["plasma"])
    if ax_tSNE_xlim is None and ax_tSNE_ylim is None:  # only getting set the first time
        ax_tSNE_xlim = [i * xy_axis_scaler for i in ax_tSNE_adj.get_xlim()]
        ax_tSNE_ylim = [i * xy_axis_scaler for i in ax_tSNE_adj.get_ylim()]
    ax_tSNE_adj.set_xlim(ax_tSNE_xlim)
    ax_tSNE_adj.set_ylim(ax_tSNE_ylim)
    # automatic x&y lims
    ax_tSNE = fig.add_subplot(gs[1, 0])
    ax_tSNE.set(adjustable='box', aspect='equal')
    ax_tSNE.scatter(tsne_results[:, 0], tsne_results[:, 1], c=c, cmap=colormaps["plasma"])

    if additional_PCA:
        # Conducting the principal components analysis
        pca = PCA(n_components=dim)
        pc_result = pca.fit_transform(array)
        # adjusted x&y lims
        ax_PCA_adj = fig.add_subplot(gs[0, 1])
        ax_PCA_adj.set(adjustable='box', aspect='equal')
        ax_PCA_adj.set_title("PCA")
        ax_PCA_adj.scatter(pc_result[:, 0], pc_result[:, 1], c=c, cmap=colormaps["plasma"])
        if ax_PCA_xlim is None and ax_PCA_ylim is None:
            ax_PCA_xlim = [i * xy_axis_scaler for i in ax_PCA_adj.get_xlim()]
            ax_PCA_ylim = [i * xy_axis_scaler for i in ax_PCA_adj.get_ylim()]
        ax_PCA_adj.set_xlim(ax_PCA_xlim)
        ax_PCA_adj.set_ylim(ax_PCA_ylim)
        # automatic x&y lims
        ax_PCA = fig.add_subplot(gs[1, 1])
        ax_PCA.set(adjustable='box', aspect='equal')
        ax_PCA.scatter(pc_result[:, 0], pc_result[:, 1], c=c, cmap=colormaps["plasma"])

    # Colorbar
    mappable = ax_tSNE_adj.scatter(tsne_results[:, 0], tsne_results[:, 1], c=c, cmap=colormaps["plasma"])

    # TODO
    #  full save
    #  Calculate PCA before tSNE
    # TODO detect if ax.dataLim > x/ylim, and generate new plot

    # Colorbar set-up
    if additional_PCA:
        cbar = fig.colorbar(mappable, ax=[ax_tSNE_adj, ax_tSNE, ax_PCA_adj, ax_PCA], shrink=1, anchor=(0.95, 0))
    else:
        cbar = fig.colorbar(mappable, ax=[ax_tSNE_adj, ax_tSNE], shrink=1, anchor=(0.95, 0))
    cbar.set_ticks(ticks=[0, 2048], labels=["Oldest", "Newest"])  # TODO: make ticks not hard-coded

    # Additional_info set-up
    if additional_info:
        # ax_text = fig.add_subplot(2, 1, 3)
        fig.text(0.02, 0.05, f"Mean: {array.mean():.8f}  std: {array.std():.8f}  Var: {array.var():.8f}",
                 fontsize=15, rotation="vertical")

    # fig.subplots_adjust(0.1, 0.05, 0.85, 0.95)
    plt.show()

    fig.savefig(f"{save_path}plots_{title}.png", bbox_inches="tight")

    return (ax_PCA_xlim, ax_PCA_ylim), (ax_tSNE_xlim, ax_tSNE_ylim)


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
        arrays=[],
        dims=None,
        save_path=None,
        plot_title=None,
        titles=None,
        time_calculations=True,
        full_save=False,
):
    """
    Method to visualize an array using PCA downsampling
    :param arrays: list of arrays which are used for the tSNE plotting
    :param dims: Dimensionality of output. eithr 2D or 3D
    :param save_path: Path to the folder in which the images will be saved
    :param titles: Titles of the plots
    :param time_calculations: Boolean determining if the calculations should be timed
    :param full_save: Boolean determining if the plot should be saved using several filetypes
    :return: None
    """
    if save_path is None:
        print("Please provide a path to save the images")
        return False
    # TODO implement 3D
    if dims is None:  # calling for 3D before 2d if not specified
        #     visualize_tSNE(
        #         arrays=arrays,
        #         dims=3,
        #         save_path=save_path,
        #         plot_title=plot_title,
        #         titles=titles,
        #         time_calculations=time_calculations,
        #     )  # 3D
        dims = 2  # calculate 2D

    # Preparation for colourmap, simply going from first entry to last entry
    c = np.zeros(len(arrays[0]))  # assuming all arrays are of the same lenght
    for i in range(len(c)):
        c[i] = i

    # TODO remove of reimplement
    # if time_calculations:  # method for timing the calculation of tSNE
    #     time_start = time.time()
    # if time_calculations:  # method for timing the calculation of tSNE
    #     print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))

    tsne_results = []
    max_axis_size = 0
    max_x_axis = 0
    max_y_axis = 0
    i = 0
    for array in arrays:  # Conducting the t-distributed stochastic neighbor embedding for all arrays
        tsne = TSNE(n_components=dims, verbose=0, perplexity=40, n_iter=300)
        tsne_results.append(tsne.fit_transform(array))
        for j in range(len(tsne_results[i])):
            max_x_axis = max(max_x_axis, tsne_results[i][j][0])
            max_y_axis = max(max_y_axis, tsne_results[i][j][1])
        max_axis_size = max(max_x_axis, max_y_axis, max_axis_size)
        i += 1

    # scale up max_axis_size
    max_axis_size = 1.05 * max_axis_size

    # Create one long figure containing all plots
    fig = plt.figure(figsize=(7, 5 * len(arrays)), layout="constrained")
    heights = [1 for _ in range(len(arrays))]
    gs = GridSpec(len(arrays), 1, figure=fig, height_ratios=heights)
    fig.suptitle(plot_title, fontsize="xx-large")

    axs_arr = []

    for i in range(len(arrays)):
        ax_l = fig.add_subplot(gs[i])
        ax_l.set(adjustable='box', aspect='equal')
        ax_l.set_title(titles[i])
        ax_l.scatter(tsne_results[i][:, 0], tsne_results[i][:, 1], c=c, cmap=colormaps["plasma"])
        ax_l.set_xlim([-max_axis_size, max_axis_size])
        ax_l.set_ylim([-max_axis_size, max_axis_size])
        axs_arr.append(ax_l)

    # Creating Colorbar
    mappable = ax_l.scatter(tsne_results[-1][:, 0], tsne_results[-1][:, 1], c=c, cmap=colormaps["plasma"])
    cbar = fig.colorbar(mappable, ax=axs_arr, aspect=50)  # , anchor=(0.95, 0)
    cbar.set_ticks(ticks=[0, 2048], labels=["Oldest", "Newest"])

    plt.show()

    # Save long figure
    if save_path is not None:
        save_path = f"{save_path}"
        # Make folder
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f"{save_path}tSNE_{plot_title}_long.png", bbox_inches="tight")
        if full_save:  # save other relevant filetypes
            fig.savefig(f"{save_path}tSNE_{plot_title}_long.pdf", bbox_inches="tight", format="pdf")
            fig.savefig(f"{save_path}tSNE_{plot_title}_long.svg", bbox_inches="tight", format="svg")

    # Create one figure per plot, and save
    for i in range(len(tsne_results)):
        fig = plt.figure(figsize=(6, 5))
        fig.suptitle(f"{plot_title} {titles[i]}", fontsize="xx-large")
        ax = fig.add_subplot()
        ax.scatter(tsne_results[i][:, 0], tsne_results[i][:, 1], c=c, cmap=colormaps["plasma"])
        ax.set_xlim([-max_axis_size, max_axis_size])
        ax.set_ylim([-max_axis_size, max_axis_size])
        mappable = ax.scatter(tsne_results[i][:, 0], tsne_results[i][:, 1], c=c, cmap=colormaps["plasma"])  # similar line 388, optimze possible?
        cbar = fig.colorbar(mappable, ax=ax)
        cbar.set_ticks(ticks=[0, 2048], labels=["Oldest", "Newest"])  # TODO: make ticks not hard-coded

        # Save each figure
        if save_path is not None:
            save_path = f"{save_path}"
            # Make folder
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(f"{save_path}tSNE_{plot_title}_{titles[i]}.png", bbox_inches="tight")
            if full_save:  # save other relevant filetypes
                fig.savefig(f"{save_path}tSNE_{plot_title}_{titles[i]}.pdf", bbox_inches="tight", format="pdf")
                fig.savefig(f"{save_path}tSNE_{plot_title}_{titles[i]}.svg", bbox_inches="tight", format="svg")

        plt.show()
