'''
This file was copied from cleanrl.pro_continous_action and modified to my usecase
Created on 31.05.24
by: jokkus
'''

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from math import ceil

from cleanrl.rpo_continuous_action import make_env

from supplementary.settings import SEED, rl_config, PROJECT_ENV, set_current_time, set_path_addition, get_current_time


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]  # TODO: what does this do?
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False  # TODO enable tracking with W&B
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = PROJECT_ENV
    """the id of the environment"""
    total_timesteps: int = rl_config["custom_total_timesteps"]
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    rpo_alpha: float = 0.5
    """the alpha parameter for RPO"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha, action_logstd=None):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        if action_logstd is not None:  # modified to enable custom action_logstd
            data = np.empty((1, np.prod(envs.single_action_space.shape)))
            data.fill(action_logstd)
            data = torch.tensor(data)
        else:
            data = torch.zeros(1, np.prod(envs.single_action_space.shape))
        # print(f"Data: {data}", "*****", f"action_logstd: {action_logstd}", "*****")
        self.actor_logstd = nn.Parameter(data)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def train_rl_model(env=None, action_std=None, path_additional=None, verbosity=3, save_array_per_rollout=False):
    """
    TODO: remove redundant "0" at place 0 of actions and observations being saved
    TODO: implement env != None
    TODO(optional): make nicer progress_bar
    :param env:
    :param action_std:
    :param path_additional:
    :param verbosity: how much information should be printed; 0: None, 1: important, 2: interesting, 3: supplemental
    :return:
    """

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # ** Logging setup **
    config_name = rl_config["config_name"]  # Configuration name used in folder structure
    current_time = get_current_time()
    if path_additional is None and current_time == 0:  # no path addition -> current time is path addition
        current_time = time.time()  # Current date and time for unique directory names
        set_current_time(current_time)  # saving to access from other methods
        set_path_addition(current_time)  # saving to access from other methods
        path_additional = current_time

    logpath = (f"./logs/{config_name}/rl_model_{path_additional}/")
    writer = SummaryWriter(logpath)  # modified to my schema
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    global device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.rpo_alpha, action_std).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # ** Own storage setup **
    np_observations = np.empty(shape=(args.total_timesteps // args.batch_size), dtype=np.ndarray)
    np_actions = np.empty(shape=(args.total_timesteps // args.batch_size), dtype=np.ndarray)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # ** Adding custom scalars **
    # Create dynamic list of action_logstds being saved to summarywriter
    list_action_logstds = []
    for i in range(envs.single_action_space.shape[0]):
        list_action_logstds.append(f"own/action_logstd_{i}")
    layout = {
        "Own Tracker": {
            "action_logstds": ["Multiline", list_action_logstds],
        }
    }
    writer.add_custom_scalars(layout)

    # Print at start of training
    if verbosity >= 1:  # Important
        if action_std is not None:
            print(f"Started training on {args.env_id} at {start_time} with action standard deviation {action_std}")
        else:
            print(f"Started training on {args.env_id} at {start_time}")

    for update in range(1, num_updates + 1):
        episodic_rewards = []
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # if verbosity >= 1:      # Classified as important print
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        episodic_rewards.append(info["episode"]["r"])

        rollout_reward = sum(episodic_rewards) / len(episodic_rewards)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if verbosity >= 1:      # important
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # ** recording agent_logstds, and reward per rollout **   # TODO: investigate if actor_logstd truly only changes per update, otherwise modify method
        # * logging agent_logstds *
        np_agent_logstd = agent.actor_logstd.detach().numpy()
        np_agent_logstd = np.squeeze(np_agent_logstd)
        for i in range(agent.actor_logstd.size(dim=1)):
            writer.add_scalar(f"own/action_logstd_{i}", np_agent_logstd[i], update)
        if verbosity >= 3:      # supplemental
            print(f"action_logstds array: {np_agent_logstd}")

        # * logging reward per rollout
        np_rewards = rewards.detach().numpy()
        if verbosity >= 3:
            print(f"rollout: {update}, sum rewards: {np_rewards.sum()}, sum abs rewards: {np.absolute(np_rewards).sum()}, rewards size: {np.shape(np_rewards)}, episodic_return / num_episodes: {rollout_reward}")
        writer.add_scalar("own/rollout_rewards_sum", np_rewards.sum(), update)  # TODO why different episodic_return / num_episodes
        writer.add_scalar("own/rollout_reward", rollout_reward, update)

        # *** log actions & observations for space exploration
        # temp np arrays of tensors
        temp_obs = np.squeeze(torch.Tensor.numpy(obs))
        temp_actions = np.squeeze(torch.Tensor.numpy(actions))
        # temp np arrays for saving of copied arrays
        copy_obs = np.empty(shape=np.shape(temp_obs))
        copy_action = np.empty(shape=np.shape(temp_actions))
        np.copyto(copy_obs, temp_obs)
        np.copyto(copy_action, temp_actions)
        np_observations[update - 1] = copy_obs
        np_actions[update - 1] = copy_action

        if verbosity >= 1:
            print("*****" * 5, f"End of Rollout {update} / {num_updates + 1}", "*****" * 5)

    # ** Prints information **
    if verbosity >= 1:      # important
        print("#####" * 10)
        print(f"Total time: {time.time() - start_time}")
        print(
            f"mean SPS over {args.total_timesteps} timesteps: {int(args.total_timesteps / (time.time() - start_time))}")
        print(f"observations size : {np_observations.shape}, actions size : {np_actions.shape}")
        print("#####" * 10)

    # ** Saving the numpy arrays, as a compressed npz file **
    np.savez_compressed(
        f"{logpath}data.npz",
        observations=np_observations,
        actions=np_actions,
    )

    envs.close()
    writer.close()
