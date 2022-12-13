import sys, os
sys.path.insert(0, os.path.abspath(".."))
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if torch.cuda.is_available():
    os.environ["MUJOCO_GL"] = "egl" # for pygame rendering
else:
    os.environ["MUJOCO_GL"] = "glfw" # for pygame rendering

from agents.pg import PG
from agents.ddpg import DDPG
from agents.pg_ac import A2C
from agents.ppo import PPO
from common import helper as h
from common import logger as logger


def to_numpy(tensor):
    return tensor.squeeze(0).cpu().numpy()


def train(agent, env):
    # Policy training function

    # Reset the environment and observe the initial state
    reward_sum, timesteps, done = 0, 0, False
    obs = env.reset()

    
    while not done:
        # PG  
        action, log_prob, act_dist = agent.get_action(obs)
        obs_old = obs.copy()
        obs, reward, done, _ = env.step(to_numpy(action))
        if type(log_prob) == tuple:
            agent.record(obs_old, log_prob[0], log_prob[1], reward, done, obs)
        elif log_prob:
            agent.record(log_prob, reward, act_dist)
        else:
            agent.record(obs_old, action, obs, reward, done)
        reward_sum += reward
        timesteps += 1
    
    # Update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episodes=10):

    test_rewards = []
    for ep in range(num_episodes):
        obs, done = env.reset(), False
        test_reward = 0

        while not done:
            # PG
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, _ = env.step(to_numpy(action))

            test_reward += reward
        
        test_rewards.append(test_reward)
        print(f"Episode {ep}: {test_reward}")
    
    print(f"Average reward over {num_episodes} episodes: {np.mean(test_rewards)}")
    print(f"Std over {num_episodes} episodes: {np.std(test_rewards)}")
    print(test_rewards)

    return np.mean(test_rewards), np.std(test_rewards)

# The main function
@hydra.main(config_path="configs", config_name="lunarlander_continuous_easy") # bipedalwalker_easy lunarlander_continuous_medium
def main(cfg):

    # Set seed for reproducibility
    h.set_seed(cfg.seed)

    # Define a run id based on current time
    cfg.run_id = int(time.time())

    # Create folders if needed
    work_dir = Path().cwd()/'results'
    if cfg.save_model: h.make_dir(work_dir/"model")
    if cfg.save_logging:
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # Use wandb to store stats
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}',
                    config=cfg)

    # Create the environment
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None, **cfg.env_parameters)
    env.seed(cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/cfg.env_name/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 100
            video_path = work_dir/'video'/cfg.env_name/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=f"{cfg.exp_name}-{str(cfg.seed)}-{str(cfg.run_id)}")

    # Get state and action dimensionality
    state_dim = env.observation_space.shape
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        action_dim = env.action_space.n
        max_action = action_dim
    else:
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]

    # Initialize the agent
    # agent = PG(state_dim, action_dim, cfg.lr, cfg.gamma)

    def do_round(x):
        if agent_type == 'PG':
            lr = x[0]
            agent = PG(state_dim[0], action_dim, lr, cfg.pg.gamma, cfg.pg.layers)
            train_episodes = cfg.pg.train_episodes
        elif agent_type == 'DDPG':
            actor_lr, critic_lr, tau, batch_size = x
            agent = DDPG(state_dim, action_dim, max_action, actor_lr, critic_lr, gamma=cfg.ddpg.gamma, tau=tau, batch_size=batch_size, 
                    uniform=cfg.ddpg.uniform, actor_layers=cfg.ddpg.actor_layers, critic_layers=cfg.ddpg.critic_layers, noise_std=cfg.ddpg.noise_std,
                    normalize=cfg.ddpg.normalize, use_ou=cfg.ddpg.use_ou, buffer_size=cfg.ddpg.buffer_size, weight_decay=cfg.ddpg.weight_decay)
            train_episodes = cfg.ddpg.train_episodes
        elif agent_type == 'A2C':
            lr = x[0]
            agent = A2C(state_dim[0], action_dim, lr, cfg.a2c.gamma, cfg.a2c.ent_coeff, cfg.a2c.normalize)
            train_episodes = cfg.a2c.train_episodes
        elif agent_type == "PPO":
            lr, batch_size, gamma = x
            agent = PPO(state_dim[0], action_dim, lr, gamma) #  cfg.ent_coef, cfg.normalize_ppo, eps=eps, batch_size=batch_size, clip=clip
            train_episodes = cfg.ppo.train_episodes
        else:
            raise ValueError('Unknown agent type')

        # Model filename
        if cfg.model_path == 'default':
            model_path = work_dir/'model'/f'{cfg.exp_name}-{str(cfg.seed)}-{str(cfg.run_id)}-{agent_type}'
        else:
            model_path = cfg.model_path

        # print(cfg)
        # print(agent.__class__.__name__, x, model_path)
        if agent_type == 'DDPG':
            print("Actor architecture", agent.pi.actor)
            print("Critic architecture", agent.q.value)

        if not cfg.testing: # training
    
            rewards = []
            for ep in tqdm(range(train_episodes)):
                # collect data and update the policy
                train_info = train(agent, env)
                train_info.update({'episode': ep})

                if cfg.use_wandb:
                    wandb.log(train_info)
                if cfg.save_logging:
                    L.log(**train_info)
                if (not cfg.silent) and (ep % 10 == 0):
                    print(train_info)
                if cfg.save_model and (ep % 100 == 0):
                    model_path_ep = f'{model_path}-{str(ep)}'
                    print("Saving model to", model_path_ep, "...")
                    h.make_dir(model_path_ep)
                    agent.save(model_path_ep)
                rewards.append(train_info['ep_reward'])

            if cfg.save_model:
                print("Saving model to", model_path, "...")
                h.make_dir(model_path)
                agent.save(model_path)

            # print("Max reward over training:", np.max(rewards))
            # print("Median reward over last 100 episodes:", np.median(rewards[-100:]))
            return -np.median(rewards[-100:])

        else: # testing
            print("Loading model from", model_path, "...")
            # load model
            agent.load(model_path)
            print('Testing ...')
            test(agent, env, num_episodes=50)

    hyperparameter_search = "one"
    if "agent_type" in cfg:
        agent_type = cfg.agent_type
    else:
        agent_type = "DDPG"

    if hyperparameter_search == "gp":
        from skopt import gp_minimize
        from skopt.space import Real, Integer

        if agent_type == 'PG':
            space = [Real(1e-5, 1e-2, name='lr', prior='log-uniform')]
        elif agent_type == 'A2C':
            space = [Real(1e-5, 1e-2, name='lr', prior='log-uniform')]
        elif agent_type == 'DDPG':
            space = [Real(1e-5, 1e-2, name='actor_lr', prior='log-uniform'),
                    Real(1e-5, 1e-2, name='critic_lr', prior='log-uniform'),
                    Real(1e-5, 1e-2, name='tau', prior='log-uniform'),
                    Integer(32, 1024, name='batch_size', prior='log-uniform'),]
        else:
            raise ValueError('Unknown agent type')

        res = gp_minimize(do_round, space, n_calls=50, n_random_starts=5, verbose=True)
        print("*"*80)
        print("Best parameters:", res.x)
        print("Best reward:", -res.fun)
    elif hyperparameter_search == "grid":
        from itertools import product

        space = [
            [1e5, 2e5], # buffer_size
            [True, False], # uniform
            [0.98, 0.99] # gamma
        ]

        for buffer_size, uniform, gamma in product(*space):
            cfg.buffer_size = buffer_size
            cfg.uniform = uniform
            cfg.gamma = gamma
            if agent_type == 'PG':
                do_round([cfg.pg.lr])
            elif agent_type == 'DDPG':
                do_round([cfg.ddpg.actor_lr, cfg.ddpg.critic_lr, cfg.ddpg.tau, cfg.ddpg.batch_size])
            elif agent_type == 'A2C':
                do_round([cfg.a2c.lr])
            
            else:
                raise ValueError('Unknown agent type')

    else:
        if agent_type == 'PG':
            do_round([cfg.pg.lr])
        elif agent_type == 'DDPG':
            do_round([cfg.ddpg.actor_lr, cfg.ddpg.critic_lr, cfg.ddpg.tau, cfg.ddpg.batch_size])
        elif agent_type == 'A2C':
            do_round([cfg.a2c.lr])
        elif agent_type == 'PPO':
            do_round([cfg.ppo.lr, cfg.ppo.batch_size, cfg.ppo.gamma])
        else:
            raise ValueError('Unknown agent type')
# Entry point of the script
if __name__ == "__main__":
    main()