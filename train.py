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

if torch.backends.mps.is_available():
    os.environ["MUJOCO_GL"] = "glfw" # for pygame rendering
else:
    os.environ["MUJOCO_GL"] = "egl" # for pygame rendering

from agents.pg import PG
from agents.ddpg import DDPG
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
        action, log_prob = agent.get_action(obs)
        obs_old = obs.copy()
        obs, reward, done, _ = env.step(to_numpy(action))
        if log_prob:
            agent.record(log_prob, reward)
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

    total_test_reward = 0
    for ep in range(num_episodes):
        obs, done = env.reset(), False
        test_reward = 0

        while not done:
            # PG
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, _ = env.step(to_numpy(action))

            test_reward += reward
        
        total_test_reward += test_reward
        logger.info(f"Episode {ep}: {test_reward}")
    
    logger.info(f"Average reward over {num_episodes} episodes: {total_test_reward/num_episodes}")

    return total_test_reward/num_episodes

# The main function
@hydra.main(config_path="configs", config_name="bipedalwalker_easy")
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
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)

    # Create the environment
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)
    env.seed(cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/cfg.env_name/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/cfg.env_name/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name)

    # Get state and action dimensionality
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize the agent
    # agent = PG(state_dim, action_dim, cfg.lr, cfg.gamma)

    print(cfg)

    def do_round(x):
        if agent_type == 'PG':
            lr = x[0]
            agent = PG(state_dim[0], action_dim, lr, cfg.gamma)
        elif agent_type == 'DDPG':
            actor_lr, critic_lr, tau, batch_size = x
            agent = DDPG(state_dim, action_dim, max_action, actor_lr, critic_lr, cfg.gamma, tau, batch_size, normalize=cfg.normalize, buffer_size=cfg.buffer_size)
        else:
            raise ValueError('Unknown agent type')

        # Model filename
        if cfg.model_path == 'default':
            if agent_type == 'PG':
                model_path = work_dir/'model'/f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}-pg.pt'
            elif agent_type == 'DDPG':
                model_path = work_dir/'model'/f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}-ddpg'
                h.make_dir(model_path)
        else:
            model_path = cfg.model_path

        print(agent.__class__.__name__, x)

        if not cfg.testing: # training
            rewards = []
            for ep in tqdm(range(cfg.train_episodes)):
                # collect data and update the policy
                train_info = train(agent, env)
                train_info.update({'episode': ep})
                
                if cfg.use_wandb:
                    wandb.log(train_info)
                if cfg.save_logging:
                    L.log(**train_info)
                if (not cfg.silent) and (ep % 10 == 0):
                    print(train_info)
                rewards.append(train_info['ep_reward'])
            
            if cfg.save_model:
                agent.save(model_path)

        else: # testing
            print("Loading model from", model_path, "...")
            # load model
            agent.load(model_path)
            print('Testing ...')
            test(agent, env, num_episodes=10)
        
        # print("Max reward over training:", np.max(rewards))
        # print("Median reward over last 100 episodes:", np.median(rewards[-100:]))
    
        return -np.median(rewards[-100:])

    hyperparameter_search = True
    agent_type = 'PG'

    if hyperparameter_search:
        from skopt import gp_minimize
        from skopt.space import Real, Integer

        if agent_type == 'PG':
            space = [Real(1e-5, 1e-3, name='lr', prior='log-uniform')]
        elif agent_type == 'DDPG':
            space = [Real(1e-5, 1e-3, name='actor_lr', prior='log-uniform'),
                    Real(1e-5, 1e-3, name='critic_lr', prior='log-uniform'),
                    Real(1e-5, 1e-3, name='tau', prior='log-uniform'),
                    Integer(32, 1024, name='batch_size'),]

        res = gp_minimize(do_round, space, n_calls=30, n_random_starts=3, verbose=True)
        print("*"*80)
        print("Best parameters:", res.x)
        print("Best reward:", -res.fun)
    else:
        if agent_type == 'PG':
            do_round([cfg.lr])
        elif agent_type == 'DDPG':
            do_round([cfg.actor_lr, cfg.critic_lr, cfg.tau, cfg.batch_size])
# Entry point of the script
if __name__ == "__main__":
    main()