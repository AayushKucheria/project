import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl" # for pygame rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.pg import PG
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
        obs, reward, done, _ = env.step(to_numpy(action))
        agent.record(log_prob, reward)
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
    
    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'

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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize the agent
    agent = PG(state_dim, action_dim, cfg.lr, cfg.gamma)

    if not cfg.testing: # training
        for ep in range(cfg.train_episodes):
            # collect data and update the policy
            train_info = train(agent, env)
            train_info.update({'episodes': ep})
            
            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})
        
        if cfg.save_model:
            agent.save(cfg.model_path)

    else: # testing
        print("Loading model from", cfg.model_path, "...")
        # load model
        agent.load(cfg.model_path)
        print('Testing ...')
        test(agent, env, num_episodes=10)

# Entry point of the script
if __name__ == "__main__":
    main()