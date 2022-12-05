import torch
from torch.distributions import Normal
# from common import helper as h
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Initialisation function for neural network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.actor_mean = torch.nn.Sequential(
            layer_init(torch.nn.Linear(state_dim, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        return probs

class PPO(object):

    def __init__(self, state_dim, action_dim, lr, gamma, clip=0.2):

        # Initialise the neural network policy
        self.policy = Policy(state_dim, action_dim).to(device)

        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.clip = clip

        # Buffers
        self.states, self.actions, self.log_probs, self.rewards = [], [], [], []

        # Baseline
        self.baseline = 0

    def update(self,):

        # Prepare dataset used to update policy
        states = torch.stack(self.states, dim=0).to(device) # TODO Check

        # Implement the PPO algorithm
        ########## Your code starts here. ##########
        #
        # Get actions and log probs for states
        action_mean, action_logstd = self.policy(states)
        action_probs = Normal(action_mean, torch.exp(action_logstd))
        actions = action_probs.sample()
        log_probs = action_probs.log_prob(actions)

        # Advantage
        rewards = torch.stack(self.rewards, dim=0).to(device)
        discounted_rewards = h.discount_rewards(rewards, self.gamma)
        advantage = discounted_rewards - self.baseline
        self.baseline = discounted_rewards

        # Surrogate loss
        old_log_probs = torch.stack(self.log_probs, dim=0).to(device)
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate_loss = ratio * advantage

        # Clipped surrogate loss
        clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        clipped_surrogate_loss = clipped_ratio * advantage

        # Loss
        loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean()

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clean buffers
        self.states, self.actions, self.log_probs, self.rewards = [], [], [], []

        return loss.item()




    def get_action(self, state, evaluation=False):

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float).to(device)

        # Get action and log prob
        action_mean, action_logstd = self.policy(state)
        action_probs = Normal(action_mean, torch.exp(action_logstd))
        action = action_probs.sample()

        if evaluation:
            action = action_mean

        log_prob = action_probs.log_prob(action)

        # Convert to numpy
        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob
    
    def record(self, action_prob, reward):
        """ Store agent's and env's outcomes to update the agent."""
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward]))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    