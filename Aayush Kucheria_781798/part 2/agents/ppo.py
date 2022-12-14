import torch
from torch.distributions import Normal
from common import helper as h
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Initialisation function for neural network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Critic(torch.nn.Module):

    def __init__(self, state_dim):
        "Initialises the critic network"
        super(Critic, self).__init__()
        self.value_network = torch.nn.Sequential(
            layer_init(torch.nn.Linear(state_dim, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 1), std=1.0),
        )
        
    def forward(self, state):
        "Returns the value of a given state"
        return self.value_network(state)

class Policy(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        "Initialises the policy network"
        super(Policy, self).__init__()
        self.actor_mean = torch.nn.Sequential(
            layer_init(torch.nn.Linear(state_dim, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, action_dim), std=0.01),
        )
        # TODO:last layer: layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),

        
        # TODO: policy_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        # ^^ ???
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        "Returns the action distribution for a given state"

        action_mean = self.actor_mean(state) # [Action_dim,]
        action_logstd = self.actor_logstd.reshape_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        return probs

class PPO(object):

    def __init__(self, state_dim, action_dim, lr, gamma, clip=0.2):

        self.name = "PPO"

        self.policy = Policy(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        # Check: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5, ) # eps from 3. in https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

        # Hyperparameters
        self.gamma = gamma
        self.clip = clip

        # Buffers
        self.states, self.actions, self.log_probs, self.rewards = [], [], [], []
        self.dist = None
        self.baseline = 0

    def update(self,):

        # OHHHHHHHHHH! I get it now!
        # The old_log_probs are the log_probs of the actions that were taken in the previous episode. That is, the actual log probabilities (not the dist).
        # So we need to store them in self.log_probs.
        # And thus we need a probability dist to get the "new" log_probs. That's where the sampling comes from!

        # Return mean if evaluation, else sample from the distribution
        
        # TODO: Can update policy for n_epochs here. 
        # CHeck: https://colab.research.google.com/github/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_colab.ipynb

        actions = self.dist.sample() 
        # log probability of all actions in the distribution
        act_logprobs = self.dist.log_prob(actions).sum() #   [Action_dim,] 

        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1) # shape [batch_size,]

        discounted_rewards = h.discount_rewards(rewards, self.gamma)
        
        advantage = discounted_rewards - self.baseline # Baseline is a scalar
        # Convert mean of discounted_rewards to float and store in baseline
        self.baseline = discounted_rewards.double().mean()

        # Surrogate loss
        old_log_probs = torch.stack(self.log_probs, dim=0).to(device)
        ratio = torch.exp(act_logprobs - old_log_probs.detach())

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
        self.dist = None

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {'logstd': self.policy.actor_logstd.cpu().detach().numpy()}

    # works
    def get_value(self, state):
        # State shape: [Observation_Shape] # Check environment documentation for the value of Observation_Shape

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float).to(device)
        self.dist = None

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {'logstd': self.policy.actor_logstd.cpu().detach().numpy()}

    # works
    def get_value(self, state):
        # State shape: [Observation_Shape] # Check environment documentation for the value of Observation_Shape

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float).to(device)

        self.critic.eval()
        value = self.critic(state)
        self.critic.train()

        return value

# Works
    def get_action(self, state, evaluation=False):
        # State shape: [Observation_Shape] # Check environment documentation for the value of Observation_Shape

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float).to(device)

        # Pass state x through the policy network (T1)
        self.policy.eval()
        act_dist = self.policy.forward(state)
        self.policy.train()
        # Return mean if evaluation, else sample from the distribution
        if evaluation:
            action = act_dist.mean # [Action_dim,]
        else:
            action = act_dist.sample() 

        act_logprob = act_dist.log_prob(action) #   [Action_dim,]
        
        # Returning log_prob sum because the probability of action is the product of the probabilities of each action. And when converted to log, the summ
        return action, act_logprob.sum(), act_dist#, self.critic(state) # [Action, LogProbability, Value]
    
    def record(self, log_prob, reward, act_dist):
        """ Store agent's and env's outcomes to update the agent."""
        self.log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward]))
        self.dist = act_dist

    def save(self, path):
        torch.save(self.policy.state_dict(), f'{path}/actor.pt')
        torch.save(self.critic.state_dict(), f'{path}/critic.pt')

    def load(self, path):
        self.policy.load_state_dict(torch.load(f'{path}/actor.pt'))
        self.critic.load_state_dict(torch.load(f'{path}/critic.pt'))


    