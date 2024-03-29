import sys, os
sys.path.insert(0, os.path.abspath(".."))
import copy
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from common import helper as h
from common.buffer import ReplayBuffer


# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_net(in_dim, layers, out_dim):
    layers = [in_dim] + layers

    net = []
    for i in range(1,len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        net.append(nn.LayerNorm(layers[i]))
        net.append(nn.ReLU())

    net.append(nn.Linear(layers[-1], out_dim))

    return nn.Sequential(*net)

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, layers, uniform=False):
        super().__init__()
        self.max_action = max_action
        self.actor = create_net(state_dim, layers, action_dim)
        self.actor.add_module('tanh', nn.Tanh())

        if uniform:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0)

            self.actor.apply(init_weights)

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layers, uniform=False):
        super().__init__()
        self.value = create_net(state_dim + action_dim, layers, 1)

        if uniform:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0)

            self.value.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x)  # output shape [batch, 1]


class DDPG(object):
    def __init__(
            self,
            state_shape,
            action_dim,
            max_action,
            actor_lr,
            critic_lr,
            actor_layers,
            critic_layers,
            gamma,
            tau,
            batch_size,
            use_ou=False,
            normalize=False,
            buffer_size=1e6,
            weight_decay=0,
            uniform=False,
            noise_std=0.01,
    ):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action
        self.pi = Policy(state_dim, action_dim, max_action, actor_layers, uniform=uniform).to(device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=actor_lr)

        self.q = Critic(state_dim, action_dim, critic_layers).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=critic_lr, weight_decay=weight_decay)

        self.buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))
        if normalize:
            self.state_scaler = h.StandardScaler()
        else:
            self.state_scaler = None

        if use_ou:
            self.noise = h.OUActionNoise(mu=np.zeros((action_dim,)))
        else:
            self.noise = None
        
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0 
        self.random_transition = 5000  # collect 5k random data for better exploration

    def update(self):
        """ After collecting one trajectory, update the pi and q for #transition times: """
        info = {}
        update_iter = self.buffer_ptr - self.buffer_head  # update the network once per transition
    
        if self.buffer_ptr > self.random_transition:  # update once have enough data
            for _ in range(update_iter):
                info = self._update()
    
        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info

    @property
    def buffer_ready(self):
        return self.buffer_ptr > self.random_transition

    def _update(self):
        batch = self.buffer.sample(self.batch_size, device=device)

        # TODO: Task 2
        ########## Your code starts here. ##########
        # Hints: 1. compute the Q target with the q_target and pi_target networks
        #        2. compute the critic loss and update the q's parameters
        #        3. compute actor loss and update the pi's parameters
        #        4. update the target q and pi using h.soft_update_params() (See the DQN code)
        
        # pass
        if self.state_scaler is not None:
            self.state_scaler.fit(batch.state)
            states = self.state_scaler.transform(batch.state)
            next_states = self.state_scaler.transform(batch.next_state)
        else:
            states = batch.state
            next_states = batch.next_state

        # compute current q
        q_cur = self.q(states, batch.action)
        
        # compute target q
        with torch.no_grad():
            next_action = (self.pi_target(next_states)).clamp(-self.max_action, self.max_action)
            q_tar = self.q_target(next_states, next_action)
            td_target = batch.reward + self.gamma * batch.not_done * q_tar
        
        # compute critic loss
        critic_loss = F.mse_loss(q_cur, td_target)

        # optimize the critic
        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        # compute actor loss
        actor_loss = -self.q(states, self.pi(states)).mean()

        # optimize the actor
        self.pi_optim.zero_grad()
        actor_loss.backward()
        self.pi_optim.step()

        # update the target q and target pi
        h.soft_update_params(self.q, self.q_target, self.tau)
        h.soft_update_params(self.pi, self.pi_target, self.tau)
        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {'q': q_cur.mean().item()}
    
    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        if self.state_scaler is not None:
            x = self.state_scaler.transform(x)

        if self.buffer_ptr < self.random_transition and not evaluation:  # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action shape is correct.
            # pass

            self.pi.eval()
            action = self.pi(x)
            self.pi.train()
            
            if not evaluation:
                std = torch.ones_like(action) * self.noise_std
                action += torch.normal(action, std)

            ########## Your code ends here. ##########

        return action, {}  # just return a positional value

    def record(self, state, action, next_state, reward, done):
        """ Save transitions to the buffer. """
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        self.pi.load_state_dict(torch.load(f'{filepath}/actor.pt'))
        self.q.load_state_dict(torch.load(f'{filepath}/critic.pt'))
    
    def save(self, filepath):
        torch.save(self.pi.state_dict(), f'{filepath}/actor.pt')
        torch.save(self.q.state_dict(), f'{filepath}/critic.pt')
