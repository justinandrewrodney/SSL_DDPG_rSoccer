import spinup.algos.pytorch.ddpg.core as core
import torch
class TScriptMLPActor(torch.nn.Module): 
    #This is modified from spinningup in order to work with torch script
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256,256], activation=torch.nn.Tanh):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = core.mlp(pi_sizes, activation, torch.nn.Tanh)

    def forward(self, obs):
        # Return output from network scaled to action space limits. (**removed act limits**)
        return self.pi(obs)

class TScriptMLPCritic(torch.nn.Module):
    #This is modified from spinningup in order to work with torch script
    def __init__(self, obs_dim, act_dim, hidden_sizes=[256,256], activation=torch.nn.ReLU):
        super().__init__()
        self.q = core.mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.