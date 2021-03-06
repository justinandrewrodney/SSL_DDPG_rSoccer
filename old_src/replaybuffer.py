import spinup.algos.pytorch.ddpg.core as core
import numpy as np
import torch
"""From: https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ddpg/ddpg.html"""
# class ReplayBuffer:
#     """
#     A simple FIFO experience replay buffer for DDPG agents.
#     """

#     def __init__(self, obs_dim, act_dim, size):
#         self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=self.dev)
#         self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), dtype=torch.float32, device=self.dev)
#         self.act_buf = torch.zeros(core.combined_shape(size, act_dim), dtype=torch.float32, device=self.dev)
#         self.rew_buf = torch.zeros(size, dtype=torch.float32, device=self.dev)
#         self.done_buf = torch.zeros(size, dtype=torch.float32, device=self.dev)
#         self.ptr, self.size, self.max_size = 0, 0, size

#     def store(self, obs, act, rew, next_obs, done):
#         self.obs_buf[self.ptr] =torch.as_tensor(obs, dtype=torch.float32, device=self.dev) 
#         self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.dev) 
#         self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32, device=self.dev) 
#         self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=self.dev) 
#         self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.dev) 
#         self.ptr = (self.ptr+1) % self.max_size
#         self.size = min(self.size+1, self.max_size)

#     def sample_batch(self, batch_size=32):
#         idxs = np.random.randint(0, self.size, size=batch_size)
#         batch = dict(obs=self.obs_buf[idxs],
#                      obs2=self.obs2_buf[idxs],
#                      act=self.act_buf[idxs],
#                      rew=self.rew_buf[idxs],
#                      done=self.done_buf[idxs])
        
#         return {k: v for k,v in batch.items()}




#########
import spinup.algos.pytorch.ddpg.core as core
import numpy as np
import torch
"""From: https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ddpg/ddpg.html"""
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.dev) for k,v in batch.items()}
