import torch as th
import torch.nn as nn


class VDNMixer(nn.Module):
    def forward(self, agent_qs, states):
        del states
        return th.sum(agent_qs, dim=2, keepdim=True)
