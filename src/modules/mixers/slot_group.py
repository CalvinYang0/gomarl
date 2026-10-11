"""Two local-group QMIXes, optionally followed by a global-state QMIX."""
from copy import copy

import torch as th
import torch.nn as nn

from .qmix import QMixer


class SlotGroupMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hierarchical = args.clean_model_type.endswith("hierarchy_hypercond")
        self.n_agents, self.obs_dim = args.n_agents, int(args.obs_shape)
        local = copy(args)
        # Preserve a fixed shape as membership changes. Membership bits
        # distinguish excluded fields from genuinely zero observed fields.
        local.state_shape = self.n_agents * self.obs_dim + self.obs_dim
        self.small = nn.ModuleList([QMixer(local), QMixer(local)])
        global_args = copy(args)
        global_args.n_agents = 2
        self.big = QMixer(global_args) if self.hierarchical else None

    def group_values(self, agent_qs, group_observations, masks):
        if agent_qs.shape[-2:] != (2, self.n_agents):
            raise ValueError("Expected two executed-action Q vectors over all agents")
        values = []
        for g in range(2):
            observed = group_observations[..., g, :, :].reshape(*agent_qs.shape[:2], -1)
            membership = masks[g].to(observed).expand(*agent_qs.shape[:2], self.obs_dim)
            info = th.cat((observed, membership), dim=-1)
            values.append(self.small[g](agent_qs[..., g, :], info))
        return th.cat(values, dim=-1)

    def forward(self, agent_qs, states, group_observations=None, masks=None):
        if group_observations is None or masks is None:
            raise ValueError("Group mixer requires actual masked group observations and membership")
        groups = self.group_values(agent_qs, group_observations, masks)
        return self.big(groups, states) if self.hierarchical else groups[..., :1]
