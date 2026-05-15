import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim
        self.hypernet_layers = int(getattr(args, "hypernet_layers", 1))

        if self.hypernet_layers == 1:
            self.hyper_w1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif self.hypernet_layers == 2:
            hyper_hidden = int(getattr(args, "hypernet_embed", self.embed_dim))
            self.hyper_w1 = nn.Sequential(
                nn.Linear(self.state_dim, hyper_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hyper_hidden, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hyper_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hyper_hidden, self.embed_dim),
            )
        else:
            raise ValueError("Unsupported hypernet_layers={} for QMixer.".format(self.hypernet_layers))

        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.value = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        batch_size, seq_len, _ = agent_qs.shape
        flat_q = agent_qs.view(-1, 1, self.n_agents)
        flat_states = states.reshape(-1, self.state_dim)

        w1 = th.abs(self.hyper_w1(flat_states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(flat_states).view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(flat_q, w1) + b1)

        w_final = th.abs(self.hyper_w_final(flat_states)).view(-1, self.embed_dim, 1)
        value = self.value(flat_states).view(-1, 1, 1)
        q_tot = th.bmm(hidden, w_final) + value
        return q_tot.view(batch_size, seq_len, 1)
