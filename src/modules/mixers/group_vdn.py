import torch as th
import torch.nn as nn


class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.group = args.group
        if self.group is None:
            self.group = [[_ for _ in range(self.n_agents)]]

        # Keep the same surface as GroupMixer so existing saver / updater code keeps working.
        self.hyper_b1 = nn.ModuleList([nn.Identity() for _ in range(max(len(self.group), 1))])

    def add_new_net(self):
        self.hyper_b1.append(nn.Identity())

    def del_net(self, idx):
        del self.hyper_b1[idx]

    def update_group(self, new_group):
        self.group = new_group

    def get_w1_avg(self, a_h):
        raise NotImplementedError("group_vdn mixer does not provide w1 statistics.")

    def forward(self, qvals, states, a_h, all_group_state, which_network):
        tot_q = th.sum(qvals, dim=2, keepdim=True)
        zero = qvals.new_zeros(qvals.size(0), qvals.size(1), 1)
        return tot_q, [], zero
