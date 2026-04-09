from .group_controller import NMAC as GroupMAC


class GraphGroupMAC(GroupMAC):
    def __init__(self, scheme, groups, args):
        super(GraphGroupMAC, self).__init__(scheme, groups, args)
        self.graph_rows = None

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states, self.group_states, self.graph_rows = self.agent(agent_inputs, self.hidden_states)

        return agent_outs
