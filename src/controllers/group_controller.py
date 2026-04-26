from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)

    def _build_graph_context(self, batch, t):
        bs = batch.batch_size
        prev_action = th.zeros_like(batch["actions_onehot"][:, t]) if t == 0 else batch["actions_onehot"][:, t - 1]
        return {
            "obs": batch["obs"][:, t].reshape(bs, self.n_agents, -1),
            "prev_action": prev_action.reshape(bs, self.n_agents, -1),
        }
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        graph_context = self._build_graph_context(ep_batch, t)

        agent_outs, self.hidden_states, self.group_states, self.group_probs, self.group_graphs = self.agent(
            agent_inputs, self.hidden_states, graph_context=graph_context
        )
        self.group_struct_features = self.agent.group_struct_features
        self.group_node_embeddings = self.agent.group_node_embeddings
        self.group_role_prototypes = self.agent.group_role_prototypes
        self.current_groups = self.agent.current_groups

        return agent_outs
