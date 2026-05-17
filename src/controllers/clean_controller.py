import torch as th

from .basic_controller import BasicMAC


class CleanMAC(BasicMAC):
    def _exclude_agent_id_from_trunk(self):
        return getattr(self.args, "clean_model_type", "baseline").replace("-", "_") in {
            "hypermarl_id",
            "hypermarl_fullnet",
            "rpg_relation_hypercond",
            "rpg_relation_route",
            "rpg_structured_hypercond",
            "rpg_full_structured_hypercond",
            "rpg_readout_structured_hypercond",
            "rpg_fixed_structured_maker",
            "two_graph_gat_hypercond",
            "hetero_gat_hypercond",
        }

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id and not self._exclude_agent_id_from_trunk():
            input_shape += self.n_agents
        return input_shape

    def _build_inputs(self, batch, t):
        batch_size = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id and not self._exclude_agent_id_from_trunk():
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(batch_size, -1, -1))
        return th.cat([x.reshape(batch_size, self.n_agents, -1) for x in inputs], dim=-1)

    def _build_model_context(self, batch, t):
        batch_size = batch.batch_size
        prev_action = batch["actions_onehot"][:, t - 1] if t > 0 else batch["actions_onehot"][:, t].new_zeros(
            batch["actions_onehot"][:, t].shape
        )
        return {
            "obs": batch["obs"][:, t].reshape(batch_size, self.n_agents, -1),
            "prev_action": prev_action.reshape(batch_size, self.n_agents, -1),
        }

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if test_mode:
            self.agent.eval()
        model_context = self._build_model_context(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, context=model_context, test_mode=test_mode
        )
        self.execution_scope = getattr(self.agent, "execution_scope", "ctde")
        self.latest_route_logits = getattr(self.agent, "latest_route_logits", None)
        self.latest_route_indices = getattr(self.agent, "latest_route_indices", None)
        self.latest_graph_adj = getattr(self.agent, "latest_graph_adj", None)
        self.latest_graph_nodes = getattr(self.agent, "latest_graph_nodes", None)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
