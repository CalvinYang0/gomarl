import copy
import torch as th

from .group_learner import GROUPLearner
from components.episode_buffer import EpisodeBatch
from utils.graph_grouping import hidden_similarity_graph, adjacency_to_groups


class GraphGROUPLearner(GROUPLearner):
    def __init__(self, mac, scheme, logger, args):
        super(GraphGROUPLearner, self).__init__(mac, scheme, logger, args)
        self.group_graph_sum = None
        self.group_graph_count = 0.0

    def change_group(self, batch: EpisodeBatch, change_group_i: int):
        group_update_mode = getattr(self.args, "group_update_mode", "graph")
        if group_update_mode == "hidden_similarity":
            self._change_group_hidden_similarity(batch, change_group_i)
            return
        if group_update_mode == "graph":
            self._change_group_graph(batch, change_group_i)
            return
        super(GraphGROUPLearner, self).change_group(batch, change_group_i)

    def _change_group_hidden_similarity(self, batch: EpisodeBatch, change_group_i: int):
        if batch is None:
            return

        if change_group_i == 0:
            self.group_graph_sum = None
            self.group_graph_count = 0

        with th.no_grad():
            mac_hidden = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                self.mac.forward(batch, t=t)
                mac_hidden.append(self.mac.hidden_states)
            mac_hidden = th.stack(mac_hidden, dim=1)
            graph = hidden_similarity_graph(mac_hidden[:, :-1])

        if self.group_graph_sum is None:
            self.group_graph_sum = graph
        else:
            self.group_graph_sum = self.group_graph_sum + graph
        self.group_graph_count += 1

        if change_group_i != self.args.change_group_batch_num - 1:
            return

        group_graph_avg = self.group_graph_sum / self.group_graph_count
        self.group_graph_sum = None
        self.group_graph_count = 0
        group_nxt = adjacency_to_groups(
            group_graph_avg,
            getattr(self.args, "graph_edge_threshold", 0.75),
        )
        self._apply_group_update(group_nxt)

    def _change_group_graph(self, batch: EpisodeBatch, change_group_i: int):
        if batch is None:
            return

        if change_group_i == 0:
            self.group_graph_sum = None
            self.group_graph_count = 0.0

        with th.no_grad():
            graph_rows = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                self.mac.forward(batch, t=t)
                graph_rows.append(self.mac.graph_rows)
            graph_rows = th.stack(graph_rows, dim=1)

        filled = batch["filled"][:, :batch.max_seq_length].float()
        graph_mask = filled.unsqueeze(-1).unsqueeze(-1)
        weighted_graph = graph_rows * graph_mask
        graph_sum = weighted_graph.sum(dim=(0, 1))
        graph_count = graph_mask.sum().item()

        if graph_count == 0:
            return

        if self.group_graph_sum is None:
            self.group_graph_sum = graph_sum
        else:
            self.group_graph_sum = self.group_graph_sum + graph_sum
        self.group_graph_count += graph_count

        if change_group_i != self.args.change_group_batch_num - 1:
            return

        group_graph_avg = self.group_graph_sum / self.group_graph_count
        self.group_graph_sum = None
        self.group_graph_count = 0.0
        group_graph_avg = (group_graph_avg + group_graph_avg.transpose(0, 1)) / 2.0
        group_graph_avg.fill_diagonal_(0.0)

        group_nxt = adjacency_to_groups(
            group_graph_avg,
            getattr(self.args, "graph_edge_threshold", 0.75),
        )
        self._apply_group_update(group_nxt)

    def _apply_group_update(self, group_nxt):
        if group_nxt == self.mixer.group:
            return

        while len(self.mixer.hyper_b1) < len(group_nxt):
            self.mixer.add_new_net()
            self.target_mixer.add_new_net()
        while len(self.mixer.hyper_b1) > len(group_nxt):
            self.mixer.del_net(len(self.mixer.hyper_b1) - 1)
            self.target_mixer.del_net(len(self.target_mixer.hyper_b1) - 1)

        self.mixer.update_group(copy.deepcopy(group_nxt))
        self.target_mixer.update_group(copy.deepcopy(group_nxt))
        self._update_targets()
