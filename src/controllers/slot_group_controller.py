"""Always act with the important head, including epsilon exploration."""
from .clean_controller import CleanMAC


class SlotGroupMAC(CleanMAC):
    def forward(self, *args, **kwargs):
        q = super().forward(*args, **kwargs)
        self.latest_group_q = self.agent.latest_group_q
        self.latest_group_observations = self.agent.latest_group_observations
        return q

    def pop_test_gate_probability_summary(self):
        # Do not publish the second Linear group under the old "attention" label.
        if self.agent.is_grouped:
            self._test_gate_probability_sum = None
            self._test_gate_probability_count = 0
            return None
        return super().pop_test_gate_probability_summary()

    def pop_test_gate_probability_trajectory(self):
        trajectory = super().pop_test_gate_probability_trajectory()
        if trajectory is not None and self.agent.is_grouped:
            for key in ("branches", "agent_probability_branches"):
                mapping = trajectory.get(key, {})
                trajectory[key] = {"important" if k == "linear" else "unimportant" if k == "attention" else k: v
                                   for k, v in mapping.items()}
            trajectory["gate_note"] = "Two independent Linear heads; binary global slot partition, not gate probabilities"
        return trajectory
