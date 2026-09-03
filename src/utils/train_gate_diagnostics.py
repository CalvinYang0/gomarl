"""Read-only diagnostics of actual learner forwards, never resample a mask."""
import torch as th


class TrainGateDiagnostics:
    def __init__(self, valid, images=False, max_steps=200):
        self.valid = valid.detach().squeeze(-1).bool()
        self.episode = int(self.valid.sum(1).argmax().item())
        self.images = images
        self.max_steps = max_steps
        self.stats = {}
        self.trajectories = {}

    @th.no_grad()
    def add(self, name, values, t):
        if values is None or t >= self.valid.shape[1]:
            return
        values = values.detach().float()
        values = values.reshape(self.valid.shape[0], -1, values.shape[-1])
        x = values[self.valid[:, t]].reshape(-1)
        if not x.numel():
            return
        # Fixed-size streaming summaries: no full replay batch retained.
        precise = x.double()
        summary = th.stack((precise.new_tensor(x.numel()), precise.sum(), precise.square().sum(),
                            x.min(), x.max(), (x < .1).sum(), (x > .9).sum(),
                            (x < .5).sum())).double()
        histogram = th.histc(x, bins=10, min=0, max=1).double()
        if name not in self.stats:
            self.stats[name] = (summary, histogram)
        else:
            old, bins = self.stats[name]
            low, high = th.minimum(old[3], summary[3]), th.maximum(old[4], summary[4])
            old += summary
            old[3], old[4] = low, high
            bins += histogram
        if self.images and t < self.max_steps and self.valid[self.episode, t]:
            self.trajectories.setdefault(name, []).append(
                (t, values[self.episode].cpu().clone()))

    def log(self, logger, t_env, slot_names):
        for name, (summary, histogram) in self.stats.items():
            n, total, square, low, high, below, above, below_half = summary.cpu().tolist()
            mean = total / n
            metrics = dict(mean=mean, std=max(0., square / n - mean * mean) ** .5,
                           min=low, max=high, fraction_below_01=below / n,
                           fraction_above_09=above / n, fraction_below_05=below_half / n,
                           valid_slot_count=n)
            metrics.update({"bin_{:.1f}_{:.1f}_fraction".format(i / 10., (i + 1) / 10.): v / n
                            for i, v in enumerate(histogram.cpu().tolist())})
            for metric, value in metrics.items():
                logger.log_stat("train_gate/" + name + "/" + metric, value, t_env)
        if self.trajectories:
            logger.log_train_gate_heatmaps(self.trajectories, slot_names, t_env, self.episode)
