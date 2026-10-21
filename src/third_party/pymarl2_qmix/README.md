# Pinned PyMARL2 QMIX

Source: https://github.com/hijkzzz/pymarl2

Revision: `8ccac7c5aa134422a2e3009be735d23cdf8ce2f8`.

The upstream Apache-2.0 license is in `LICENSE.upstream`. The original paths,
immutable URLs, original SHA-256 values and every import adaptation are listed
in `provenance.json`. `upstream_qmix.yaml` is the original reference config.

Only imports have been adapted. Algorithm class and function bodies are not
rewritten. The Qatten and matrix-game-only imports in `learner.py` are omitted;
the registry wrapper rejects everything except SC2 + QMIX before calling it.
`controller_base.py` uses this project's agent registry to select the vendored
agent; its action selector is the vendored upstream implementation.

The registered learner adds configuration guards only and inherits the entire
upstream training/update/save/load implementation. In particular its TD loss
keeps the upstream `0.5` coefficient and its agent contains only fc1, GRU, fc2.

This pins the official PyMARL2 model and learning algorithm, not an exact
reproduction of the GoMARL paper's unpublished baseline run configuration.
The surrounding Sacred/W&B integration, replay and fault-tolerant SC2 runner
remain project infrastructure. Installed SC2/SMAC versions must be reported
separately; they cannot be inferred from this source revision.
