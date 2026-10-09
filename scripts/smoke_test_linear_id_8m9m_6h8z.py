#!/usr/bin/env python3
"""Real map dimensions and production GRU/condition sizes; no SC2 launch."""
import torch as th

from smoke_test_linear_id_baseline import check


if __name__ == "__main__":
    th.set_num_threads(1)
    for scene in ("8m_vs_9m", "6h_vs_8z"):
        check(scene, production_shapes=True)
    print("PASS: both maps retain production GRU inputs, Linear ID heads and main-TD-only learner")
