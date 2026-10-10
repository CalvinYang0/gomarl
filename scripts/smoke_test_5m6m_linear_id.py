#!/usr/bin/env python3
"""Production 5m6m GRU + Linear ID generated head; no simulator launch."""
import torch as th
from smoke_test_linear_id_baseline import check


if __name__ == "__main__":
    th.set_num_threads(1)
    check("5m_vs_6m", production_shapes=True)
