#!/usr/bin/env python3
"""Verify the direct Linear KL80 profile without running cyclic-obs tests."""
import torch as th

from smoke_test_linear_cyclic_obs_and_kl80 import check_linear_direct_kl80


if __name__ == "__main__":
    th.set_num_threads(1)
    check_linear_direct_kl80()
