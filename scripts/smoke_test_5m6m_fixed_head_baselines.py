#!/usr/bin/env python3
"""Real 5m_vs_6m dimensions and fixed-head learner; no simulator launch."""
from smoke_test_8m9m_fixed_head_baselines import check_map


if __name__ == "__main__":
    check_map("5m_vs_6m", 5, 12, 55, 98)
