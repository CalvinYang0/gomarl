#!/usr/bin/env python3
"""Append the single-linear relation + KL80 auxiliary experiment only."""
from ozstar_submit_counter_kl80aux_no_relation import main


if __name__ == "__main__":
    main("linear_relation_kl80aux", "smoke_test_counter_linear_kl80aux.py")
