#!/usr/bin/env python3
"""Append the Transformer-only mixer-KL80 auxiliary control only."""
from ozstar_submit_counter_kl80aux_no_relation import main


if __name__ == "__main__":
    main("mixer_kl80aux", "smoke_test_counter_mixer_kl80aux.py")
