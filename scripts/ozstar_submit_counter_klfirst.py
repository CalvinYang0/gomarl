#!/usr/bin/env python3
"""Append the KL-first auxiliary mask-order control only."""
from ozstar_submit_counter_kl80aux_no_relation import main


if __name__ == "__main__":
    main("relation_kl80aux_klfirst", "smoke_test_counter_klfirst.py")
