#!/usr/bin/env python3
"""Append only the fixed-p=.8 Concrete auxiliary control; never cancel jobs."""
from ozstar_submit_counter_kl80aux_no_relation import main


if __name__ == "__main__":
    main("relation_random80", "smoke_test_counter_random80_aux.py")
