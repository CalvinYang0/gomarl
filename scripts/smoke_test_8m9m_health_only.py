#!/usr/bin/env python3
"""Real 8m9m model dimensions, synthetic TD/gradient probes; no SC2 launch."""
from smoke_test_5m6m_health_only import check_map


def main():
    check_map("8m_vs_9m", 8, 85,
              [8, 13, 18, 23, 28, 33, 38, 43, 48,
               53, 58, 63, 68, 73, 78, 83, 84])


if __name__ == "__main__":
    main()
