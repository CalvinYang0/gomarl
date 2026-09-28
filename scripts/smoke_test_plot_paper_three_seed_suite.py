#!/usr/bin/env python3
"""Offline checks for paper-suite W&B run discovery."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_paper_three_seed_suite import discover_complete_series


def main():
    names = []
    for model, budget in (
        ("qmix", "5m"),
        ("id_hypernet", "5m"),
        ("obs_hypernet", "5m"),
        ("hyperselect", "10m"),
    ):
        names.extend(
            "grf_counter_paper_{}_{}_s{}".format(model, budget, seed)
            for seed in (1, 2, 3)
        )
    names.extend((
        "grf_counter_paper_vdn_5m_s1",
        "grf_counter_paper_vdn_5m_s2",
        "grf_counter_unrelated_s3",
        "smac_mmm2_paper_qmix_5m_s1",
    ))
    assert discover_complete_series(names, "grf_counter") == [
        ("QMIX", "grf_counter_paper_qmix_5m_s{seed}"),
        ("ID-based Hypernet", "grf_counter_paper_id_hypernet_5m_s{seed}"),
        ("Obs-based Hypernet", "grf_counter_paper_obs_hypernet_5m_s{seed}"),
        ("HyperSelect", "grf_counter_paper_hyperselect_10m_s{seed}"),
    ]

    doubled = names + [
        "grf_counter_paper_hyperselect_5m_s{}".format(seed)
        for seed in (1, 2, 3)
    ]
    assert discover_complete_series(doubled, "grf_counter")[-2:] == [
        ("HyperSelect (5M steps)", "grf_counter_paper_hyperselect_5m_s{seed}"),
        ("HyperSelect (10M steps)", "grf_counter_paper_hyperselect_10m_s{seed}"),
    ]
    print("paper three-seed suite discovery smoke test passed")


if __name__ == "__main__":
    main()
