#!/usr/bin/env python3
"""Plan/config checks, no cluster/cloud writes and no simulator launch."""
import copy
from pathlib import Path
import shlex

from ozstar_submit_qtot_and_onefill_15jobs import build_plans, QTOT_MODELS
from ozstar_submit_5m6m_onefill_gate_15jobs import build_plans as reference
from ozstar_submit_linear_three_model_counter_mmm2_10m_3seeds import validate_config_keys, guard_and_route


def extras(plan):
    return dict(x.split("=", 1) for x in shlex.split(plan["exports"]["EXTRA_ARGS"]))


def main():
    root = Path(__file__).resolve().parents[1]
    plans = build_plans(root)
    assert len(plans) == len({p["job_name"] for p in plans}) == 15
    assert {p["seed"] for p in plans} == {1, 2, 3}
    legacy = {p["seed"]: p for p in reference(root) if p["label"] == "qmix_onefill_p10"}
    for p in plans:
        e = extras(p)
        assert p["exports"]["CONFIG"] == "clean_hyper"
        assert p["exports"]["MODEL_TYPE"] == "qmix_minimal"
        assert p["exports"]["T_MAX"] == "10050000"
        assert p["exports"]["BATCH_SIZE_RUN"] == "8" and p["exports"]["BATCH_SIZE"] == "128"
        assert next(a for a in p["sbatch_args"] if a.startswith("--time=")) == "--time=1-00:00:00"
        assert p["exports"]["MAP_NAME"] == p["map_name"]
        assert e["mixer_onefill_objective"] == "td" and e["mixer_onefill_residual"] == "False"
        old = extras(legacy[p["seed"]])
        if p["label"] in QTOT_MODELS:
            assert p["map_name"] == "5m_vs_6m" and p["scene"] == "smac_5m6m"
            assert e["learner"] == "qtot_composition_learner"
            changed = {key for key in set(old) | set(e) if old.get(key) != e.get(key)}
            assert changed <= {"learner", "mixer_residual_combination", "mixer_residual_secondary_ones",
                               "mixer_residual_fill_value", "mixer_qtot_combination",
                               "mixer_qtot_secondary_ones", "mixer_qtot_alpha_initial"}, changed
        else:
            assert p["label"] == "qmix_onefill_p10" and p["map_name"] in {"MMM2", "corridor"}
            assert e == old  # Map transfer changes NO learner/agent/input/loss override.
    routed = copy.deepcopy(plans)
    guard_and_route(routed, Path("/home/kyang/gomarl-runtime/gomarl-dual-branch"))
    validate_config_keys(root, routed)
    coverage = {(p["map_name"], p["label"]): {q["seed"] for q in plans
                if (q["map_name"], q["label"]) == (p["map_name"], p["label"])} for p in plans}
    assert len(coverage) == 5 and all(seeds == {1, 2, 3} for seeds in coverage.values())
    print("PASS: nine 5m6m output-composition + six unchanged-head map-transfer plans, all 24h, isolated names")


if __name__ == "__main__":
    main()
