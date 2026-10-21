#!/usr/bin/env python3
"""Opt-in deterministic soft one-fill p10, unchanged 5m6m actor, 3 seeds, 24h."""
import copy
import shlex

from ozstar_submit_5m6m_onefill_gate_15jobs import build_plans as historical_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite
from ozstar_submit_5m6m_id_kl80_3seeds import validate_installed_map

GROUP = "5m6m_soft_onefill_p10_10m_3seeds_20261021"
SMOKE_SCRIPTS = ("scripts/smoke_test_soft_onefill_gate.py",)


def build_plans(repo):
    plans = []
    for original in historical_plans(repo):
        if original["label"] != "qmix_onefill_p10":
            continue
        plan = copy.deepcopy(original)
        extra = dict(x.split("=", 1) for x in shlex.split(plan["exports"]["EXTRA_ARGS"]))
        extra["learner"] = "soft_onefill_gate_learner"
        # These hard-gate controls have NO role in the new forward.
        extra.pop("mixer_input_gate_temperature", None)
        plan["exports"]["EXTRA_ARGS"] = " ".join(k + "=" + v for k, v in extra.items())
        plan["exports"]["GROUP_NAME"] = GROUP
        plan["label"] = "qmix_soft_onefill_p10"
        rename(plan, "smac_5m6m_qmix_soft_onefill_p10_10m_s{}_soft20261021".format(plan["seed"]))
        plan["sbatch_args"] = ["--time=1-00:00:00" if a.startswith("--time=") else a for a in plan["sbatch_args"]]
        plan["inventory_note"] = (
            "Deterministic per-state-scalar sigmoid gate p0=0.1; x=p*state+(1-p) in train/target/eval; "
            "single complete mixer, exact autograd, joint historical TD only; no sampling/ST/temperature/threshold; "
            "no KL/MC/extra gate loss; unchanged historical GRU + two-layer ELU action head; 3 seeds, 10M, 24h")
        plans.append(plan)
    if len(plans) != 3 or {p["seed"] for p in plans} != {1, 2, 3}:
        raise RuntimeError("Expected exactly three new 5m6m soft one-fill runs")
    return plans


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_installed_map)


if __name__ == "__main__":
    main()
