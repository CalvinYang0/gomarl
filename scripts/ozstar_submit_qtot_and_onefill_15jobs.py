#!/usr/bin/env python3
"""5m6m: three complete-Q_tot compositions; MMM2/corridor: old onefill_p10.

Three seeds each, 10M, 24-hour walltime. Deliberately retains the historical
qmix_minimal GRU + TWO-LAYER ELU head, NOT the corrected official baseline.
"""
import copy
import os
from pathlib import Path
import shlex

from ozstar_submit_5m6m_onefill_gate_15jobs import build_plans as onefill_plans
from ozstar_submit_qmix_vdn_calibration_gate_18jobs import rename
from ozstar_submit_8m9m_ones_gate_regularization_9jobs import submit_suite

GROUP = "qtot_composition_onefill_transfer_10m_15jobs_20261021"
QTOT_MODELS = {
    "qmix_qtot_sum_bothones": ("sum", True),
    "qmix_qtot_sum_p10": ("sum", False),
    "qmix_qtot_alpha10_p10": ("alpha_sum", False),
}
TRANSFER_MAPS = (("smac_mmm2", "MMM2", 10, 12), ("smac_corridor", "corridor", 6, 24))
SMOKE_SCRIPTS = ("scripts/smoke_test_qtot_composition.py", "scripts/smoke_test_submit_qtot_map_suite.py")


def finish(plan, label, scene, map_name, memory):
    plan.update(scene=scene, map_name=map_name, label=label, memory=memory)
    plan["exports"].update(GROUP_NAME=GROUP, MAP_NAME=map_name, MODEL_TYPE="qmix_minimal", T_MAX="10050000")
    rename(plan, "{}_{}_10m_s{}_qtot20261021".format(scene, label, plan["seed"]))
    plan["sbatch_args"] = ["--time=1-00:00:00" if arg.startswith("--time=") else
                          "--mem=" + memory if arg.startswith("--mem=") else arg
                          for arg in plan["sbatch_args"]]
    return plan


def build_plans(repo):
    originals = [p for p in onefill_plans(repo) if p["label"] == "qmix_onefill_p10"]
    if {p["seed"] for p in originals} != {1, 2, 3}:
        raise RuntimeError("Expected the historical one-fill p10 reference with three seeds")
    plans = []
    for original in originals:
        for label, (combination, both) in QTOT_MODELS.items():
            plan = copy.deepcopy(original)
            extra = dict(x.split("=", 1) for x in shlex.split(plan["exports"]["EXTRA_ARGS"]))
            extra.update(learner="qtot_composition_learner", mixer_onefill_residual="False",
                mixer_residual_combination="centered", mixer_residual_secondary_ones="False",
                mixer_residual_fill_value="1", mixer_qtot_combination=combination,
                mixer_qtot_secondary_ones=str(both), mixer_qtot_alpha_initial="0.1")
            plan["exports"]["EXTRA_ARGS"] = " ".join(k + "=" + v for k, v in extra.items())
            finish(plan, label, "smac_5m6m", "5m_vs_6m", "24G")
            plan["inventory_note"] = (
                "SAME historical GRU + two-layer ELU action head; same agent Qi feeds two COMPLETE QMIXes; "
                + ("Qtot=alpha*Qmain+(1-alpha)*Qsecondary; learned global alpha0=0.1; "
                   if combination == "alpha_sum" else "Qtot=Qmain+Qsecondary; no abs on Qtot; ")
                + ("both mixer inputs ones; no operative gate; " if both else
                   "main input ones, secondary input mask*state+(1-mask), p0=0.1; ")
                + "each branch owns W1/W2/b1/b2; independent nonzero secondary initialization; "
                "one joint historical TD loss, joint actor/mixers/gate updates, no auxiliary TD/KL/MC; 24h")
            plans.append(plan)
        for scene, map_name, _, _ in TRANSFER_MAPS:
            plan = copy.deepcopy(original)
            finish(plan, "qmix_onefill_p10", scene, map_name, "32G")
            plan["inventory_note"] = (
                "Historical single-mixer onefill_p10 copied unchanged apart from map/resource/run identity; "
                "SAME GRU + two-layer ELU head (action count follows target map); p0=0.1; "
                "TD jointly trains gate, actor, mixer; no KL/MC; 24h")
            plans.append(plan)
    if len(plans) != len({p["job_name"] for p in plans}) or len(plans) != 15:
        raise RuntimeError("Expected exactly nine 5m6m Q_tot-composition + six map-transfer jobs")
    return plans


def validate_maps():
    from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry
    registry = get_smac_map_registry()
    root = Path(os.environ.get("SC2PATH", "/home/kyang/StarCraftII")) / "Maps"
    for _, name, allies, enemies in (("smac_5m6m", "5m_vs_6m", 5, 6),) + TRANSFER_MAPS:
        params = registry.get(name, {})
        if (params.get("n_agents"), params.get("n_enemies")) != (allies, enemies):
            raise RuntimeError("Installed map unit count mismatch: " + name)
        if not any(root.rglob(name + ".SC2Map")):
            raise RuntimeError("Missing SC2 map file: " + name)


def main():
    submit_suite(GROUP, __file__, build_plans, SMOKE_SCRIPTS, map_validator=validate_maps)


if __name__ == "__main__":
    main()
