#!/usr/bin/env python3
"""Check all plotted slots and legacy-run exclusions without W&B access."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ozstar_submit_linear_directkl_four_model_3seeds import (  # noqa: E402
    MODELS, build_plans,
)
from plot_linear_directkl_four_model_3seeds import TITLES  # noqa: E402


def main():
    plans = build_plans(ROOT)
    assert len(plans) == 48
    assert len({(plan["scene"], plan["label"], plan["seed"])
                for plan in plans}) == 48
    assert set(TITLES) == {plan["scene"] for plan in plans}
    assert {label for label, _ in MODELS} == {plan["label"] for plan in plans}
    assert all(plan["exports"]["T_MAX"] == "5050000" for plan in plans)
    assert not any("controlled15" in candidate for plan in plans
                   for candidate in plan["historical_candidates"])
    for plan in plans:
        if plan["scene"] == "grf_counter" and plan["seed"] == 1:
            assert plan["historical_candidates"]
    print("four-map Linear seed-plot inventory smoke test passed")


if __name__ == "__main__":
    main()
