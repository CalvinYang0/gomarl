"""Fail before SC2 launch when required future-test media cannot be produced."""
import json
from pathlib import Path
from hashlib import sha256

from utils.battle_video import check_video_dependencies


def preflight_visualizations(args):
    if args.env != "sc2" or not getattr(args, "test_visualizations_required", False):
        return
    for key in ("test_battle_videos", "test_policy_importance"):
        if not getattr(args, key, False):
            raise ValueError("Required SMAC visualization disabled: " + key +
                             "; short/debug runs may explicitly set test_visualizations_required=False")
    try:
        check_video_dependencies()
        import matplotlib.pyplot  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Required SMAC visualization dependencies missing; install imageio, "
                           "imageio-ffmpeg, Pillow and matplotlib BEFORE launch") from exc
    for prefix in ("test_battle_video", "test_policy_importance"):
        if int(getattr(args, prefix + "_interval", 0)) <= 0:
            raise ValueError("Invalid visualization interval: " + prefix)
        count = int(getattr(args, prefix + "_episodes", 0))
        batch_size = int(args.batch_size_run)
        normal_episodes = max(1, args.test_nepisode // batch_size) * batch_size
        if count <= 0 or count > normal_episodes:
            raise ValueError("Normal test episodes must cover visualization episodes: " + prefix)


def record_visualization_inventory(args, logger, videos, policy, hyper):
    """Output evidence, not an assertion that offline media reached the cloud."""
    if args.env != "sc2" or not (getattr(videos, "due", False) or policy.due or hyper.due):
        return
    row = dict(t_env=policy.t_env, seed=args.seed,
               run_name=getattr(args, "wandb_run_name", None) or args.name,
               video_enabled=videos.enabled, video_collected=videos.collected,
               video_rendered=videos.rendered, video_due=getattr(videos, "due", False),
               video_failed=sum("error" in item for item in getattr(videos, "inventory", [])),
               policy_importance_due=policy.due, policy_importance_error=policy.error,
               policy_importance_files={k: str(v) for k, v in policy.files.items()},
               hyper_obs_enabled=hyper.enabled,
               hyper_obs_due=hyper.due, hyper_obs_error=hyper.error,
               hyper_obs_note="Hyper-only figures apply to ungated raw/health-only Linear Obs; "
                              "other models use whole-policy sensitivity instead",
               cloud_upload_verified=False)
    identity = sha256(row["run_name"].encode()).hexdigest()[:8]
    directory = Path(args.local_results_path) / "test_visualization_inventory" / args.unique_token / (
        "seed_{}_{}".format(args.seed, identity))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "step_{:09d}.json".format(policy.t_env)
    path.write_text(json.dumps(row, indent=2))
    required = getattr(args, "test_visualizations_required", False)
    if required and row["video_due"] and (videos.collected != videos.limit or videos.rendered != videos.limit
                                           or row["video_failed"]):
        raise RuntimeError("Required battle videos incomplete: " + str(path))
    if required and policy.due and (policy.error or not {
            "parameter_sensitivity", "observation_sensitivity"}.issubset(policy.files)):
        raise RuntimeError("Required importance figures incomplete: " + str(path))
    if required and hyper.due and hyper.error:
        raise RuntimeError("Required hyper-only Obs importance failed: " + str(path))
    logger.log_stat("test_visualization/recorded", 1, policy.t_env)
    logger.console_logger.info("Visualization output inventory: %s", path)
