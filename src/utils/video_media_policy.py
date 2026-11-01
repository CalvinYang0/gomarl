"""Post-merge media policy; deliberately does not alter models or evaluation."""


def apply_video_media_policy(config):
    if config.get("disable_video_media", True):
        # Keep non-video images, importance probes and normal test episodes.
        for key in ("test_battle_videos", "battle_trace_make_video",
                    "wandb_test_trajectory_video"):
            config[key] = False
    return config
