from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
    from .grf import (Academy_3_vs_1_with_Keeper, 
                      Academy_Counterattack_Easy,
                      Academy_Pass_and_Shoot_with_Keeper)
except:
    gfootball = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

def sc2_env(**kwargs):
    noise = kwargs.pop('mixer_state_noise', False)
    noise_seed = kwargs.pop('mixer_state_noise_seed', 271828)
    noise_stream = kwargs.pop('mixer_state_noise_stream', 0)
    if type(noise) is not bool:
        raise ValueError('mixer_state_noise must be boolean')
    if noise:
        from .starcraft_mixer_noise import MixerNoiseStarCraft2Env
        return MixerNoiseStarCraft2Env(noise_seed=noise_seed, noise_stream=noise_stream, **kwargs)
    return StarCraft2Env(**kwargs)

REGISTRY["sc2"] = sc2_env

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)
    REGISTRY["academy_3_vs_1_with_keeper"] = partial(env_fn, env=Academy_3_vs_1_with_Keeper)
    REGISTRY["academy_counterattack_easy"] = partial(env_fn, env=Academy_Counterattack_Easy)
    REGISTRY["academy_pass_and_shoot_with_keeper"] = partial(env_fn, env=Academy_Pass_and_Shoot_with_Keeper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
