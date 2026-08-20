import datetime
import os
import pprint
import time
import threading
import shutil
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.battle_trace import save_battle_trace, render_battle_trace
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def _write_battle_trace_outputs(args, logger, trace):
    if trace is None:
        logger.console_logger.warning("Battle trace was requested but runner returned no trace.")
        return

    output_dir = getattr(args, "battle_trace_dir", "")
    if output_dir in [None, ""]:
        output_dir = os.path.join(args.local_results_path, "battle_traces", args.unique_token)
    prefix = _safe_name(trace.get("prefix", "battle_trace"))

    trace_path = save_battle_trace(trace, output_dir, prefix)
    paths = {"trace_json": trace_path}
    paths.update(
        render_battle_trace(
            trace,
            output_dir,
            prefix,
            frame_stride=int(getattr(args, "battle_trace_frame_stride", 4)),
            fps=int(getattr(args, "battle_trace_fps", 6)),
            make_video=bool(getattr(args, "battle_trace_make_video", True)),
            similarity_sample_size=int(getattr(args, "battle_trace_similarity_sample_size", 256)),
        )
    )

    logger.console_logger.info("Battle trace saved to {}".format(trace_path))
    if "video" in paths:
        logger.console_logger.info("Rendered trajectory video saved to {}".format(paths["video"]))
    elif bool(getattr(args, "battle_trace_make_video", True)):
        logger.console_logger.warning("No rendered trajectory video was produced for this trace.")
    if bool(getattr(args, "battle_trace_upload_wandb", True)):
        logger.log_battle_trace_media(paths, int(trace.get("t_env", 0)), fps=int(getattr(args, "battle_trace_fps", 6)))


def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config,
            getattr(args, "wandb_team", None),
            getattr(args, "wandb_project", None),
            getattr(args, "wandb_mode", "offline"),
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish external logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    if str(getattr(args, "clean_model_type", "")).endswith(
        "dual_branch_binary_concrete_adaptive_trajectory_parameter_likelihood_hypercond"
    ):
        scheme["trajectory_parameter_projection"] = {
            "vshape": (
                int(getattr(args, "clean_trajectory_parameter_projection_dim", 64)),
            ),
            "group": "agents",
            "dtype": th.float16,
        }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    last_battle_trace_T = 0
    test_video_written = False

    start_time = time.time()
    last_time = start_time
    learner_updates_per_collect = max(
        1, int(getattr(args, "learner_updates_per_collect", 1))
    )

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    logger.console_logger.info(
        "Rollout configuration: {} env workers, {} learner update(s) per collection".format(
            args.batch_size_run, learner_updates_per_collect
        )
    )

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            # A larger rollout pool improves SC2 throughput. Repeat learner updates
            # so changing the number of environment workers does not silently reduce
            # the update-per-episode ratio of the original experiment setting.
            for _ in range(learner_updates_per_collect):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)
                del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            trace_interval = int(getattr(args, "battle_trace_interval", 1000000))
            trace_due = (
                trace_interval > 0
                and runner.t_env - last_battle_trace_T >= trace_interval
            )
            save_periodic_trace = bool(getattr(args, "save_battle_trace", False))
            save_one_test_video = (
                bool(getattr(args, "wandb_test_trajectory_video", False))
                and bool(getattr(args, "use_wandb", False))
                and not test_video_written
            )
            should_trace = (
                trace_due
                and (save_periodic_trace or save_one_test_video)
                and hasattr(runner, "request_battle_trace")
            )
            trace_prefix = None
            if should_trace:
                map_name = getattr(args, "env_args", {}).get("map_name", getattr(args, "env", "env"))
                trace_prefix = "{}_{}_t{}".format(args.name, map_name, runner.t_env)
                logger.console_logger.info("Collecting battle trace at t_env={}".format(runner.t_env))

            for test_run_idx in range(n_test_runs):
                if should_trace and test_run_idx == 0:
                    runner.request_battle_trace(prefix=trace_prefix, t_env=runner.t_env)
                runner.run(test_mode=True)
                if should_trace and test_run_idx == 0:
                    _write_battle_trace_outputs(args, logger, runner.pop_battle_trace())
                    last_battle_trace_T = runner.t_env
                    if save_one_test_video:
                        test_video_written = True

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(logger.wandb.dir, "models", args.unique_token, str(runner.t_env))
                os.makedirs(wandb_save_dir, exist_ok=True)
                for file_name in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, file_name),
                        os.path.join(wandb_save_dir, file_name),
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
