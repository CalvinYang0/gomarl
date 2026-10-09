"""Periodic, unselected SMAC evaluation videos from real simulator snapshots.

No additional evaluation episodes, model forwards, RNG draws or training data.
Frames show the state BEFORE the displayed joint action and include the final
state. RGB frames are streamed to the encoder, never retained as a PNG tree.
"""
import json
import math
import subprocess
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np


def action_label(action, attack_offset=6, can_heal=False):
    if action is None:
        return "terminal"
    action = int(action)
    labels = {0: "no-op", 1: "stop", 2: "north", 3: "south", 4: "east", 5: "west"}
    return labels.get(action, ("heal A{}" if can_heal else "attack E{}").format(action - attack_offset))


def focus_groups(frame):
    snapshot = frame.get("snapshot") or {}
    alive = {u["id"] for u in snapshot.get("allies", []) if u.get("alive")}
    healers = {u["id"] for u in snapshot.get("allies", []) if u.get("can_heal")}
    offset = int(snapshot.get("n_actions_no_attack", 6))
    groups = {}
    for agent, action in enumerate(frame.get("actions") or []):
        if agent in alive and agent not in healers and int(action) >= offset:
            groups.setdefault(int(action) - offset, []).append(agent)
    return groups


@lru_cache(maxsize=8)
def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        # Matplotlib is already a training dependency and ships this font.
        import matplotlib
        try:
            return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)
        except OSError:
            return ImageFont.load_default()


def video_bounds(frames):
    units = [u for frame in frames for side in ("allies", "enemies")
             for u in (frame.get("snapshot") or {}).get(side, [])]
    if not units:
        raise ValueError("No real unit snapshots were recorded")
    xs, ys = [float(u["x"]) for u in units], [float(u["y"]) for u in units]
    pad = max(2., .08 * max(max(xs) - min(xs), max(ys) - min(ys)))
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad


def draw_video_frame(frame, trace, bounds):
    from PIL import Image, ImageDraw
    snapshot = frame.get("snapshot")
    if not snapshot:
        raise ValueError("Missing snapshot at decision step {}".format(frame.get("t")))
    line_height = min(31, max(15, 390 // max(1, len(snapshot["allies"]))))
    # Preserve all agent/target rows on larger maps, not just the Marine probes.
    height = max(720, 150 + len(snapshot["allies"])*line_height + 29
                 + len(snapshot["enemies"])*21 + 48)
    height = 16 * math.ceil(height / 16)
    image = Image.new("RGB", (1120, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font, small, title = _font(15), _font(12), _font(19)
    x0, x1, y0, y1 = bounds
    scale = min(620. / (x1 - x0), 568. / (y1 - y0))
    left, top = 38., 100.

    def point(unit):
        return left + (float(unit["x"]) - x0) * scale, top + (y1 - float(unit["y"])) * scale

    draw.text((24, 18), "{} | seed {} | eval episode {:02d}".format(
        trace.get("map_name", "SMAC"), trace.get("seed", "?"), trace["episode_index"] + 1), font=title, fill="#0f172a")
    draw.text((24, 49), "training step {:,} | decision step {} | result: {}".format(
        trace["t_env"], frame["t"], "WIN" if trace.get("battle_won") else "LOSS / TIME LIMIT"), font=font, fill="#334155")
    draw.text((24, 75), "Pre-action state + chosen simultaneous commands (arrows are NOT confirmed hits)", font=small, fill="#475569")
    if trace.get("synthetic"):
        draw.text((720, 75), "SYNTHETIC TEST - not experiment data", font=small, fill="#dc2626")
    draw.rectangle((left, top, left + (x1-x0)*scale, top + (y1-y0)*scale), fill="#f1f0e8", outline="#cbd5e1")
    for x in range(math.ceil(x0), math.floor(x1) + 1, 2):
        px = left + (x-x0)*scale
        draw.line((px, top, px, top+(y1-y0)*scale), fill="#e2e3dc")
        draw.text((px, top+(y1-y0)*scale+6), str(x), font=small, fill="#64748b")
    for y in range(math.ceil(y0), math.floor(y1) + 1, 2):
        py = top+(y1-y)*scale
        draw.line((left, py, left+(x1-x0)*scale, py), fill="#e2e3dc")
        draw.text((9, py-5), str(y), font=small, fill="#64748b")
    allies = {u["id"]: u for u in snapshot["allies"]}
    enemies = {u["id"]: u for u in snapshot["enemies"]}
    actions = frame.get("actions")
    offset = int(snapshot.get("n_actions_no_attack", 6))
    palette = ("#2563eb", "#0891b2", "#7c3aed", "#16a34a", "#ea580c", "#db2777", "#4f46e5", "#0d9488")
    if actions is not None:
        for agent, action in enumerate(actions):
            unit = allies.get(agent)
            if not unit or not unit.get("alive"):
                continue
            start = point(unit)
            end = None
            targets = allies if unit.get("can_heal") else enemies
            if int(action) >= offset and int(action)-offset in targets:
                end = point(targets[int(action)-offset])
            elif int(action) in (2, 3, 4, 5):
                dx, dy = {2: (0, -24), 3: (0, 24), 4: (24, 0), 5: (-24, 0)}[int(action)]
                end = start[0]+dx, start[1]+dy
            if end is not None:
                color = palette[agent % len(palette)]
                draw.line((*start, *end), fill=color, width=2)
                angle = math.atan2(end[1]-start[1], end[0]-start[0])
                draw.polygon([end, (end[0]-9*math.cos(angle-.4), end[1]-9*math.sin(angle-.4)),
                              (end[0]-9*math.cos(angle+.4), end[1]-9*math.sin(angle+.4))], fill=color)
    for prefix, units in (("E", enemies), ("A", allies)):
        for unit_id, unit in sorted(units.items()):
            x, y = point(unit)
            color = "#dc2626" if prefix == "E" else palette[unit_id % len(palette)]
            if not unit.get("alive"):
                color = "#a1a1aa"
            shape = draw.rectangle if prefix == "E" else draw.ellipse
            shape((x-7, y-7, x+7, y+7), fill=color, outline="#0f172a")
            draw.text((x+9, y-17), "{}{} HP {:g}".format(prefix, unit_id, unit["health"]), font=small, fill=color)
            ratio = max(0., min(1., unit["health"] / max(unit["health_max"], 1.)))
            draw.rectangle((x-13, y+10, x+13, y+14), fill="#d1d5db")
            if ratio > 0:
                draw.rectangle((x-13, y+10, x-13+26*ratio, y+14), fill=color)
    panel = 720
    draw.text((panel, 100), "Agents: current HP and chosen action", font=font, fill="#0f172a")
    for row, (agent, unit) in enumerate(sorted(allies.items())):
        action = None if actions is None else actions[agent]
        label = "DEAD" if not unit.get("alive") else action_label(action, offset, unit.get("can_heal", False))
        shield = " S {:g}".format(unit.get("shield", 0)) if unit.get("shield_max", 0) else ""
        draw.text((panel, 135+row*line_height), "A{}  HP {:g}/{:g}{}  {}".format(
            agent, unit["health"], unit["health_max"], shield, label), font=small if line_height < 23 else font,
            fill=palette[agent % len(palette)] if unit.get("alive") else "#64748b")
    group_y = 150 + len(allies)*line_height
    draw.text((panel, group_y), "Focus fire: chosen targets", font=font, fill="#0f172a")
    groups = focus_groups(frame)
    for row, (enemy, agents) in enumerate(sorted(groups.items())):
        draw.text((panel, group_y+29+row*21), "E{} <- {} ({} agents)".format(
            enemy, ",".join("A{}".format(a) for a in agents), len(agents)), font=small, fill="#334155")
    if not groups:
        draw.text((panel, group_y+29), "No selected attacks / terminal", font=small, fill="#64748b")
    draw.text((24, height-20), "Blue/categorical circles: allies | red squares: enemies | coordinates and HP from simulator", font=small, fill="#475569")
    return np.asarray(image)


def render_battle_video(trace, output_path, fps=6):
    import imageio.v2 as imageio
    frames = trace["frames"]
    if len(frames) < 2 or any(not f.get("snapshot") for f in frames):
        raise ValueError("Video requires every pre-action snapshot and the terminal snapshot")
    bounds = video_bounds(frames)
    # One frame in memory; no temporary PNGs or all-frame RGB arrays.
    with imageio.get_writer(str(output_path), format="FFMPEG", fps=fps,
                            codec="libx264", quality=7, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(draw_video_frame(frame, trace, bounds))
    if not Path(output_path).is_file() or Path(output_path).stat().st_size == 0:
        raise RuntimeError("Video encoder produced no output")
    return str(output_path)


def check_video_dependencies():
    """Fail early in preflight rather than discovering a missing encoder at 1M."""
    import imageio.v2  # noqa: F401
    import imageio_ffmpeg
    from PIL import Image  # noqa: F401
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   check=True, timeout=10)


class BattleVideoSession:
    """Select exactly the first N episodes from an existing normal test suite."""
    def __init__(self, args, logger):
        self.args, self.logger = args, logger
        self.enabled = bool(getattr(args, "test_battle_videos", True)) and args.env == "sc2"
        self.interval = int(getattr(args, "test_battle_video_interval", 1000000))
        self.limit = int(getattr(args, "test_battle_video_episodes", 10))
        self.fps = int(getattr(args, "test_battle_video_fps", 6))
        self.output_root = Path(getattr(args, "test_battle_video_dir", "") or
                                Path(args.local_results_path)/"test_battle_videos"/args.unique_token)
        # Names/timestamp tokens can coincide across seeds and model conditions.
        run_name = getattr(args, "wandb_run_name", None) or args.name
        run_id = sha256(run_name.encode()).hexdigest()[:12]
        self.output_root = self.output_root / "seed_{}_{}".format(args.seed, run_id)
        if self.enabled and (self.interval <= 0 or self.limit <= 0 or self.fps <= 0):
            raise ValueError("Battle video interval/episodes/fps must be positive")
        if self.enabled and self.limit > args.test_nepisode:
            raise ValueError("test_nepisode must cover all requested battle videos")
        if self.enabled:
            try:
                check_video_dependencies()
            except Exception as exc:
                self.logger.console_logger.warning(
                    "Battle videos DISABLED: encoder unavailable (%s). Install imageio>=2.9, imageio-ffmpeg and Pillow before launch.", exc)
                self.enabled = False
            self.logger.log_stat("test_battle_video/available", int(self.enabled), 0)
        self.last_milestone = 0
        self.active = False

    def begin(self, t_env):
        milestone = int(t_env) // self.interval if self.enabled else 0
        self.active = self.enabled and milestone > self.last_milestone
        self.milestone = milestone
        self.t_env = int(t_env)
        self.collected = self.rendered = 0
        self.inventory = []

    def request(self, runner):
        if self.active and self.collected < self.limit:
            runner.request_battle_videos(min(runner.batch_size, self.limit-self.collected), self.t_env, self.collected)

    def consume(self, runner):
        if not self.active:
            return
        for trace in runner.pop_battle_videos():
            trace.update(seed=int(self.args.seed), run_name=getattr(self.args, "wandb_run_name", None) or self.args.name,
                         milestone=self.milestone*self.interval)
            self.collected += 1
            row = {key: trace.get(key) for key in ("episode_index", "battle_won", "episode_return", "episode_length", "t_env", "milestone")}
            try:
                directory = self.output_root / "step_{:09d}".format(trace["milestone"])
                directory.mkdir(parents=True, exist_ok=True)
                stem = "episode_{:02d}".format(trace["episode_index"]+1)
                # Raw truth remains locally inspectable even if encoding fails.
                (directory/(stem+".json")).write_text(json.dumps(trace))
                path = render_battle_video(trace, directory/(stem+".mp4"), self.fps)
                row["video"] = path
                self.rendered += 1
                self.logger.log_test_battle_video(path, self.t_env, trace["episode_index"]+1, trace, self.fps)
            except Exception as exc:
                row["error"] = repr(exc)
                self.logger.console_logger.warning("Battle video episode %s failed: %s", trace["episode_index"]+1, exc)
            self.inventory.append(row)

    def finish(self):
        if not self.active:
            return
        self.last_milestone = self.milestone
        self.active = False
        self.logger.log_stat("test_battle_video/collected", self.collected, self.t_env)
        self.logger.log_stat("test_battle_video/rendered", self.rendered, self.t_env)
        self.logger.log_stat("test_battle_video/failed", sum("error" in row for row in self.inventory), self.t_env)
        try:
            directory = self.output_root / "step_{:09d}".format(self.milestone*self.interval)
            directory.mkdir(parents=True, exist_ok=True)
            (directory/"inventory.json").write_text(json.dumps(self.inventory, indent=2))
        except OSError as exc:
            self.logger.console_logger.warning("Battle video inventory failed: %s", exc)
        self.logger.console_logger.info("Battle videos at t_env=%s: collected=%s/%s rendered=%s", self.t_env, self.collected, self.limit, self.rendered)
