import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.battle_trace import render_battle_trace


def main():
    parser = argparse.ArgumentParser(description="Render a saved SMAC battle trace.")
    parser.add_argument("trace_json")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--similarity-sample-size", type=int, default=256)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    with open(args.trace_json, "r") as f:
        trace = json.load(f)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.trace_json))
    prefix = args.prefix or os.path.splitext(os.path.basename(args.trace_json))[0].replace("_trace", "")
    paths = render_battle_trace(
        trace,
        output_dir,
        prefix,
        frame_stride=args.frame_stride,
        fps=args.fps,
        make_video=not args.no_video,
        similarity_sample_size=args.similarity_sample_size,
    )
    for key, path in sorted(paths.items()):
        print("{}: {}".format(key, path))


if __name__ == "__main__":
    main()
