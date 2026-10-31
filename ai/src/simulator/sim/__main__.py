"""The simulator's command line. Run it from ai/src/simulator.

    python -m sim check scenarios/yellow_ball.py [--seed S] [--episodes N]
        Run a Scenario's physics on your laptop, without rendering: catches
        mistakes, prints what was drawn, and says how often each camera sees
        each labeled object.

    python -m sim run scenarios/yellow_ball.py [--seed S] [--out DIR]
        Simulate and render one episode, then make its previews. Rendering
        needs bpy (pip install bpy==4.5.4, Python 3.11). In the cloud,
        `wildhat launch --episodes N -- python -m sim run ...` runs it with
        each episode's seed, and the outputs go to the bucket.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sim import runner


def cmd_check(args) -> int:
    # Only MuJoCo and numpy, so it runs on any laptop.
    from sim.world import jsonable

    scenario, name = runner.load(args.scenario)
    for i in range(args.episodes):
        seed = args.seed + i
        episode = runner.run(scenario, seed, "physics")
        world = episode.world
        print(f"{name}, seed {seed}: {episode.frames} frames in {episode.seconds:.1f}s")
        if i == 0:
            for item_name, settings in world.drawn.items():
                print(f"  {item_name}: {json.dumps(jsonable(settings))}")
        for cam, seen in episode.in_view.items():
            for obj, frames in seen.items():
                print(f"  {cam} sees {obj} in {frames}/{episode.frames} frames")
        if not world.cameras:
            print("  There are no cameras, so nothing would be rendered.")
    return 0


def cmd_run(args) -> int:
    from sim import records
    from sim.logger import Logger
    from sim.postprocess import PostProcessor

    scenario, name = runner.load(args.scenario)
    seed = args.seed if args.seed is not None else int(os.environ.get("WILDHAT_SEED", "0"))
    out = Path(args.out or os.environ.get("WILDHAT_OUTPUT_DIR") or f"out/{Path(name.split(':')[0]).stem}/seed-{seed}")
    info = {"scenario": name, "scene": scenario.scene_class().__name__, "backend": args.backend}
    logger = None

    def make_logger(world):
        nonlocal logger
        logger = Logger(out, world, info, render=args.backend != "physics")
        return logger

    print(f"sim: {name}, seed {seed}, {args.backend} backend, writing to {out}", flush=True)
    result = runner.run(scenario, seed, args.backend, make_logger, progress=True)
    episode = logger.finish()
    previews = PostProcessor(out, episode).run()
    print(f"sim: {result.frames} frames in {result.seconds:.0f}s; made {len(previews)} previews", flush=True)
    records.record(episode, previews)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sim", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="run a Scenario's physics locally, without rendering")
    check.add_argument("scenario", help="e.g. scenarios/yellow_ball.py, or file.py:ClassName")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--episodes", type=int, default=1, help="check this many seeds, from --seed up")
    check.set_defaults(func=cmd_check)
    run = sub.add_parser("run", help="simulate and render one episode")
    run.add_argument("scenario", help="e.g. scenarios/yellow_ball.py, or file.py:ClassName")
    run.add_argument("--seed", type=int, help="default: $WILDHAT_SEED, or 0")
    run.add_argument("--out", help="default: $WILDHAT_OUTPUT_DIR, or out/<scenario>/seed-<seed>")
    run.add_argument("--backend", default="blender", choices=["blender", "physics"])
    run.set_defaults(func=cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
