"""Read-only by default. Launch is explicit and never installs dependencies."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

from .compat import inspect_source
from .policy import Settings


def launch_command(model, extra):
    # The wrapper owns these flags; allowing later duplicates can invalidate the
    # plan printed to the agent. The scheduler also validates effective config.
    owned = {"--scheduler-cls", "--scheduling-policy", "--async-scheduling",
             "--no-async-scheduling", "--host"}
    if any(arg.split("=", 1)[0] in owned for arg in extra):
        raise ValueError("scheduler, policy, async, and host flags are managed by vvip")
    return ["vllm", "serve", model, "--host", "127.0.0.1",
            "--scheduling-policy", "priority", "--no-async-scheduling",
            "--scheduler-cls", "vvip_runtime.scheduler.VVIPScheduler", *extra]


def main():
    parser = argparse.ArgumentParser(description="VVIP: portable vLLM priority preemption")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check exact source; does not import torch")
    doctor.add_argument("--source-root", type=Path, help="path to vllm package source")
    serve = sub.add_parser("serve", help="print launch plan; add --execute to launch")
    serve.add_argument("model")
    serve.add_argument("--mode", choices=["off", "shadow", "enforce"], default="off")
    serve.add_argument("--action", choices=["recompute", "abort"], default="recompute")
    serve.add_argument("--execute", action="store_true")
    # vLLM args go after --; do not silently absorb misspelled vvip arguments.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:]
    try:
        if args.command == "doctor":
            if extra:
                parser.error("doctor does not accept vLLM arguments")
            report = inspect_source(args.source_root)
            print(json.dumps(report, indent=2))
            raise SystemExit(0 if report["compatible"] else 2)
        command = launch_command(args.model, extra)
        env = dict(os.environ, VVIP_MODE=args.mode, VVIP_ACTION=args.action)
        # The copied skill's sibling module must be visible in spawned workers.
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
        os.environ["VVIP_MODE"], os.environ["VVIP_ACTION"] = args.mode, args.action
        settings = Settings.from_env()
        if not args.execute:
            display = ["env", f"PYTHONPATH={env['PYTHONPATH']}", *[
                f"VVIP_{name.upper()}={value}" for name, value in asdict(settings).items()
            ], *command]
            print(json.dumps({"execute": False, "settings": asdict(settings),
                              "argv": command, "shell_display": shlex.join(display),
                              "required_pythonpath": root,
                              "note": "Source/config gates and GPU acceptance still required"}, indent=2))
            return
        report = inspect_source()
        if not report["compatible"]:
            parser.error("; ".join(report["errors"]))
        if not shutil.which("vllm"):
            parser.error("vllm executable is missing from PATH")
        os.execvpe(command[0], command, env)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"vvip: {exc}\n")


if __name__ == "__main__":
    main()
