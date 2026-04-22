"""
Batch runner: runs test_collision_metrics.py for every office/study task,
saves video + collision data under <output_dir>/<task_name>/.

USAGE:
    cd customized_robotwin
    source set_env.sh
    python script/bench_script/run_collision_metrics_all.py [options]

EXAMPLES:
    python script/bench_script/run_collision_metrics_all.py
    python script/bench_script/run_collision_metrics_all.py --output-dir ./col_results --seed 0
    python script/bench_script/run_collision_metrics_all.py --tasks mouse_on_pad move_cup
    python script/bench_script/run_collision_metrics_all.py --subdir office
"""
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path

# ── task registry ──────────────────────────────────────────────────────────────
OFFICE_TASKS = [
    "grab_battery",
    "grab_book",
    "items_to_shelf",
    "milktea_to_laptop",
    "milktea_to_shelf",
    "mouse_on_pad",
    "pencup_on_pad",
    "place_phone_desk",
    "place_phone_holder",
    "stack_book",
]

STUDY_TASKS = [
    "empty_box",
    "lift_cup_from_book",
    "lift_cup_from_box",
    "lift_pen_from_pencup",
    "move_cup_next_to_book",
    "move_cup_put_pen_in_cup",
    "move_cup",
    "move_cups_into_box",
    "move_seal_cup_next_to_box",
    "move_seal_next_to_box",
    "move_seal_onto_book",
    "move_seal_onto_table",
    "put_cup_in_box",
    "put_cup_on_coaster",
    "put_glue_in_box",
    "put_pen_in_box",
    "put_pen_in_pencup",
    "put_seal_in_box",
    "take_book_from_bookcase",
]

# task_name -> bench_subdir
TASK_SUBDIR: dict[str, str] = {t: "office" for t in OFFICE_TASKS}
TASK_SUBDIR.update({t: "study" for t in STUDY_TASKS})


def _col(code: str, text: str) -> str:
    codes = {"GREEN": "32", "RED": "31", "YELLOW": "33", "CYAN": "36", "BOLD": "1"}
    return f"\033[{codes.get(code, '0')}m{text}\033[0m"


def run_task(task_name: str, subdir: str, task_config: str, seed: int,
             base_output: Path, capture_every: int, contact_image_every: int,
             timeout: int) -> dict:
    """Run test_collision_metrics.py for one task. Returns a result summary dict."""
    task_output = base_output / task_name
    task_output.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parent / "test_collision_metrics.py"
    cmd = [
        sys.executable, str(script),
        task_name, task_config,
        "--seed", str(seed),
        "--output-dir", str(task_output),
        "--bench-subdir", subdir,
        "--capture-every", str(capture_every),
        "--contact-image-every", str(contact_image_every),
        "--render-freq", "0",
    ]

    print(_col("CYAN", f"\n{'='*60}"))
    print(_col("BOLD", f"  [{subdir}] {task_name}"))
    print(_col("CYAN", f"{'='*60}"))
    print(f"  output → {task_output}")
    print(f"  cmd: {' '.join(cmd[2:])}\n")

    log_file = task_output / "run.log"
    result = {"task": task_name, "subdir": subdir, "status": "unknown",
              "output_dir": str(task_output)}

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
        # stream output to console AND write to log
        print(proc.stdout)
        log_file.write_text(proc.stdout)

        if proc.returncode == 0:
            result["status"] = "ok"
            print(_col("GREEN", f"  ✓ {task_name} finished"))
        else:
            result["status"] = "error"
            result["returncode"] = proc.returncode
            print(_col("RED", f"  ✗ {task_name} exited with code {proc.returncode}"))

        # parse metrics json if present
        metrics_path = task_output / "collision_metrics.json"
        if metrics_path.exists():
            result["metrics"] = json.loads(metrics_path.read_text())

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        print(_col("RED", f"  ✗ {task_name} TIMED OUT after {timeout}s"))
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        print(_col("RED", f"  ✗ {task_name} exception: {e}"))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run test_collision_metrics.py for all (or selected) tasks"
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Subset of task names to run (default: all)")
    parser.add_argument("--subdir", choices=["office", "study"], default=None,
                        help="Only run tasks from this subdir")
    parser.add_argument("--task-config", default="bench_demo_clean",
                        help="Task config name (default: bench_demo_clean)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./collision_metrics_results")
    parser.add_argument("--capture-every", type=int, default=5,
                        help="Capture video frame every N physics steps")
    parser.add_argument("--contact-image-every", type=int, default=50,
                        help="Save contact debug image every N steps")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-task timeout in seconds (default: 600)")
    args = parser.parse_args()

    base_output = Path(args.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    # Build task list
    all_tasks = [(t, s) for t, s in TASK_SUBDIR.items()]
    if args.subdir:
        all_tasks = [(t, s) for t, s in all_tasks if s == args.subdir]
    if args.tasks:
        requested = set(args.tasks)
        unknown = requested - TASK_SUBDIR.keys()
        if unknown:
            print(_col("RED", f"Unknown tasks: {unknown}"))
            sys.exit(1)
        all_tasks = [(t, s) for t, s in all_tasks if t in requested]

    print(_col("BOLD", f"\nRunning collision metrics for {len(all_tasks)} tasks"))
    print(f"Output root: {base_output}")
    print(f"Config: {args.task_config}  seed: {args.seed}")

    results = []
    for task_name, subdir in all_tasks:
        r = run_task(
            task_name=task_name,
            subdir=subdir,
            task_config=args.task_config,
            seed=args.seed,
            base_output=base_output,
            capture_every=args.capture_every,
            contact_image_every=args.contact_image_every,
            timeout=args.timeout,
        )
        results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(_col("CYAN", f"\n{'='*60}"))
    print(_col("BOLD", "  SUMMARY"))
    print(_col("CYAN", f"{'='*60}"))

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    for r in ok:
        metrics_str = ""
        if "metrics" in r:
            m = r["metrics"]
            metrics_str = (
                f"  robot↔furniture={m.get('robot_to_furniture','-')}"
                f"  robot↔static={m.get('robot_to_static_object','-')}"
                f"  target↔static={m.get('target_to_static_object','-')}"
            )
        print(_col("GREEN", f"  ✓ [{r['subdir']}] {r['task']}") + metrics_str)

    for r in failed:
        print(_col("RED", f"  ✗ [{r['subdir']}] {r['task']}  ({r['status']})"))

    # Save summary json
    summary_path = base_output / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary saved: {summary_path}")
    print(f"Done. {len(ok)}/{len(results)} tasks succeeded.")


if __name__ == "__main__":
    main()
