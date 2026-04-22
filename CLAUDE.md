# RoboTwin Bench - Session Notes

## Environment
- Always operate within a conda venv under miniforge3. Use the `RoboTwin` conda environment.
- Activate with: `conda run -n RoboTwin` or `conda activate RoboTwin`
- All pip installs and Python commands must run inside this venv.

## Writing new tasks
- When a task is hard to write or a plan fails, before inventing from scratch, look at analogous tasks in the sibling benchmarks for reference recipes:
  - `benchmark/bench_envs/kitchenl/` (kitchenLarge)
  - `benchmark/bench_envs/office/`
  - `benchmark/bench_envs/study/`
  These cover most manipulation patterns (pick/place, sink/rack/microwave, tool use, multi-step chains). Copying and adapting a proven recipe is almost always faster than debugging a novel one.
