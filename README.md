# RoboPRO

**P**erturbation-**R**esilient **O**bstacle-awareness — a bimanual manipulation benchmark for policy robustness evaluation.

**Website:** https://markli1hoshipu.github.io/RoboPRO/

RoboPRO extends the RoboTwin simulation framework with:
- **Realistic scenes** across office, study, kitchen (small & large) domains
- **Systematic perturbation suite** — Language, Vision, and Object axes for evaluating policy robustness
- **Aloha-Agilex** bimanual embodiment with CuRobo motion planning

## Installation

```bash
git clone https://github.com/markli1hoshipu/RoboPRO.git
cd RoboPRO
```

Follow the RoboTwin install guide: https://robotwin-platform.github.io/doc/usage/robotwin-install.html

Then:

1. Copy `benchmark/bench_assets/*` into `customized_robotwin/assets/objects_bench/`.
2. Overlay `benchmark/bench_assets/embodiments/aloha-agilex/*.yml` onto `customized_robotwin/assets/embodiments/aloha-agilex/` and update paths in `curobo_left.yml` / `curobo_right.yml`.
3. Fetch the large `box2_Link.dae` mesh (not tracked here — see `docs/install.md`).

### CuRobo cache patch

In `customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py`, replace `clear_cache` with:

```python
def clear_cache(self):
    self._wp_mesh_cache = {}
    if self._mesh_tensor_list is not None:
        self._mesh_tensor_list[2][:] = 0
    if self._env_n_mesh is not None:
        self._env_n_mesh[:] = 0
    if self._env_mesh_names is not None:
        for i in range(self.n_envs):
            for j in range(len(self._env_mesh_names)):
                self._env_mesh_names[i][j] = None
    super().clear_cache()
```

## Usage

```bash
cd customized_robotwin
source set_env.sh
export ROBOTWIN_BENCH_TASK="bench"
```

Collect data for a task/config pair:

```bash
bash collect_data.sh <task_name> <task_config> <gpu_id>
# Example:
bash collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
```

## Perturbation configs

Drop-in YAMLs under `benchmark/bench_task_config/`:

| Config | What it perturbs |
|---|---|
| `bench_demo_*_clean.yml` | Baseline (no perturbation) |
| `bench_demo_language.yml` | Per-episode instruction sampled from `instruction_bank.json` |
| `bench_demo_vision.yml` | Lighting (L1–L4) + blur (cycle 5 types) + per-frame pixel shake |
| `bench_demo_vision_lighting.yml` | Lighting only |
| `bench_demo_vision_blur.yml` | Blur only |
| `bench_demo_vision_shake.yml` | Pixel shake only |
| `bench_demo_object.yml` | Target texture swap + unseen obstacles + background_plus |

See `docs/perturbations.md` for parameter-level details, and `presentation_material/perturbation videos/` for sample outputs.

## Scenes and tasks

| Scene | Tasks |
|---|---|
| Office | `put_mouse_on_pad`, `put_phone_on_holder`, `put_book_on_book`, `put_book_in_fileholder`, `put_milktea_on_shelf`, `put_stapler_in_drawer`, `open_drawer`, `close_drawer`, ... |
| Study | `put_book_on_stand`, `put_pen_in_cup`, ... |
| Kitchen (Small) | `put_dish_in_rack`, `place_in_sink`, ... |
| Kitchen (Large) | `microwave_heat`, `fridge_store`, ... |

Full list in `benchmark/bench_envs/`.

## New tasks

1. Write the task env under `benchmark/bench_envs/<scene>/<task>.py`.
2. Add `_eval_step_lim.yml` entry under `benchmark/bench_task_config/`.
3. Add a description template under `benchmark/bench_description/task_instructions/`.

Naming tip: never reuse an existing RoboTwin task name. Start from an analogous sibling task (`kitchenl/`, `office/`, `study/`) — copying a proven recipe is faster than inventing from scratch.

## License

See `LICENSE`.
