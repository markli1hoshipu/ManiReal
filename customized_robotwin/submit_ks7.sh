#!/bin/bash
set -euo pipefail
TEMPLATE=scripts/study_task.sbatch

TASKS=(
  move_hamburger_onto_plate_ks
  pick_apple_from_bowl_ks
  pick_apple_from_sink_ks
  pick_fork_from_sink_ks
  put_bowl_in_sink_ks
  put_plate_in_sink_ks
  put_spoon_in_dishrack_ks
)
CONFIGS=(bench_demo_kitchens_clean)
for d in $(seq 6 15); do CONFIGS+=(bench_demo_kitchens_d${d}); done

count=0
for task in "${TASKS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    sbatch --job-name="rt_${task}_${cfg##bench_demo_}" --parsable \
      "$TEMPLATE" "$task" "$cfg" > /dev/null
    count=$((count + 1))
  done
done
echo "Submitted $count jobs (${#TASKS[@]} tasks x ${#CONFIGS[@]} configs)"
