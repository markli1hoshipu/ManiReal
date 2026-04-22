#!/bin/bash
#SBATCH --job-name=rtwks7
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/shared_work/robotwin_bench/customized_robotwin/logs/ks7-%A_%a.out
#SBATCH --error=/shared_work/robotwin_bench/customized_robotwin/logs/ks7-%A_%a.err
#SBATCH --array=0-76

set -eo pipefail

TASKS=(move_hamburger_onto_plate_ks pick_apple_from_bowl_ks pick_apple_from_sink_ks pick_fork_from_sink_ks put_bowl_in_sink_ks put_plate_in_sink_ks put_spoon_in_dishrack_ks)
CFGS=(bench_demo_clean bench_demo_kitchens_d6 bench_demo_kitchens_d7 bench_demo_kitchens_d8 bench_demo_kitchens_d9 bench_demo_kitchens_d10 bench_demo_kitchens_d11 bench_demo_kitchens_d12 bench_demo_kitchens_d13 bench_demo_kitchens_d14 bench_demo_kitchens_d15)

N=${#CFGS[@]}
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / N))]}
CFG=${CFGS[$((SLURM_ARRAY_TASK_ID % N))]}

echo "[$(date)] job=$SLURM_ARRAY_JOB_ID:$SLURM_ARRAY_TASK_ID node=$(hostname) task=$TASK config=$CFG"

source /shared_work/miniforge3/etc/profile.d/conda.sh
conda activate /shared_work/miniforge3/envs/RoboTwin

cd /shared_work/robotwin_bench/customized_robotwin
source set_env.sh >/dev/null
export ROBOTWIN_BENCH_TASK=bench
export PYTHONWARNINGS=ignore::UserWarning

python -u script/collect_data.py "$TASK" "$CFG"
rm -rf "data/bench_data/$TASK/$CFG/.cache"
echo "[$(date)] done $TASK/$CFG"
