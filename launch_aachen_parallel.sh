#!/usr/bin/env bash
# Launch nc2 / nc3 / nc4 / mambaglue Aachen runs in parallel: one tmux session + one GPU each.
# Detached tmux => jobs run on the server and survive SSH/internet drops.
# Usage:  bash launch_aachen_parallel.sh
set -euo pipefail

REPO=/home/ubuntu/work/MambaGlue/Hierarchical-Localization
CKPT_DIR=/home/ubuntu/work/MambaGlue/glue-factory/outputs/training
CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
ENV=hlocaachen

mkdir -p "$REPO/logs"

# spec = tag : gpu : matcher_name : n_cross_layers : checkpoint_path
#   (leave n_cross_layers empty for mambaglue)
RUNS=(
  "nc2:0:latemambaglue:2:$CKPT_DIR/latemambaglueMDnc2/checkpoint_best.tar"
  "nc3:1:latemambaglue:3:$CKPT_DIR/latemambaglueMDnc3/checkpoint_best.tar"
  "nc4:2:latemambaglue:4:$CKPT_DIR/latemambaglueMDnc4/checkpoint_best.tar"
  "mg:3:mambaglue::$CKPT_DIR/sp+mg_megadepth/checkpoint_best.tar"
)

for spec in "${RUNS[@]}"; do
  IFS=":" read -r tag gpu mname ncl ckpt <<< "$spec"
  session="aachen_${tag}"

  if [[ ! -f "$ckpt" ]]; then
    echo "!! SKIP $tag: checkpoint not found -> $ckpt"; continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "!! SKIP $tag: tmux session '$session' already exists (kill it first)"; continue
  fi

  ncl_arg=""
  [[ -n "$ncl" ]] && ncl_arg="--n_cross_layers $ncl"

  cmd="source $CONDA_SH && conda activate $ENV && cd $REPO && \
CUDA_VISIBLE_DEVICES=$gpu python run_aachen_lmg.py \
--dataset datasets/aachen_v1.1 \
--outputs outputs/aachen_v1.1_${tag} \
--checkpoint $ckpt \
--matcher_name $mname $ncl_arg --tag $tag \
--num_covis 20 --num_loc 50 \
2>&1 | tee logs/aachen_${tag}.log"

  tmux new-session -d -s "$session"
  tmux send-keys -t "$session" "$cmd" Enter
  echo "launched $session  (GPU $gpu, matcher=$mname ${ncl:+nc=$ncl})"
done

echo; echo "Sessions:"; tmux ls