#!/bin/bash

CMD='python3 ibc/ibc/train_eval.py '
GIN='ibc/ibc/configs/d4rl/mlp_ebm_langevin_best.gin'
DATA="train_eval.dataset_path='ibc/data/d4rl_trajectories/pen-human-v0/*.tfrecord'"

$CMD -- \
  --alsologtostderr \
  --gin_file=$GIN \
  --task=pen-human-v0 \
  --tag=ibc_langevin \
  --add_time=True \
  --gin_bindings=$DATA
  # not currently calling --video because rendering is broken in the docker?