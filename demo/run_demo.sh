#!/bin/bash
# End-to-end demo on the shipped d = 5 heterogeneous dataset and ADAG checkpoint.
# Run from the repository root:  bash demo/run_demo.sh
set -e
CKPT=demo/checkpoints/adag_d5_n1000_general.ckpt
DATA="--data_root demo/data --level hetero --graph_type ER --s0 1 --d 5 --n 1000"

echo "== 1. zero-shot evaluation of the shipped ADAG model on the 1000 held-out domains"
python evaluate_adag.py --ckpt $CKPT $DATA --out_dir demo/results/adag

echo "== 2. the same domains with NOTEARS and DAGMA (CPU), and the comparison table"
python run_baselines.py --methods adag notears dagma --ckpt $CKPT $DATA --out_dir demo/results

echo "== 3. (optional) a short supervised training run on a freshly generated small dataset"
python generate_data.py --level hetero --s0 1 --d 5 --M 6000 --seed 7 --root data
python train_adag_supervised.py --data_root data --level hetero --s0 1 --d 5 --M 6000 --data_seed 7 \
    --n 100 --M_train 5000 --epochs 20 --step_size 10 --out_dir checkpoints
