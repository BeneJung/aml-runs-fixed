#!/bin/bash
# Full v2 chain: R1 (n=6 total) + R3 + R4 + R5 + R6.
# Idempotent — re-running picks up where it left off via SKIP guards.
#
# Usage on Renku (inside a tmux pane):
#   export REPO=~/work/aml-runs-fixed
#   export DRIVE=~/work/drive
#   cd $REPO
#   bash scripts/run_chain.sh
#
# Detach with Ctrl+B D once the first training progress bar appears.

set -u   # no `set -e` — one bad run must not stop the rest of the chain

# ============ env ============
REPO=${REPO:-$HOME/work/aml-runs-fixed}
DRIVE=${DRIVE:-$HOME/work/drive}
cd "$REPO"

SEEDS_R1_EXTRA="45 46 47"   # bs1+bs4 with these — chain SKIPs already-done seeds 42-44
SEEDS_R3="42 43 44"
EPOCHS_E2=100
EPOCHS_E4=80
VAL_SAMPLES_PER_T=2
PER_RUN_TIMEOUT=7200
FROZEN_ALPHAS="0.06 0.50"

CHAIN_LOG="logs/chain_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs results/per_run_floor6

log()  { echo "$@" | tee -a "$CHAIN_LOG"; }
fail() { log "[!] $*"; }

log "=== CHAIN START $(date) ==="
log "  R1 extra seeds: $SEEDS_R1_EXTRA  R3 seeds: $SEEDS_R3"
log "  REPO=$REPO  DRIVE=$DRIVE"

# Sanity check on required scripts
for s in run_e2_fast.py run_e4_t2_frozen_fast.py run_e3.py \
         run_e5_tc_decomp.py mnist_fd.py merge_e2_stats.py \
         merge_e2_holm.py viz_schedule.py; do
    if [ ! -f "$s" ]; then
        fail "missing $s — pull from git on this branch"
        exit 2
    fi
done

if ! grep -q "Score FID on best.pt" run_e2_fast.py; then
    fail "run_e2_fast.py is the OLD version (no best.pt patch). Pull from git."
    exit 2
fi

# ============ R1: add seeds 45-47 to reach n=6 ============
log ""
log "=== R1: train extra seeds at T=4 to reach n=6 ==="
for bs in 1 4; do
  for seed in $SEEDS_R1_EXTRA; do
    tag="bs${bs}_s${seed}"
    plog="logs/r1_${tag}.log"
    out="results/per_run_floor6/${tag}.json"
    if [ -f "checkpoints_e2_floor6/${tag}_best.pt" ] && [ -f "$out" ]; then
      log "SKIP R1 $tag"
      continue
    fi
    log "=== R1 START $tag $(date) ==="
    timeout $PER_RUN_TIMEOUT python -u run_e2_fast.py \
        --device cuda --epochs $EPOCHS_E2 \
        --seeds $seed --block_sizes $bs \
        --sigmoid_offset -6.0 --loss_form true_elbo \
        --val_fraction 0.1 --select_by val_loss \
        --val_samples_per_t $VAL_SAMPLES_PER_T \
        --save_dir checkpoints_e2_floor6 \
        --results_json "$out" 2>&1 | tee -a "$plog" | tee -a "$CHAIN_LOG"
    rc=${PIPESTATUS[0]}
    log "=== R1 END $tag rc=$rc $(date) ==="
  done
done

# ============ R1 aggregation (operates on whatever JSONs exist) ============
log ""
log "=== AGGREGATE R1 $(date) ==="
python -u merge_e2_stats.py --sources results/per_run_floor6/*.json \
    --bs_a 1 --bs_b 4 \
    --out results/results_e2_floor6_merged.json 2>&1 | tee -a "$CHAIN_LOG"
python -u merge_e2_holm.py --sources results/per_run_floor6/*.json \
    --out results/results_e2_floor6_holm.json 2>&1 | tee -a "$CHAIN_LOG"
python -u viz_schedule.py --e2_dir checkpoints_e2_floor6 \
    --out_prefix figures/viz_schedule_e2_floor6 \
    --summary_json results/schedule_summary_floor6.json 2>&1 | tee -a "$CHAIN_LOG"

# ============ R3: T=2 frozen schedule (no v1 ckpt dependency) ============
log ""
log "=== R3: T=2 frozen schedule with alphas [$FROZEN_ALPHAS] ==="
for bs in 1 4; do
  for seed in $SEEDS_R3; do
    tag="T2_bs${bs}_s${seed}_frozen"
    plog="logs/r3_${tag}.log"
    out="results/results_e4_t2_frozen_${tag}.json"
    if [ -f "checkpoints_e4_t2_frozen/${tag}_best.pt" ] && [ -f "$out" ]; then
      log "SKIP R3 $tag"
      continue
    fi
    log "=== R3 START $tag $(date) ==="
    timeout $PER_RUN_TIMEOUT python -u run_e4_t2_frozen_fast.py \
        --device cuda --T 2 \
        --block_sizes $bs --seeds $seed --epochs $EPOCHS_E4 \
        --alphas $FROZEN_ALPHAS \
        --loss_form true_elbo --val_fraction 0.1 --select_by val_loss \
        --val_samples_per_t $VAL_SAMPLES_PER_T \
        --save_dir checkpoints_e4_t2_frozen \
        --results_json "$out" 2>&1 | tee -a "$plog" | tee -a "$CHAIN_LOG"
    rc=${PIPESTATUS[0]}
    log "=== R3 END $tag rc=$rc $(date) ==="
  done
done

# ============ R4: E3 (within-block TC) on the ORIGINAL v1 |G|=4 ckpts ============
log ""
log "=== R4 E3 (on v1 ckpts, all 6 bs4 seeds) $(date) ==="
python -u run_e3.py --device cuda --ckpt_dir checkpoints_e2 \
    --results_json results/results_e3_n6.json \
    --fig_prefix figures/e3_n6 \
    2>&1 | tee logs/r4_e3.log | tee -a "$CHAIN_LOG"

# ============ R5: within-vs-between block TC decomposition ============
log ""
log "=== R5 TC decomp (on v1 |G|=4 ckpts) $(date) ==="
python -u run_e5_tc_decomp.py --device cuda \
    --ckpt_dir checkpoints_e2 --block_size 4 \
    --n_images 512 --mc_samples 8 --n_pair_samples 64 \
    --results_json results/results_e5_tc_decomp.json \
    --fig_prefix figures/e5_tc_decomp \
    2>&1 | tee logs/r5_tc.log | tee -a "$CHAIN_LOG"

# ============ R6: MNIST-FD secondary metric ============
log ""
log "=== R6 classifier $(date) ==="
if [ ! -f ~/.cache/fldd_mnist_clf.pt ]; then
    python -u mnist_fd.py --train_classifier --epochs 3 --device cuda \
        2>&1 | tee logs/r6_classifier.log | tee -a "$CHAIN_LOG"
else
    log "  classifier already cached"
fi

log "=== R6 score v1 originals $(date) ==="
python -u mnist_fd.py --score_ckpts checkpoints_e2 --device cuda \
    --n_samples 10000 \
    --results_json results/mnist_fd_e2_original.json \
    2>&1 | tee logs/r6_score_original.log | tee -a "$CHAIN_LOG"

log "=== R6 score v2 R1 ckpts $(date) ==="
python -u mnist_fd.py --score_ckpts checkpoints_e2_floor6 --device cuda \
    --n_samples 10000 \
    --results_json results/mnist_fd_e2_floor6.json \
    2>&1 | tee logs/r6_score_floor6.log | tee -a "$CHAIN_LOG"

log ""
log "=== CHAIN DONE $(date) ==="
log "Keep-alive tmux session is still running. Stop it manually with:"
log "  tmux kill-session -t keepalive"
