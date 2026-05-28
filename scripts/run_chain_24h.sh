#!/bin/bash
# ~10-hour single-invocation chain. Single tmux pane, walk away, come back.
#
# Default budget (no flags): ~11 hours.
#   P1 PILOT     ~3h   5 cells × bs4 seed 42 × 50 epochs
#   P2 MAIN n=3  ~7.5h winning config × bs ∈ {1, 4} × seeds 42-44 × 100ep
#   P7 STATS+FIG ~0.5h paired Holm 1v4, schedule plot, viz_samples,
#                      recompute_elbo on v1 ckpts
#   P8 SUMMARY   <1m   results/v3/SUMMARY.json
#
# Opt-in extensions (each adds time to the same single invocation):
#   --with_bs2       adds P3 bs=2 n=3 (~3.5h)  → totals ~14.5h
#   --with_extras    adds P4 extras seeds 45-47 bs∈{1,4} (~7.5h) and
#                    (if --with_bs2 also given) P5 extras × bs=2 (~3.5h)
#   --with_fid_mc    adds P6 FID Monte-Carlo on the v3 ckpts (~1.5h)
#   --max            shortcut for "--with_bs2 --with_extras --with_fid_mc"
#
# All phases are idempotent (SKIP guards on existing ckpts + JSONs). If a
# run crashes the chain logs the failure and continues to the next item —
# never aborts mid-chain.
#
# Usage (Renku, tmux):
#   export REPO=~/work/aml-runs-fixed
#   cd $REPO
#   tmux new -s aml-chain
#   bash scripts/run_chain_24h.sh
#   Ctrl+B D to detach.

set -u   # no `set -e` — one bad run must not stop the rest of the chain

# ============ env + flags ============
REPO=${REPO:-$HOME/work/aml-runs-fixed}
cd "$REPO"

WITH_BS2=0
WITH_EXTRAS=0
WITH_FID_MC=0
for arg in "$@"; do
  case "$arg" in
    --with_bs2)    WITH_BS2=1 ;;
    --no_bs2)      WITH_BS2=0 ;;
    --with_extras) WITH_EXTRAS=1 ;;
    --with_fid_mc) WITH_FID_MC=1 ;;
    --max)         WITH_BS2=1; WITH_EXTRAS=1; WITH_FID_MC=1 ;;
    *) ;;
  esac
done

CHAIN_START=$(date +%s)
CHAIN_LOG="logs/chain24h_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs results/pilot_ablation results/v3 figures

log()  { echo "$@" | tee -a "$CHAIN_LOG"; }
fail() { log "[!] $*"; }

elapsed() {
  local now=$(date +%s); local sec=$((now - CHAIN_START))
  printf '%dh%02dm' $((sec / 3600)) $(((sec % 3600) / 60))
}
phase_log() { log ""; log "=== $1 (elapsed $(elapsed)) ==="; }

log "=== CHAIN START $(date) ==="
log "  REPO=$REPO  WITH_BS2=$WITH_BS2  WITH_EXTRAS=$WITH_EXTRAS  WITH_FID_MC=$WITH_FID_MC"
log "  CHAIN_LOG=$CHAIN_LOG"

# Expected runtime preview
expected_h=11
[ "$WITH_BS2" = "1" ]    && expected_h=$((expected_h + 3))
[ "$WITH_EXTRAS" = "1" ] && expected_h=$((expected_h + 8))
[ "$WITH_EXTRAS" = "1" ] && [ "$WITH_BS2" = "1" ] && expected_h=$((expected_h + 4))
[ "$WITH_FID_MC" = "1" ] && expected_h=$((expected_h + 2))
log "  expected runtime ≈ ${expected_h}h"

# Sanity check on required scripts
need=(run_e2_fast.py train_mnist.py merge_e2_stats.py merge_e2_holm.py
      viz_schedule.py viz_samples.py recompute_elbo.py
      scripts/pick_best_ablation.py)
[ "$WITH_FID_MC" = "1" ] && need+=(scripts/fid_mc.py)

for s in "${need[@]}"; do
  if [ ! -f "$s" ]; then
    fail "missing $s — pull from PERFECT_REPO_PROPOSAL"; exit 2
  fi
done
if ! grep -q "ema_decay" train_mnist.py; then
  fail "train_mnist.py has no ema_decay flag — pull v3 train_mnist.py"; exit 2
fi
if ! grep -q "class EMA" fldd/train.py; then
  fail "fldd/train.py has no EMA class — pull v3 fldd/train.py"; exit 2
fi

# ============================================================================
# P1 — 5-cell pilot ablation
# ============================================================================
phase_log "P1 PILOT ABLATION (5 cells, bs4 seed 42, 50 epochs)"

PILOT_EPOCHS=50
PILOT_SEED=42
PILOT_BS=4
PILOT_DIR=checkpoints_pilot
PER_RUN_TIMEOUT=3600
mkdir -p "$PILOT_DIR"

run_cell () {
  local cell="$1"; local ema_flag="$2"; local sel="$3"; local extra="$4"
  local tag="cell_${cell}"
  local plog="logs/p1_${tag}.log"
  local out="results/pilot_ablation/${tag}.json"
  local prefix="${tag}"

  if [ -f "$out" ] && [ -f "${PILOT_DIR}/${prefix}_best.pt" ]; then
    log "SKIP P1 $tag"; return
  fi
  log "=== P1 START $tag $(date) ==="
  timeout $PER_RUN_TIMEOUT python -u run_e2_fast.py \
      --device cuda --T 4 --epochs $PILOT_EPOCHS \
      --seeds $PILOT_SEED --block_sizes $PILOT_BS \
      --sigmoid_offset -6.0 --loss_form true_elbo \
      --val_fraction 0.1 --select_by "$sel" \
      --val_samples_per_t 1 \
      $ema_flag $extra \
      --save_dir "$PILOT_DIR" --save_prefix "$prefix" \
      --results_json "$out" --cell_name "$cell" \
      2>&1 | tee "$plog" | tee -a "$CHAIN_LOG"
  log "=== P1 END $tag rc=${PIPESTATUS[0]} $(date) ==="
}

run_cell "ema_off_select_train" ""                   "train_loss" ""
run_cell "ema_off_select_val"   ""                   "val_loss"   ""
run_cell "ema_on_select_train"  "--ema_decay 0.9999" "train_loss" ""
run_cell "ema_on_select_val"    "--ema_decay 0.9999" "val_loss"   ""
run_cell "ema_on_v1sched_val"   "--ema_decay 0.9999" "val_loss"   "--fixed_alphas 0.06 0.06 0.06 0.50"

phase_log "P1 PILOT AGGREGATE"
WINNER_LINE=$(python -u scripts/pick_best_ablation.py 2>>"$CHAIN_LOG")
log "$WINNER_LINE"
if echo "$WINNER_LINE" | grep -q "WINNER_CELL"; then
  eval "$WINNER_LINE"
else
  fail "pilot did not produce a winner — falling back to ema_on + val_loss"
  WINNER_CELL="ema_on_select_val"
fi

case "$WINNER_CELL" in
  *ema_on*)     MAIN_EMA="--ema_decay 0.9999" ;;
  *)            MAIN_EMA="" ;;
esac
case "$WINNER_CELL" in
  *select_val*) MAIN_SELECT="val_loss" ;;
  *)            MAIN_SELECT="train_loss" ;;
esac
case "$WINNER_CELL" in
  *v1sched*)    MAIN_EXTRA="--fixed_alphas 0.06 0.06 0.06 0.50" ;;
  *)            MAIN_EXTRA="" ;;
esac
log "  main sweep config: select_by=$MAIN_SELECT ema=$MAIN_EMA extra='$MAIN_EXTRA'"

# ============================================================================
# Shared main-sweep runner
# ============================================================================

MAIN_EPOCHS=100
MAIN_DIR=checkpoints_v3
MAIN_TIMEOUT=7200
mkdir -p "$MAIN_DIR"

run_main () {
  local bs="$1"; local seed="$2"
  local tag="bs${bs}_s${seed}"
  local plog="logs/p2_${tag}.log"
  local out="results/v3/${tag}.json"
  local prefix="${tag}"

  if [ -f "$out" ] && [ -f "${MAIN_DIR}/${prefix}_best.pt" ]; then
    log "SKIP $tag"; return
  fi
  log "=== MAIN START $tag $(date) ==="
  timeout $MAIN_TIMEOUT python -u run_e2_fast.py \
      --device cuda --T 4 --epochs $MAIN_EPOCHS \
      --seeds $seed --block_sizes $bs \
      --sigmoid_offset -6.0 --loss_form true_elbo \
      --val_fraction 0.1 --select_by "$MAIN_SELECT" \
      --val_samples_per_t 1 \
      $MAIN_EMA $MAIN_EXTRA \
      --save_dir "$MAIN_DIR" --save_prefix "$prefix" \
      --results_json "$out" \
      2>&1 | tee "$plog" | tee -a "$CHAIN_LOG"
  log "=== MAIN END $tag rc=${PIPESTATUS[0]} $(date) ==="
}

# ============================================================================
# P2 — main n=3 sweep, bs ∈ {1, 4}  (default ON — the headline)
# ============================================================================
phase_log "P2 MAIN SWEEP bs∈{1,4} seeds 42,43,44"
for seed in 42 43 44; do
  for bs in 1 4; do
    run_main "$bs" "$seed"
  done
done

# ============================================================================
# P3 — bs=2 n=3 (optional, --with_bs2)
# ============================================================================
if [ "$WITH_BS2" = "1" ]; then
  phase_log "P3 BS=2 SWEEP seeds 42,43,44"
  for seed in 42 43 44; do
    run_main 2 "$seed"
  done
fi

# ============================================================================
# P4 — extra seeds 45,46,47 × bs∈{1,4} (optional, --with_extras)
# ============================================================================
if [ "$WITH_EXTRAS" = "1" ]; then
  phase_log "P4 EXTRA SEEDS 45,46,47 bs∈{1,4} (n=6 headline)"
  for seed in 45 46 47; do
    for bs in 1 4; do
      run_main "$bs" "$seed"
    done
  done
fi

# ============================================================================
# P5 — extras × bs=2 (optional, --with_extras AND --with_bs2)
# ============================================================================
if [ "$WITH_BS2" = "1" ] && [ "$WITH_EXTRAS" = "1" ]; then
  phase_log "P5 EXTRA SEEDS 45,46,47 bs=2"
  for seed in 45 46 47; do
    run_main 2 "$seed"
  done
fi

# ============================================================================
# P6 — FID Monte-Carlo on the v3 ckpts (optional, --with_fid_mc)
# ============================================================================
SAMPLE_KIND="best"
if [ "$MAIN_SELECT" = "val_loss" ]; then SAMPLE_KIND="valbest"; fi

if [ "$WITH_FID_MC" = "1" ]; then
  phase_log "P6 FID MONTE-CARLO (3 sample sets × ckpt)"
  if [ ! -f results/v3/fid_mc_v3.json ]; then
    timeout 7200 python -u scripts/fid_mc.py \
        --ckpt_dir "$MAIN_DIR" --ckpt_kind "$SAMPLE_KIND" \
        --n_samples 10000 --fid_seeds 0 1 2 \
        --results_json results/v3/fid_mc_v3.json \
        2>&1 | tee logs/p6_fid_mc.log | tee -a "$CHAIN_LOG"
  else
    log "SKIP P6 (results/v3/fid_mc_v3.json exists)"
  fi
fi

# ============================================================================
# P7 — aggregate stats + figures + retroactive ELBO
# ============================================================================
phase_log "P7 STATS + FIGURES"

# Paired stats on bs1 vs bs4
python -u merge_e2_stats.py --sources results/v3/bs*_s*.json \
    --bs_a 1 --bs_b 4 \
    --out results/v3/results_e2_v3_merged_1v4.json \
    2>&1 | tee -a "$CHAIN_LOG"

# Holm (1v2, 1v4, 2v4) — only meaningful if bs2 is in
if [ "$WITH_BS2" = "1" ]; then
  python -u merge_e2_holm.py --sources results/v3/bs*_s*.json \
      --out results/v3/results_e2_v3_holm.json \
      2>&1 | tee -a "$CHAIN_LOG"
fi

# Schedule plots from v3 ckpts
python -u viz_schedule.py \
    --e2_dir "$MAIN_DIR" \
    --out_prefix figures/viz_schedule_v3 \
    --summary_json results/v3/schedule_summary_v3.json \
    2>&1 | tee -a "$CHAIN_LOG"

# Qualitative sample figure (uses EMA weights if ckpt has them)
BS_LIST="1 4"
[ "$WITH_BS2" = "1" ] && BS_LIST="1 2 4"
python -u viz_samples.py --dataset mnist \
    --ckpt_dir "$MAIN_DIR" --seed 42 \
    --block_sizes $BS_LIST --ckpt_kind "$SAMPLE_KIND" \
    --out figures/v3_samples_comparison.png \
    2>&1 | tee -a "$CHAIN_LOG"

# Retroactive ELBO correction on v1 ckpts (analytical, no GPU)
if [ -f results/schedule_summary.json ]; then
  python -u recompute_elbo.py \
      --schedule_summary results/schedule_summary.json \
      --sources results/results_e4.json results/results_e2_merged.json \
                results/results_e2_bs2.json \
      --source_T_fixed none 4 4 \
      --out results/v3/results_elbo_corrected_v1.json \
      2>&1 | tee -a "$CHAIN_LOG"
fi

# ============================================================================
# P8 — write a single SUMMARY.json with the headline numbers
# ============================================================================
phase_log "P8 SUMMARY"

python -u - <<'PY' 2>&1 | tee -a "$CHAIN_LOG"
import glob, json, os
out = {"v3": {}}

rows = []
for p in sorted(glob.glob("results/v3/bs*_s*.json")):
    with open(p) as f: d = json.load(f)
    for r in d.get("per_run", []):
        rows.append(r)
out["v3"]["per_run"] = rows

agg = {}
for bs in (1, 2, 4):
    fids = [r["fid"] for r in rows if r["block_size"] == bs and "fid" in r]
    if fids:
        n = len(fids)
        mean = sum(fids) / n
        var = sum((f - mean) ** 2 for f in fids) / (n - 1) if n > 1 else 0.0
        agg[str(bs)] = {"n": n, "fid_mean": mean, "fid_std": var ** 0.5}
out["v3"]["aggregate_fid_seed_sd"] = agg

for path, key in [
    ("results/v3/results_e2_v3_merged_1v4.json", "paired_1v4"),
    ("results/v3/results_e2_v3_holm.json",       "holm"),
    ("results/v3/fid_mc_v3.json",                "fid_mc"),
    ("results/v3/results_elbo_corrected_v1.json", "v1_elbo_corrected"),
]:
    if os.path.exists(path):
        with open(path) as f: out["v3"][key] = json.load(f)

with open("results/v3/SUMMARY.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results/v3/SUMMARY.json")
print("\nv3 aggregate FID (across seeds):")
for bs, a in sorted(agg.items()):
    print(f"  |G|={bs}  n={a['n']}  FID={a['fid_mean']:.2f} ± {a['fid_std']:.2f}")
PY

phase_log "CHAIN DONE"
log "Total elapsed: $(elapsed)"
log "Next: read results/v3/SUMMARY.json, eyeball figures/v3_samples_comparison.png,"
log "      figures/viz_schedule_v3.png. Update README headline table from SUMMARY.json."
