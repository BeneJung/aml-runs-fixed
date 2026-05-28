#!/bin/bash
# Phase 2 — run AFTER scripts/run_chain.sh has completed.
# Adds the analyses the main chain didn't cover:
#   1. Fair v1 rescore from best.pt (the apples-to-apples FID comparison)
#   2. MNIST-FD on the R3 frozen-schedule ckpts
#   3. Sample-image grids for the paper figure
#
# Total time: ~2h on an A10. All steps idempotent via SKIP guards.

set -u

REPO=${REPO:-$HOME/work/aml-runs-fixed}
DRIVE=${DRIVE:-$HOME/work/drive}
cd "$REPO"

CHAIN_LOG="logs/phase2_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs results/v1_rescored figures/samples

log() { echo "$@" | tee -a "$CHAIN_LOG"; }

log "=== PHASE 2 START $(date) ==="

# ============ 1. v1 rescore from best.pt ============
log ""
log "=== v1 rescore from best.pt (~90 min) ==="
if ls results/v1_rescored/*.json 1>/dev/null 2>&1 && \
   [ "$(ls results/v1_rescored/*.json | wc -l)" -ge 18 ]; then
    log "SKIP v1 rescore (18+ JSONs already exist; pass --force to redo)"
else
    python -u scripts/rescore_best_pt.py \
        --ckpt_dir checkpoints_e2 \
        --results_dir results/v1_rescored \
        --force 2>&1 | tee -a "$CHAIN_LOG"
fi

# Aggregate the rescored v1 numbers with Holm for the apples-to-apples table
log ""
log "=== aggregate v1 rescored ==="
python -u merge_e2_stats.py \
    --sources results/v1_rescored/bs1_s*.json results/v1_rescored/bs4_s*.json \
    --bs_a 1 --bs_b 4 \
    --out results/v1_rescored_merged_1v4.json 2>&1 | tee -a "$CHAIN_LOG"
python -u merge_e2_holm.py \
    --sources results/v1_rescored/*.json \
    --out results/v1_rescored_holm.json 2>&1 | tee -a "$CHAIN_LOG"

# ============ 2. MNIST-FD on R3 frozen-schedule ckpts ============
log ""
log "=== MNIST-FD on R3 frozen ckpts ==="
if [ -d checkpoints_e4_t2_frozen ] && \
   [ "$(ls checkpoints_e4_t2_frozen/*_best.pt 2>/dev/null | wc -l)" -gt 0 ]; then
    python -u mnist_fd.py --score_ckpts checkpoints_e4_t2_frozen \
        --device cuda --n_samples 10000 \
        --results_json results/mnist_fd_e4_t2_frozen.json \
        2>&1 | tee -a "$CHAIN_LOG"
else
    log "SKIP — no R3 ckpts found yet (did the main chain finish R3?)"
fi

# ============ 3. Sample image grids for the paper figure ============
log ""
log "=== Sample grids for paper figure ==="
python -u <<'PY' 2>&1 | tee -a "$CHAIN_LOG"
import os, torch
from fldd.unet import UNet
from fldd.forward import LearnedForwardProcess
from fldd.sample import sample, save_samples

os.makedirs("figures/samples", exist_ok=True)
device = "cuda"
torch.manual_seed(0)   # reproducible sample grid

# Pairs: (path, output PNG)
pairs = [
    # v2 ckpts (R1)
    ("checkpoints_e2_floor6/bs1_s42_best.pt", "figures/samples/v2_bs1_s42.png"),
    ("checkpoints_e2_floor6/bs4_s42_best.pt", "figures/samples/v2_bs4_s42.png"),
    # v1 ckpts (originals)
    ("checkpoints_e2/bs1_s42_best.pt", "figures/samples/v1_bs1_s42.png"),
    ("checkpoints_e2/bs4_s42_best.pt", "figures/samples/v1_bs4_s42.png"),
]
for ckpt_path, out_png in pairs:
    if not os.path.exists(ckpt_path):
        print(f"  skip {ckpt_path} — not found")
        continue
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    bs = ckpt["block_size"]
    sigmoid_offset = ckpt.get("sigmoid_offset",
                              LearnedForwardProcess.HISTORICAL_OFFSET)
    m = UNet(channels=(32, 64, 128), block_size=bs).to(device).eval()
    m.load_state_dict(ckpt["model"])
    fp = LearnedForwardProcess(T=ckpt["T"],
                                sigmoid_offset=sigmoid_offset).to(device).eval()
    fp.load_state_dict(ckpt["forward"])
    s = sample(m, fp, ckpt["T"], n_samples=64, device=device, block_size=bs)
    save_samples(s, out_png)
    print(f"  wrote {out_png}")
PY

log ""
log "=== PHASE 2 DONE $(date) ==="
log "Final paper-relevant artifacts:"
log "  - results/v1_rescored_merged_1v4.json (fair v1 stats)"
log "  - results/v1_rescored_holm.json (Holm-corrected family)"
log "  - results/mnist_fd_e4_t2_frozen.json (R3 secondary metric)"
log "  - figures/samples/v[12]_bs[14]_s42.png (qualitative comparison)"
log ""
log "When you no longer need keepalive:  tmux kill-session -t keepalive"
