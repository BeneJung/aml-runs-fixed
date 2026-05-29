#!/bin/bash
# Settle the v3 FID reversal: is it a real ELBO–FID gap (a) or an EMA–block
# interaction artifact (b)? Builds the {EMA on, EMA off} × {|G|=1, |G|=4}
# 2×2 that v3 never completed, and reads off whether EMA flips the sign of
# Δ FID(1−4).
#
#   D1  fid_no_ema.py   live-vs-EMA re-score of the 6 v3 ckpts   ~0.5h  (decisive, cheap)
#   D2  pilot inspect   which cell v3's main sweep inherited     ~0     (CPU)
#   D4  EMA-off sweep   bs∈{1,4} × seeds 42–44 × 100ep, NO EMA   ~7.5h  (clean control)
#   D3  sample grids    viz_samples (patched) EMA vs live        ~0.5h
#   D5  ELBO–FID table  decoupling from D4 logs                  ~0     (CPU)
#   D6  extra seeds     EMA-off, +~2.5h PER seed (bs1+bs4)       optional
#
# BUDGET (single GPU, ~75 min / 100-epoch run — your v3 chain's own estimate):
#   D1 + D4(n=3) + D3 + D5  ≈ 8.3h   ← fits 12h with ~3.7h to spare.
#   Each EXTRA_SEEDS seed adds ~2.5h. So:
#       n=4 (EXTRA_SEEDS="45")        ≈ 10.8h   ← recommended stretch, fits 12h
#       n=6 (EXTRA_SEEDS="45 46 47")  ≈ 15.5h   ← does NOT fit; run as a 2nd session
#
# All GPU steps idempotent (SKIP guards). No `set -e`: a bad run logs and the
# chain continues. Outputs land under results/ (symlinked to SwitchDrive).
#
# Usage (Renku, tmux):
#   export REPO=~/work/aml-runs-fixed && cd "$REPO"
#   git pull                          # get fid_no_ema.py + this + the sample.py patch
#   tmux new -s settle
#   bash scripts/run_settle_science.sh                 # D1+D2+D4(n=3)+D3+D5   (~8.3h)
#   EXTRA_SEEDS="45" bash scripts/run_settle_science.sh  # also seed 45 → n=4   (~10.8h)
#   Ctrl+B D to detach.
#
# Idempotent: safe to re-run later with more EXTRA_SEEDS; only new runs execute.

set -u

REPO=${REPO:-$HOME/work/aml-runs-fixed}
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"   # so `python scripts/*.py` can import fldd

EXTRA_SEEDS=${EXTRA_SEEDS:-}     # space-separated, e.g. EXTRA_SEEDS="45 46"

START=$(date +%s)
LOG="logs/settle_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs results/v3 results/emaoff figures checkpoints_emaoff
log()  { echo "$@" | tee -a "$LOG"; }
fail() { log "[!] $*"; }
elapsed() { local s=$(( $(date +%s) - START )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }
phase()  { log ""; log "=== $1 (elapsed $(elapsed)) ==="; }

V3_DIR=checkpoints_v3        # v3 main-sweep ckpts (EMA on, val-sel)
EMAOFF_DIR=checkpoints_emaoff
N_FID=10000

log "=== SETTLE START $(date) ===   REPO=$REPO  EXTRA_SEEDS='$EXTRA_SEEDS'  LOG=$LOG"

# ---------------------------------------------------------------- Step 0: verify state
phase "STEP 0  verify Renku state"
ok=1
for f in "$V3_DIR/bs1_s42_valbest.pt" "$V3_DIR/bs4_s42_valbest.pt"; do
  [ -f "$f" ] || { fail "missing $f — v3 ckpts not on this machine"; ok=0; }
done
grep -q "def sample" fldd/sample.py && grep -q "z_init" fldd/sample.py \
  || { fail "fldd/sample.py is unpatched (no z_init) — D3 grids will crash; git pull"; }
[ -f scripts/fid_no_ema.py ] || { fail "scripts/fid_no_ema.py missing — git pull"; ok=0; }
if [ "$ok" = "0" ]; then
  fail "Step 0 failed. Fix the above (usually: git pull on Renku) and rerun."
  exit 2
fi
log "  ok: v3 ckpts present, fid_no_ema.py present, sample.py patched."

# ---------------------------------------------------------------- D2: pilot inspection
phase "D2  pilot-cell inspection (which config v3's main sweep inherited)"
if [ -d results/pilot_ablation ]; then
  python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json
for f in sorted(glob.glob("results/pilot_ablation/cell_*.json")):
    d = json.load(open(f)); c = d.get("config", {}); r = d["per_run"][0]
    print(f"  {c.get('cell_name','?'):>22}  FID={r['fid']:7.2f}  "
          f"ema={c.get('ema_decay')}  select={c.get('select_by')}  "
          f"best_val={r.get('best_val')}")
print("  -> D4 must match the WINNING cell in everything EXCEPT --ema_decay.")
PY
else
  fail "results/pilot_ablation/ not found — inspect manually before trusting D4 config."
fi

# ---------------------------------------------------------------- D1: live-vs-EMA re-score
phase "D1  fid_no_ema.py — live vs EMA FID on the v3 ckpts (DECISIVE, cheap)"
if [ -f results/v3/fid_no_ema.json ]; then
  log "  SKIP D1 (results/v3/fid_no_ema.json exists)"
else
  python -u scripts/fid_no_ema.py \
      --ckpt_dir "$V3_DIR" --ckpt_glob '*_valbest.pt' \
      --n_fid_samples "$N_FID" \
      --out results/v3/fid_no_ema.json \
      2>&1 | tee -a "$LOG"
fi
log "  --> peek at the VERDICT line above before the 7.5h D4 sweep commits."

# ---------------------------------------------------------------- D4/D6: EMA-off sweep
run_emaoff () {           # $1=bs  $2=seed
  local bs="$1" seed="$2" tag="bs${1}_s${2}"
  local out="results/emaoff/${tag}.json"
  if [ -f "$out" ] && [ -f "${EMAOFF_DIR}/${tag}_valbest.pt" ]; then
    log "  SKIP $tag (done)"; return
  fi
  log "  === EMAOFF START $tag $(date) ==="
  timeout 7200 python -u run_e2_fast.py \
      --device cuda --T 4 --epochs 100 \
      --seeds "$seed" --block_sizes "$bs" \
      --sigmoid_offset -6.0 --loss_form true_elbo \
      --val_fraction 0.1 --select_by val_loss --val_samples_per_t 1 \
      --save_dir "$EMAOFF_DIR" --save_prefix "$tag" \
      --results_json "$out" --cell_name "emaoff_${tag}" \
      2>&1 | tee -a "$LOG"
  log "  === EMAOFF END $tag rc=${PIPESTATUS[0]} $(date) ==="
}

phase "D4  EMA-OFF main sweep bs∈{1,4} seeds 42,43,44 (NO --ema_decay)"
for seed in 42 43 44; do for bs in 1 4; do run_emaoff "$bs" "$seed"; done; done

if [ -n "$EXTRA_SEEDS" ]; then
  phase "D6  EMA-OFF extra seeds: $EXTRA_SEEDS  (~2.5h each; full n=6 = 2nd session)"
  for seed in $EXTRA_SEEDS; do for bs in 1 4; do run_emaoff "$bs" "$seed"; done; done
fi

# ---------------------------------------------------------------- D3: sample grids
phase "D3  qualitative sample grids (viz_samples, now unblocked by the sample.py patch)"
python -u viz_samples.py --dataset mnist \
    --ckpt_dir "$V3_DIR" --seed 42 --block_sizes 1 4 --ckpt_kind valbest \
    --out figures/settle_v3_emaon_samples.png 2>&1 | tee -a "$LOG" || \
    fail "viz_samples failed — check fldd/sample.py patch / viz_samples API (non-fatal)"

# ---------------------------------------------------------------- D4 paired stats
phase "D4-stats  paired Δ FID(1−4) on the EMA-off sweep"
if ls results/emaoff/bs*_s*.json >/dev/null 2>&1; then
  python -u merge_e2_stats.py --sources results/emaoff/bs*_s*.json \
      --bs_a 1 --bs_b 4 --out results/emaoff/emaoff_merged_1v4.json \
      2>&1 | tee -a "$LOG" || fail "merge_e2_stats failed (non-fatal)"
fi

# ---------------------------------------------------------------- D5 + final 2×2
phase "D5 + SUMMARY  ELBO–FID decoupling + the {EMA on/off}×{bs} 2×2"
python - <<'PY' 2>&1 | tee -a "$LOG"
import glob, json, os
from collections import defaultdict

def load_runs(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        rows += d.get("per_run", [])
    return rows

def paired_delta(rows, key="fid", a=1, b=4):
    by = defaultdict(dict)
    for r in rows:
        if key in r and r[key] is not None:
            by[r["seed"]][r["block_size"]] = r[key]
    ds = [(s, d[a]-d[b]) for s, d in sorted(by.items()) if a in d and b in d]
    if not ds: return None
    v = [x for _, x in ds]; n = len(v); m = sum(v)/n
    sd = (sum((x-m)**2 for x in v)/(n-1))**0.5 if n > 1 else 0.0
    out = {"n": n, "mean": m, "sd": sd, "per_seed": {str(s): x for s, x in ds}}
    if n > 1 and sd > 0: out["t"] = m/(sd/n**0.5)
    return out

settled = {}

# EMA-on (v3): prefer the per-run files, fall back to SUMMARY.json
v3_rows = load_runs(sorted(glob.glob("results/v3/bs*_s*.json")))
if not v3_rows and os.path.exists("results/v3/SUMMARY.json"):
    v3_rows = json.load(open("results/v3/SUMMARY.json")).get("v3", {}).get("per_run", [])
settled["emaon_v3_delta_1v4"] = paired_delta(v3_rows, "fid")

# EMA-off retrain
off_rows = load_runs(sorted(glob.glob("results/emaoff/bs*_s*.json")))
settled["emaoff_delta_1v4"] = paired_delta(off_rows, "fid")

# Live re-score of the v3 ckpts (from D1)
if os.path.exists("results/v3/fid_no_ema.json"):
    fne = json.load(open("results/v3/fid_no_ema.json"))
    settled["v3_live_rescore_delta_1v4"] = fne.get("paired_delta_1v4_live")
    settled["v3_ema_rescore_delta_1v4"]  = fne.get("paired_delta_1v4_ema")

# ELBO–FID decoupling on the EMA-off ckpts (best_val is the corrected val-ELBO)
def agg(rows, key):
    o = {}
    for bs in (1, 4):
        vals = [r[key] for r in rows if r["block_size"] == bs and r.get(key) is not None]
        if vals:
            n = len(vals); m = sum(vals)/n
            o[str(bs)] = {"n": n, "mean": m}
    return o
settled["emaoff_fid_by_bs"]     = agg(off_rows, "fid")
settled["emaoff_valelbo_by_bs"] = agg(off_rows, "best_val")

json.dump(settled, open("results/v3/SETTLED.json", "w"), indent=2)

def fmt(d):
    if not d: return "n/a"
    s = f"Δ={d['mean']:+.2f}±{d['sd']:.2f} (n={d['n']}"
    if "t" in d: s += f", t={d['t']:+.2f}"
    return s + ")"

print("\n================ Δ FID(1−4)  (positive = block |G|=4 WINS) ================")
print(f"  v3 EMA-on   (the reversal) : {fmt(settled.get('emaon_v3_delta_1v4'))}")
print(f"  v3 ckpts, EMA re-score     : {fmt(settled.get('v3_ema_rescore_delta_1v4'))}")
print(f"  v3 ckpts, LIVE re-score    : {fmt(settled.get('v3_live_rescore_delta_1v4'))}   <-- D1")
print(f"  EMA-OFF retrain            : {fmt(settled.get('emaoff_delta_1v4'))}   <-- D4")
print("---------------------------------------------------------------------------")
fb = settled.get("emaoff_fid_by_bs", {}); vb = settled.get("emaoff_valelbo_by_bs", {})
def g(d, bs, k):
    return f"{d.get(str(bs), {}).get(k, float('nan')):.2f}" if d.get(str(bs)) else "n/a"
print(f"  EMA-OFF |G|=1 : FID {g(fb,1,'mean')}   val-ELBO {g(vb,1,'mean')}")
print(f"  EMA-OFF |G|=4 : FID {g(fb,4,'mean')}   val-ELBO {g(vb,4,'mean')}")
print("  Read: if EMA-OFF has |G|=4 lower in BOTH FID and val-ELBO -> (b) EMA artifact,")
print("        restore H2. If |G|=4 lower val-ELBO but higher FID -> (a) real ELBO–FID gap.")
print("===========================================================================")
print("\nwrote results/v3/SETTLED.json")
PY

phase "SETTLE DONE"
log "Total elapsed: $(elapsed)"
log "Verdict inputs: results/v3/SETTLED.json + the VERDICT line from D1."
log "Go/no-go to restore H2: D1-live AND D4-emaoff BOTH show Δ FID(1−4) > 0."
