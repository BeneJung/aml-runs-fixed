"""Parse the 4 pilot-ablation result JSONs and print the winning config.

Reads four files matching:
    results/pilot_ablation/cell_<name>.json

where <name> in {ema_off_select_train, ema_off_select_val,
                 ema_on_select_train, ema_on_select_val}.

Picks the cell with the **lowest FID** on bs4 seed 42 (the pilot config).
Prints a one-line summary to stdout suitable for `WINNER=$(python ...)`
parsing, plus a human table to stderr.
"""
import argparse
import glob
import json
import os
import sys


def load_pilot(path):
    with open(path) as f:
        d = json.load(f)
    # Expect a single run in per_run; if multiple, take min FID
    runs = d.get("per_run", [])
    if not runs:
        return None
    r = min(runs, key=lambda r: r.get("fid", float("inf")))
    cell = d.get("config", {}).get("cell_name") or \
           d.get("cell_name") or \
           os.path.splitext(os.path.basename(path))[0].replace("cell_", "")
    return {
        "cell": cell,
        "fid": r.get("fid"),
        "best_loss": r.get("best_loss"),
        "best_val": r.get("best_val"),
        "best_epoch": r.get("best_epoch"),
        "selected_alphas": r.get("selected_alphas") or r.get("final_alphas"),
        "ema_decay": d.get("config", {}).get("ema_decay"),
        "select_by": d.get("config", {}).get("select_by"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", default="results/pilot_ablation")
    ap.add_argument("--pattern", default="cell_*.json")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pilot_dir, args.pattern)))
    if not paths:
        print(f"no pilot results in {args.pilot_dir}/{args.pattern}",
              file=sys.stderr)
        sys.exit(2)

    rows = [load_pilot(p) for p in paths]
    rows = [r for r in rows if r and r["fid"] is not None]
    if not rows:
        print("no valid runs found", file=sys.stderr)
        sys.exit(2)

    rows.sort(key=lambda r: r["fid"])

    print(f"\n{'cell':<32} {'ema_decay':>10} {'select_by':>12} "
          f"{'best_ep':>8} {'val/train_loss':>15} {'FID':>10}",
          file=sys.stderr)
    for r in rows:
        loss = r["best_val"] if r["select_by"] == "val_loss" else r["best_loss"]
        print(f"{r['cell']:<32} {str(r['ema_decay']):>10} "
              f"{str(r['select_by']):>12} "
              f"{str(r['best_epoch']):>8} "
              f"{(loss if loss is not None else float('nan')):>15.4f} "
              f"{r['fid']:>10.4f}",
              file=sys.stderr)

    winner = rows[0]
    print(f"\nWINNER: {winner['cell']}  FID={winner['fid']:.4f}",
          file=sys.stderr)

    # Machine-readable line on stdout
    print(f"WINNER_CELL={winner['cell']} WINNER_FID={winner['fid']:.4f} "
          f"WINNER_EMA={winner['ema_decay']} "
          f"WINNER_SELECT={winner['select_by']}")


if __name__ == "__main__":
    main()
