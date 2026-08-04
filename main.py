"""Entry point: runs the full study and writes results + figures.

    python main.py --quick     # ~6 min, smoke-test settings, for iterating
    python main.py --full      # ~1-2 h on CPU, the numbers quoted in the README
    python main.py --stage e1  # a single experiment

Everything is deterministic given --seed. Results land in results/ as JSON so
figures can be regenerated without retraining (--plots-only).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from cancer_unc import plots
from cancer_unc.data import PhantomConfig, make_benchmark, make_loaders, oracle_stats
from cancer_unc.experiments import (
    exp_aleatoric_recovery,
    exp_calibration,
    exp_epistemic_vs_data,
    exp_identifiability,
    exp_mc_budget,
    exp_ood,
    exp_selective,
    predict_split,
)
from cancer_unc.train import TrainConfig, train_ensemble

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------
def presets(quick: bool, seed: int) -> tuple[PhantomConfig, TrainConfig, dict]:
    """Two budgets. `quick` exists so the pipeline can be exercised end to end
    without a multi-hour wait; it is not expected to reproduce the numbers."""
    phantom = PhantomConfig()
    if quick:
        cfg = TrainConfig(epochs=4, ensemble_size=2, width=16, seed=seed)
        sizes = {"n_train": 1200, "n_val": 400, "n_test": 800, "n_ood": 400}
        sweep = {
            "betas": (1.0, 2.5),
            "e1_n_train": 800,
            "e2_sizes": (250, 1000),
            "e1_cfg": replace(cfg, epochs=3, ensemble_size=2),
            "e2_cfg": replace(cfg, epochs=3, ensemble_size=2),
        }
    else:
        # Sized for a 4-core laptop CPU (~1.5-2 h end to end). The binding
        # constraint is that E1 and E2 train a fresh ensemble *per sweep point*,
        # so they dominate the budget -- hence the lighter config for those.
        cfg = TrainConfig(epochs=14, ensemble_size=4, width=24, seed=seed)
        sizes = {"n_train": 4000, "n_val": 1200, "n_test": 2500, "n_ood": 1200}
        sweep = {
            "betas": (0.6, 1.0, 1.6, 2.5, 4.0),
            "e1_n_train": 2500,
            "e2_sizes": (250, 500, 1000, 2000, 4000),
            # The sweeps run fewer members and epochs than the main model. That
            # shifts the absolute values a little -- an under-trained model
            # carries some of its underfitting in the aleatoric term -- but E1
            # and E2 claim *trends* against a fixed target, and those hold.
            "e1_cfg": replace(cfg, epochs=9, ensemble_size=2, width=20),
            "e2_cfg": replace(cfg, epochs=9, ensemble_size=2, width=20),
        }
    return phantom, cfg, {**sizes, **sweep}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def save(name: str, obj) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    print(f"  -> results/{name}.json")


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke-test budget")
    ap.add_argument("--full", action="store_true", help="full budget (default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    choices=["all", "e1", "e2", "e3", "e4", "e5", "e6", "audit"])
    ap.add_argument("--plots-only", action="store_true",
                    help="regenerate figures from saved results/*.json")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = leave default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    quick = args.quick or not args.full
    phantom, cfg, P = presets(quick, args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 74)
    print("Bayesian Uncertainty Calibration & OOD Detection for Oncological Scans")
    print("=" * 74)
    print(f"budget      : {'quick' if quick else 'full'}")
    print(f"device      : {cfg.device}   threads={torch.get_num_threads()}")
    print(f"phantom     : beta={phantom.beta}  size={phantom.image_size}")
    o = oracle_stats(phantom)
    print(f"oracle      : bayes_error={o['bayes_error']:.4f}  "
          f"aleatoric={o['aleatoric_entropy']:.4f} nats")
    print()

    t_start = time.time()
    stage = args.stage

    # ---- identifiability audit (cheap, always worth having) --------------
    if stage in ("all", "audit") and not args.plots_only:
        print("[audit] bracketing the oracle's validity ...")
        bracket = exp_identifiability(phantom, n=1500)
        print(f"  aleatoric in [{bracket['aleatoric_lower_bound']:.4f}, "
              f"{bracket['aleatoric_upper_bound']:.4f}] nats "
              f"(width {bracket['bracket_width']:.4f})")
        save("audit_identifiability", bracket)
        print()

    # ---- main model ------------------------------------------------------
    need_model = stage in ("all", "e3", "e4", "e5", "e6")
    benchmark = models = None
    if need_model and not args.plots_only:
        print("[main] building benchmark + training ensemble ...")
        benchmark = make_benchmark(
            phantom, n_train=P["n_train"], n_val=P["n_val"],
            n_test=P["n_test"], n_ood=P["n_ood"], seed=args.seed,
        )
        loaders = make_loaders(benchmark, batch_size=cfg.batch_size)
        models, logs = train_ensemble(cfg, loaders, out_dir=ROOT / "checkpoints")
        save("train_logs", {"config": asdict(cfg), "logs": logs})

        out_test = predict_split(models, benchmark["test"])
        acc = float((out_test.prediction == benchmark["test"].labels).mean())
        print(f"  test accuracy {acc:.4f}  (Bayes ceiling {1 - o['bayes_error']:.4f})")
        print()
        plots.plot_phantoms(benchmark, FIGURES)
        plots.plot_uncertainty_vs_oracle(out_test, benchmark["test"], FIGURES)

    # ---- experiments -----------------------------------------------------
    if not args.plots_only:
        if stage in ("all", "e1"):
            print("[E1] aleatoric recovery across the label-noise sweep")
            save("e1_aleatoric_recovery", exp_aleatoric_recovery(
                betas=P["betas"], base_phantom=phantom, cfg=P["e1_cfg"],
                n_train=P["e1_n_train"]))
            print()

        if stage in ("all", "e2"):
            print("[E2] epistemic decay with training set size")
            save("e2_epistemic_vs_data", exp_epistemic_vs_data(
                sizes=P["e2_sizes"], phantom=phantom, cfg=P["e2_cfg"]))
            print()

        if stage in ("all", "e3"):
            print("[E3] calibration, in-distribution and under shift")
            save("e3_calibration", exp_calibration(models, benchmark))
            print()

        if stage in ("all", "e4"):
            print("[E4] OOD detection")
            save("e4_ood", exp_ood(models, benchmark))
            print()

        if stage in ("all", "e5"):
            print("[E5] selective prediction")
            save("e5_selective", exp_selective(models, benchmark))
            print()

        if stage in ("all", "e6"):
            print("[E6] MC sample budget")
            save("e6_mc_budget", exp_mc_budget(models, benchmark))
            print()

    # ---- figures ---------------------------------------------------------
    print("[figures]")
    bracket = load("audit_identifiability")
    for name, fn in [
        ("e1_aleatoric_recovery", lambda r: plots.plot_aleatoric_recovery(r, bracket, FIGURES)),
        ("e2_epistemic_vs_data", lambda r: plots.plot_epistemic_vs_data(r, FIGURES)),
        ("e3_calibration", lambda r: (plots.plot_reliability(r, FIGURES),
                                      plots.plot_calibration_under_shift(r, FIGURES))),
        ("e4_ood", lambda r: plots.plot_ood(r, FIGURES)),
        ("e5_selective", lambda r: plots.plot_risk_coverage(r, FIGURES)),
        ("e6_mc_budget", lambda r: plots.plot_mc_budget(r, FIGURES)),
    ]:
        res = load(name)
        if res is None:
            continue
        try:
            fn(res)
            print(f"  ok {name}")
        except Exception as e:  # a failed figure must not lose the results
            print(f"  !! {name}: {type(e).__name__}: {e}")

    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")
    print(f"results  -> {RESULTS}")
    print(f"figures  -> {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
