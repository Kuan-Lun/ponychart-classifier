#!/usr/bin/env python3
"""Analyse the card-stamping model for PonyChart character distributions.

Model
-----
6 characters, stamps drawn uniformly with replacement.
- Always stamp once; after each stamp, continue with probability p.
- Discard cards with N >= 4 stamps; keep N <= 3.
- Count distinct characters K = 1, 2, 3.

Theoretical ratio (given):
    A : B : C  =  (36 + 6p + p^2) : (30p + 15p^2) : (20p^2)
    S = 36 + 36p + 36p^2
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence

# ---------------------------------------------------------------------------
# Core formulae
# ---------------------------------------------------------------------------


def raw_counts(p: float) -> tuple[float, float, float]:
    """Unnormalised A, B, C for a given p."""
    p2 = p * p
    a = 36.0 + 6.0 * p + p2
    b = 30.0 * p + 15.0 * p2
    c = 20.0 * p2
    return a, b, c


def theoretical_probs(p: float) -> tuple[float, float, float]:
    """Normalised P1, P2, P3."""
    a, b, c = raw_counts(p)
    s = a + b + c
    if s == 0:
        return (1.0, 0.0, 0.0)
    return a / s, b / s, c / s


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def chi2_loss(obs: Sequence[float], exp: Sequence[float]) -> float:
    """Pearson chi-square statistic."""
    return sum((o - e) ** 2 / e for o, e in zip(obs, exp) if e > 0)


def nll_loss(obs: Sequence[float], exp: Sequence[float]) -> float:
    """Multinomial negative log-likelihood (cross-entropy direction)."""
    n = sum(obs)
    if n == 0:
        return 0.0
    s = sum(exp)
    return -sum(o * math.log(max(e / s, 1e-300)) for o, e in zip(obs, exp) if o > 0)


def sse_loss(obs: Sequence[float], exp: Sequence[float]) -> float:
    """Sum of squared errors on proportions."""
    n_obs = sum(obs)
    n_exp = sum(exp)
    if n_obs == 0 or n_exp == 0:
        return 0.0
    return sum((o / n_obs - e / n_exp) ** 2 for o, e in zip(obs, exp))


LOSS_FUNCS: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    "chi2": chi2_loss,
    "nll": nll_loss,
    "sse": sse_loss,
}

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def expected_counts(p: float, n: float) -> tuple[float, float, float]:
    p1, p2, p3 = theoretical_probs(p)
    return n * p1, n * p2, n * p3


def fit_p(
    obs: tuple[int, int, int],
    loss_name: str = "chi2",
    grid_size: int = 20000,
) -> float:
    """Find best p in [0, 0.999999] minimising the chosen loss."""
    n = sum(obs)
    loss_fn = LOSS_FUNCS[loss_name]

    def objective(p: float) -> float:
        e = expected_counts(p, n)
        return loss_fn(obs, e)

    # Try scipy first
    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(objective, bounds=(1e-9, 0.999999), method="bounded")
        return float(res.x)
    except ImportError:
        pass

    # Fallback: grid search + golden-section refinement
    best_p = 0.0
    best_val = float("inf")
    for i in range(grid_size + 1):
        p = max(i / grid_size * 0.999999, 1e-9)
        v = objective(p)
        if v < best_val:
            best_val = v
            best_p = p

    # Local refinement (golden section on a narrow bracket)
    lo = max(best_p - 1.0 / grid_size, 1e-9)
    hi = min(best_p + 1.0 / grid_size, 0.999999)
    gr = (math.sqrt(5) + 1) / 2
    for _ in range(80):
        if hi - lo < 1e-12:
            break
        c = hi - (hi - lo) / gr
        d = lo + (hi - lo) / gr
        if objective(c) < objective(d):
            hi = d
        else:
            lo = c
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def fmt(x: float, decimals: int = 6) -> str:
    return f"{x:.{decimals}f}"


def ratio_str(a: float, b: float, c: float, decimals: int = 4) -> str:
    mn = min(x for x in (a, b, c) if x > 0) if any(x > 0 for x in (a, b, c)) else 1.0
    return f"{a/mn:.{decimals}f} : {b/mn:.{decimals}f} : {c/mn:.{decimals}f}"


def print_separator(title: str = "") -> None:
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_fit(args: argparse.Namespace) -> None:
    c1, c2, c3 = args.counts
    n = c1 + c2 + c3
    if n == 0:
        msg = "Error: total count is 0."
        print(msg, file=sys.stderr)
        raise ValueError(msg)

    obs = (c1, c2, c3)
    obs_prop = (c1 / n, c2 / n, c3 / n)

    best_p = fit_p(obs, loss_name=args.loss, grid_size=args.grid)
    p1, p2, p3 = theoretical_probs(best_p)
    e1, e2, e3 = expected_counts(best_p, n)
    # ---- Fitted model output ----
    print_separator("FIT RESULT")
    print(f"  Observations      : K=1 {c1},  K=2 {c2},  K=3 {c3}   (n={n})")
    print(f"  Best p            : {fmt(best_p, 8)}")
    print()
    print(f"  Theoretical probs : P1={fmt(p1)}  P2={fmt(p2)}  P3={fmt(p3)}")
    print(
        f"  Observed  props   : P1={fmt(obs_prop[0])}  P2={fmt(obs_prop[1])}  P3={fmt(obs_prop[2])}"
    )
    print()
    print(f"  Expected counts   : {fmt(e1, 2)}  {fmt(e2, 2)}  {fmt(e3, 2)}")
    print(
        f"  Residuals (O - E) : {fmt(c1 - e1, 2)}  {fmt(c2 - e2, 2)}  {fmt(c3 - e3, 2)}"
    )
    print()


def cmd_predict(args: argparse.Namespace) -> None:
    p = args.p
    if not 0 <= p < 1:
        msg = "Error: p must be in [0, 1)."
        print(msg, file=sys.stderr)
        raise ValueError(msg)

    p1, p2, p3 = theoretical_probs(p)

    print_separator(f"PREDICT  p = {p}")

    if args.n is not None:
        n = args.n
        print()
        print(f"  Predicted counts (n={n}):")
        print(f"    K=1 : {fmt(n * p1, 2)}")
        print(f"    K=2 : {fmt(n * p2, 2)}")
        print(f"    K=3 : {fmt(n * p3, 2)}")
    print()


def cmd_scan(args: argparse.Namespace) -> None:
    start = args.start
    end = args.end
    step = args.step

    rows: list[tuple[float, float, float, float]] = []
    p = start
    header = f"{'p':>10s}  {'P1':>10s}  {'P2':>10s}  {'P3':>10s}  {'ratio A:B:C':>30s}"
    print_separator("SCAN")
    print(header)
    print("-" * len(header))
    while p <= end + 1e-12:
        p_val = min(p, 0.999999)
        p1, p2, p3 = theoretical_probs(p_val)
        a, b, c = raw_counts(p_val)
        rows.append((p_val, p1, p2, p3))
        print(
            f"{p_val:10.4f}  {p1:10.6f}  {p2:10.6f}  {p3:10.6f}  {ratio_str(a, b, c):>30s}"
        )
        p += step
    print()

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            ps = [r[0] for r in rows]
            plt.figure(figsize=(9, 5))
            plt.plot(ps, [r[1] for r in rows], label="P1 (K=1)")
            plt.plot(ps, [r[2] for r in rows], label="P2 (K=2)")
            plt.plot(ps, [r[3] for r in rows], label="P3 (K=3)")
            plt.xlabel("p (continuation probability)")
            plt.ylabel("Probability")
            plt.title("Theoretical probabilities vs p")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  (matplotlib not available; skipping plot)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyse the card-stamping model for character distributions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- fit --
    p_fit = sub.add_parser("fit", help="Fit p from observed K=1,2,3 counts")
    p_fit.add_argument(
        "counts",
        nargs=3,
        type=int,
        metavar=("C1", "C2", "C3"),
        help="Observed counts for K=1, K=2, K=3",
    )
    p_fit.add_argument(
        "--loss",
        choices=["chi2", "nll", "sse"],
        default="chi2",
        help="Loss function (default: chi2)",
    )
    p_fit.add_argument(
        "--grid",
        type=int,
        default=20000,
        help="Grid size for fallback search (default: 20000)",
    )
    p_fit.add_argument("--verbose", action="store_true")

    # -- predict --
    p_pred = sub.add_parser("predict", help="Show theoretical probs for a given p")
    p_pred.add_argument("p", type=float, help="Continuation probability")
    p_pred.add_argument(
        "--n", type=int, default=None, help="Total sample size for predicted counts"
    )
    p_pred.add_argument("--verbose", action="store_true")

    # -- scan --
    p_scan = sub.add_parser("scan", help="Scan p values and show probs")
    p_scan.add_argument("--start", type=float, default=0.0)
    p_scan.add_argument("--end", type=float, default=0.999)
    p_scan.add_argument("--step", type=float, default=0.01)
    p_scan.add_argument(
        "--plot", action="store_true", help="Plot if matplotlib available"
    )
    p_scan.add_argument("--verbose", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "scan":
        cmd_scan(args)


if __name__ == "__main__":
    main()
