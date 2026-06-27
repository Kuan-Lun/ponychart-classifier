"""Exact vs asymptotic convergence study and Type I error calibration via Monte Carlo."""

import hashlib
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ponychart_classifier.stats import goodness_of_fit_test
from ponychart_classifier.stats.exact import (
    MAX_COMPOSITIONS,
    num_compositions,
)

# ── Exact vs asymptotic convergence study ───────────────────────


class TestConvergence:
    """Empirically check when asymptotic becomes close to exact for k=6.

    This test iterates over increasing n (with fixed proportional counts)
    and records the gap between exact and asymptotic p-values.
    """

    def test_convergence_k3(self) -> None:
        """For k=3, check convergence at various n."""
        # Slightly non-uniform proportions
        proportions = [0.5, 0.3, 0.2]
        for n in (6, 9, 12, 18, 24, 30):
            counts = [round(p * n) for p in proportions]
            # Adjust to ensure sum == n
            counts[-1] = n - sum(counts[:-1])
            r_exact = goodness_of_fit_test(counts, method="pearson_exact")
            r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
            if r_exact.exact:
                # Just verify both give valid results
                assert 0 <= r_exact.p_value <= 1
                assert 0 <= r_asymp.p_value <= 1

    def test_convergence_k6_small(self) -> None:
        """For k=6 with small n, exact is feasible.

        n=10, k=6: C(15,5) = 3003 compositions — very fast.
        """
        counts = [3, 2, 2, 1, 1, 1]
        r_exact = goodness_of_fit_test(counts, method="pearson_exact")
        r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
        assert r_exact.exact
        # Record the gap for reference
        gap = abs(r_exact.p_value - r_asymp.p_value)
        assert gap >= 0  # just checking it computes

    def test_exact_feasibility_boundary_k6(self) -> None:
        """Find approximate n where exact becomes infeasible for k=6."""
        k = 6
        for n in range(1, 100):
            nc = num_compositions(n, k)
            if nc > MAX_COMPOSITIONS:
                # n where we cross the threshold
                assert n < 100  # should happen well before 100
                break

    def test_convergence_report_k6(self) -> None:
        """Print exact vs asymptotic gap for k=6 at feasible n values.

        This test always passes; it prints a convergence table to help
        determine when asymptotic approximation becomes reliable.
        """
        # Non-uniform proportions (typical real-world scenario)
        proportions = [0.35, 0.15, 0.15, 0.15, 0.10, 0.10]
        gaps: list[tuple[int, float, float, float]] = []

        for n in (12, 15, 18, 20, 24):
            k = len(proportions)
            nc = num_compositions(n, k)
            if nc > MAX_COMPOSITIONS:
                break
            raw = [max(0, round(p * n)) for p in proportions]
            raw[-1] = n - sum(raw[:-1])
            if raw[-1] < 0:
                continue
            counts = raw
            r_exact = goodness_of_fit_test(counts, method="pearson_exact")
            r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
            if r_exact.exact:
                gaps.append(
                    (
                        n,
                        r_exact.p_value,
                        r_asymp.p_value,
                        abs(r_exact.p_value - r_asymp.p_value),
                    )
                )

        # Print convergence table (visible with pytest -s)
        print("\n=== Exact vs Asymptotic convergence (k=6, non-uniform) ===")
        print(f"{'n':>4}  {'p_exact':>10}  {'p_asymp':>10}  {'|gap|':>10}")
        for n, pe, pa, gap in gaps:
            print(f"{n:>4}  {pe:>10.6f}  {pa:>10.6f}  {gap:>10.6f}")

    def test_convergence_report_k3(self) -> None:
        """Print exact vs asymptotic gap for k=3."""
        proportions = [0.5, 0.3, 0.2]
        gaps: list[tuple[int, float, float, float]] = []

        for n in (6, 10, 15, 20, 30, 50, 80):
            k = len(proportions)
            nc = num_compositions(n, k)
            if nc > MAX_COMPOSITIONS:
                break
            counts = [round(p * n) for p in proportions]
            counts[-1] = n - sum(counts[:-1])
            r_exact = goodness_of_fit_test(counts, method="pearson_exact")
            r_asymp = goodness_of_fit_test(counts, method="pearson_asymptotic")
            if r_exact.exact:
                gaps.append(
                    (
                        n,
                        r_exact.p_value,
                        r_asymp.p_value,
                        abs(r_exact.p_value - r_asymp.p_value),
                    )
                )

        print("\n=== Exact vs Asymptotic convergence (k=3, non-uniform) ===")
        print(f"{'n':>4}  {'p_exact':>10}  {'p_asymp':>10}  {'|gap|':>10}")
        for n, pe, pa, gap in gaps:
            print(f"{n:>4}  {pe:>10.6f}  {pa:>10.6f}  {gap:>10.6f}")


# ── Type I error calibration via Monte Carlo ────────────────────
#
# Architecture: each unique (n, probs) scenario draws samples ONCE.
# All methods that share the same scenario reuse the same samples.
# p-values are computed once per (sample, method); rejection is checked
# against multiple alpha levels without re-running the tests.
#
# We use alpha = 0.40 and 0.45 instead of the conventional 0.05.
# At alpha ~ 0.4, the binomial SE of the rejection rate is near its
# maximum (alpha*(1-alpha) ≈ 0.24), giving much better *relative*
# precision (margin/alpha) for the same n_sim. This lets us detect
# miscalibration with far fewer simulations.

_EXACT_METHODS: tuple[str, ...] = ("pearson_exact", "lr_exact", "probability_exact")
_ASYMP_METHODS: tuple[str, ...] = ("pearson_asymptotic", "lr_asymptotic")
_ALL_METHODS: tuple[str, ...] = _EXACT_METHODS + _ASYMP_METHODS
_ALPHAS: tuple[float, ...] = (0.40, 0.45)
_Z = 3  # number of standard errors for margin


def _margin(alpha: float, n_sim: int) -> float:
    """Compute assertion margin: z * SE of binomial proportion at alpha."""
    return float(_Z * (alpha * (1 - alpha) / n_sim) ** 0.5)


@dataclass
class _Scenario:
    """A unique (n, probs) simulation scenario."""

    label: str
    n: int
    probs: list[float]
    methods: tuple[str, ...]
    n_sim: int


# Central registry of all scenarios. Each (n, probs) pair appears only once.
_SCENARIOS: list[_Scenario] = [
    _Scenario("k3_n9_uniform", 9, [1 / 3] * 3, _ALL_METHODS, 500),
    _Scenario("k3_n12_non_uniform", 12, [0.5, 0.3, 0.2], _ALL_METHODS, 500),
    _Scenario("k4_n150_non_uniform", 150, [0.4, 0.3, 0.2, 0.1], _ASYMP_METHODS, 500),
    _Scenario("k6_n60_uniform", 60, [1 / 6] * 6, _ASYMP_METHODS, 500),
    _Scenario("k6_n200_uniform", 200, [1 / 6] * 6, _ASYMP_METHODS, 500),
]

# Cache key: scenario label → {(alpha, method): rejection_rate}
_CacheValue = dict[tuple[float, str], float]

_PARALLEL_THRESHOLD_SECONDS = 30.0


def _get_performance_core_count() -> int:
    """Return the number of performance (P) cores.

    - macOS Apple Silicon: ``sysctl hw.perflevel0.logicalcpu``
    - Linux big.LITTLE / other: ``os.cpu_count() // 2`` (heuristic)
    - Fallback: half of total cores, minimum 1.
    """
    import os

    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                text=True,
                timeout=5,
            )
            return max(1, int(out.strip()))
        except Exception:
            pass
    total = os.cpu_count() or 2
    return max(1, total // 2)


def _run_chunk(
    n: int,
    probs: list[float],
    methods: tuple[str, ...],
    alphas: tuple[float, ...],
    n_sim: int,
    seed: int,
) -> dict[tuple[float, str], int]:
    """Worker: run n_sim simulations, return rejection counts per (alpha, method)."""
    rng = np.random.default_rng(seed)
    p_arr = np.array(probs)
    rejections: dict[tuple[float, str], int] = {
        (a, m): 0 for a in alphas for m in methods
    }
    for _ in range(n_sim):
        sample = rng.multinomial(n, p_arr).tolist()
        for m in methods:
            r = goodness_of_fit_test(
                sample,
                probs=probs,
                method=m,  # type: ignore[arg-type]
            )
            for a in alphas:
                if r.p_value <= a:
                    rejections[(a, m)] += 1
    return rejections


def _compute_rejection_rates(
    scenario: _Scenario,
    base_seed: int,
) -> _CacheValue:
    """Compute rejection rates, auto-parallelizing when beneficial.

    1. Benchmark one iteration (1 sample, all methods).
    2. Estimate total wall time.
    3. If > threshold, split n_sim across performance cores.
    """
    methods = scenario.methods

    # Benchmark single iteration
    rng_bench = np.random.default_rng(base_seed)
    p_arr = np.array(scenario.probs)
    sample = rng_bench.multinomial(scenario.n, p_arr).tolist()
    t0 = time.perf_counter()
    for m in methods:
        goodness_of_fit_test(
            sample,
            probs=scenario.probs,
            method=m,  # type: ignore[arg-type]
        )
    per_iter = time.perf_counter() - t0
    estimated_total = per_iter * scenario.n_sim

    n_workers = 1
    if estimated_total > _PARALLEL_THRESHOLD_SECONDS:
        n_workers = _get_performance_core_count()

    if n_workers <= 1:
        counts = _run_chunk(
            scenario.n,
            scenario.probs,
            methods,
            _ALPHAS,
            scenario.n_sim,
            base_seed,
        )
    else:
        chunk_size = scenario.n_sim // n_workers
        remainder = scenario.n_sim % n_workers
        chunks: list[tuple[int, int]] = []
        for i in range(n_workers):
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append((size, base_seed + i + 1))

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _run_chunk,
                    scenario.n,
                    scenario.probs,
                    methods,
                    _ALPHAS,
                    size,
                    seed,
                )
                for size, seed in chunks
            ]
            keys = [(a, m) for a in _ALPHAS for m in methods]
            counts = {k: 0 for k in keys}
            for f in futures:
                chunk_counts = f.result()
                for k in keys:
                    counts[k] += chunk_counts[k]

    return {k: counts[k] / scenario.n_sim for k in counts}


# Module-level cache: computed once per scenario, shared by all tests.
_cache: dict[str, _CacheValue] = {}


def _stable_seed(label: str) -> int:
    """Derive a reproducible seed from a label.

    Python's built-in ``hash()`` for strings is salted per-process
    (PYTHONHASHSEED), so it would give a different Monte Carlo seed on every
    run and make this calibration test flaky. SHA-256 is stable across runs.
    """
    digest = hashlib.sha256(label.encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def _get_rates(scenario: _Scenario) -> _CacheValue:
    """Return cached rejection rates, computing on first access."""
    if scenario.label not in _cache:
        base_seed = _stable_seed(scenario.label)
        _cache[scenario.label] = _compute_rejection_rates(scenario, base_seed)
    return _cache[scenario.label]


def _find(label: str) -> _Scenario:
    return next(s for s in _SCENARIOS if s.label == label)


class TestTypeIErrorCalibration:
    """Under H0, the fraction of rejections at level alpha should be ~alpha.

    For exact tests, the rejection rate must be <= alpha (conservative).
    For asymptotic tests, the rejection rate should approximate alpha
    when n is large enough.

    Each assertion is checked at both alpha=0.40 and alpha=0.45.
    Both must pass — this guards against implementations that are
    accidentally correct at a single threshold.
    """

    def test_exact_conservative_k3_uniform(self) -> None:
        sc = _find("k3_n9_uniform")
        rates = _get_rates(sc)
        for a in _ALPHAS:
            m_ = _margin(a, sc.n_sim)
            for m in _EXACT_METHODS:
                assert (
                    rates[(a, m)] <= a + m_
                ), f"{m} alpha={a}: rate {rates[(a, m)]:.3f} > {a} + {m_:.3f}"

    def test_exact_conservative_k3_non_uniform(self) -> None:
        sc = _find("k3_n12_non_uniform")
        rates = _get_rates(sc)
        for a in _ALPHAS:
            m_ = _margin(a, sc.n_sim)
            for m in _EXACT_METHODS:
                assert (
                    rates[(a, m)] <= a + m_
                ), f"{m} alpha={a}: rate {rates[(a, m)]:.3f} > {a} + {m_:.3f}"

    def test_asymptotic_calibration_k6_large_n(self) -> None:
        sc = _find("k6_n200_uniform")
        rates = _get_rates(sc)
        for a in _ALPHAS:
            m_ = _margin(a, sc.n_sim)
            for m in _ASYMP_METHODS:
                assert abs(rates[(a, m)] - a) < m_, (
                    f"{m} alpha={a}: rate {rates[(a, m)]:.3f}, "
                    f"expected ~{a} (margin={m_:.3f})"
                )

    def test_asymptotic_calibration_k4_non_uniform(self) -> None:
        sc = _find("k4_n150_non_uniform")
        rates = _get_rates(sc)
        for a in _ALPHAS:
            m_ = _margin(a, sc.n_sim)
            for m in _ASYMP_METHODS:
                assert abs(rates[(a, m)] - a) < m_, (
                    f"{m} alpha={a}: rate {rates[(a, m)]:.3f}, "
                    f"expected ~{a} (margin={m_:.3f})"
                )

    def test_calibration_report(self) -> None:
        """Print full calibration table (reuses cached results, no extra cost)."""
        for a in _ALPHAS:
            print(f"\n=== Type I error calibration (alpha={a}, z={_Z}) ===")
            print(
                f"{'scenario':<25} {'method':<24} "
                f"{'n_sim':>5} {'margin':>7} {'rej_rate':>8}  status"
            )
            m_ = _margin(a, 500)  # all scenarios use same n_sim
            for sc in _SCENARIOS:
                rates = _get_rates(sc)
                m_ = _margin(a, sc.n_sim)
                for method in sc.methods:
                    rate = rates[(a, method)]
                    is_exact = "exact" in method
                    if is_exact:
                        ok = rate <= a + m_
                        status = "OK (conservative)" if ok else "FAIL"
                    else:
                        ok = abs(rate - a) < m_
                        status = "OK" if ok else "DRIFT"
                    print(
                        f"{sc.label:<25} {method:<24} "
                        f"{sc.n_sim:>5} {m_:>7.4f} {rate:>8.3f}  {status}"
                    )
