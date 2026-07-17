"""
GA Test Run
-----------
Uses the standard pay table (L1-H4, symbols 0-8, cents).
Minimum cluster size = 5 (indices 0-3 are zero).

Target: hit rate ~0.12, avg win ~1.20$ per winning spin,
        premium symbols (H3=7, H4=8) contributing less frequently,
        low symbols (L1-L3 = 0,1,2) contributing more frequently.

Steps:
1. Run GA to find optimized reel strip
2. Save evolved CSV
3. Load fresh, run 1M Monte Carlo to verify
"""

import numpy as np
import time

from reels_csv_matrix import ReelsMatrix
from models import PayTable
from tools.reel_set_simulator.win_mechanics.cluster_win import ClusterWinMechanic
from simulator import ReelSetSimulator, SimulationParams
from tools.reel_set_simulator.optimization_algos.base_ga_generator import GeneticOptimizer, GAParams


# ── Standard pay table (cents) ───────────────────────────────────────────────
PAY_TABLE = PayTable({
    0: [0, 0, 0, 0,  20,  40,  60,  60,  150,  150,  500,  500,  1000,  1000,  2500],  # L1
    1: [0, 0, 0, 0,  20,  40,  60,  60,  150,  150,  500,  500,  1000,  1000,  2500],  # L2
    2: [0, 0, 0, 0,  20,  40,  60,  60,  150,  150,  500,  500,  1000,  1000,  2500],  # L3
    3: [0, 0, 0, 0,  30,  50,  80,  80,  200,  200,  600,  600,  1200,  1200,  3000],  # L4
    4: [0, 0, 0, 0,  30,  50,  80,  80,  200,  200,  600,  600,  1200,  1200,  3000],  # L5
    5: [0, 0, 0, 0,  50, 100, 200, 200,  400,  400, 1200, 1200,  2500,  2500,  5000],  # H1
    6: [0, 0, 0, 0, 100, 150, 250, 250,  500,  500, 1500, 1500,  3000,  3000,  6000],  # H2
    7: [0, 0, 0, 0, 150, 200, 300, 300,  600,  600, 2000, 2000,  4000,  4000,  8000],  # H3
    8: [0, 0, 0, 0, 200, 250, 400, 400,  800,  800, 2500, 2500,  5000,  5000, 10000],  # H4
})

# ── Target HitRateVector ──────────────────────────────────────────────────────
# [avg_win($), hit_rate, sym0_rate ... sym8_rate]
#
# Feasibility constraints (sym rates only count PAYING clusters):
#   * sym_rate[i] <= hit_rate for every i (a symbol win implies a hit)
#   * sum(sym_rates) <= ~1.5 x hit_rate realistically — a 5x5 board fits at
#     most 5 disjoint clusters of size >= 5, and multi-cluster spins are rare
#
# Profile: hit_rate 0.12 (2x the base strip's ~0.06), commons (L1-L3) carry
# most wins, premiums (H1-H4) taper off, avg_win moderate since wins are
# mostly small L clusters.
TARGET = np.array([
    1.20,    # avg_win in dollars (per winning spin)
    0.12,    # hit_rate
    0.030,   # sym 0 (L1)
    0.030,   # sym 1 (L2)
    0.030,   # sym 2 (L3)
    0.020,   # sym 3 (L4)
    0.020,   # sym 4 (L5)
    0.012,   # sym 5 (H1)
    0.008,   # sym 6 (H2)
    0.005,   # sym 7 (H3)
    0.002,   # sym 8 (H4) — rare premium
])
# sum(sym_rates) = 0.157 ≈ 1.3 x hit_rate, every component <= hit_rate

# ── Weights: scale-normalized + manual priority boost ─────────────────────────
WEIGHTS          = 1.0 / (TARGET ** 2 + 1e-8)
WEIGHTS[0]      *= 2.0   # avg_win   — double priority
WEIGHTS[1]      *= 4.0   # hit_rate  — highest priority

# ── GA params ────────────────────────────────────────────────────────────────
GA_PARAMS = GAParams(
    population_size      = 50,
    elite_count          = 2,
    parent_count         = 10,
    mutations_per_parent = 5,
    max_generations      = 100,
    target_fitness       = 0.05,
    stagnation_limit     = 15,
    num_iterations       = 10_000,
)

EVOLVED_CSV_PATH  = "evolved_reels.csv"
MONTE_CARLO_SPINS = 1_000_000


def run_ga(rm: ReelsMatrix) -> None:
    print("=" * 60)
    print("PHASE 1 — GENETIC OPTIMIZATION")
    print("=" * 60)
    print(f"Base strip : {rm.matrix.shape}, symbols {rm.min_val}-{rm.max_val}")
    print(f"Population : {GA_PARAMS.population_size}  "
          f"Generations: {GA_PARAMS.max_generations}  "
          f"Spins/candidate: {GA_PARAMS.num_iterations:,}")
    print(f"Target     : hit_rate={TARGET[1]:.2f}  avg_win={TARGET[0]:.2f}$")
    print()

    sim_params = SimulationParams(
        matrix       = rm.matrix.copy(),
        win_mechanic = ClusterWinMechanic(),
        pay_table    = PAY_TABLE,
        board_reels  = 5,
        board_rows   = 5,
    )

    optimizer = GeneticOptimizer(
        sim_params = sim_params,
        target_hrv = TARGET,
        weights    = WEIGHTS,
        save_path  = EVOLVED_CSV_PATH,
        ga_params  = GA_PARAMS,
    )

    t0      = time.time()
    best    = optimizer.run(describe=True)
    elapsed = time.time() - t0

    print(f"\nGA finished in {elapsed:.1f}s")
    print(f"Best fitness : {best.fitness:.6f}")
    print(f"Saved to     : {EVOLVED_CSV_PATH}")


def run_monte_carlo() -> None:
    print()
    print("=" * 60)
    print("PHASE 2 — MONTE CARLO VERIFICATION (1M spins)")
    print("=" * 60)

    evolved_rm = ReelsMatrix(EVOLVED_CSV_PATH)

    print("Symbol counts per reel after evolution:")
    header = "       " + "".join(f"  reel{i}" for i in range(evolved_rm.num_reels))
    print(header)
    for s in range(evolved_rm.min_val, evolved_rm.max_val + 1):
        row = f"  sym {s}:"
        for reel in range(evolved_rm.num_reels):
            count = int(np.sum(evolved_rm.matrix[:, reel] == s))
            row += f"  {count:5d}"
        print(row)
    print()

    sim_params = SimulationParams(
        matrix       = evolved_rm.matrix.copy(),
        win_mechanic = ClusterWinMechanic(),
        pay_table    = PAY_TABLE,
        board_reels  = 5,
        board_rows   = 5,
    )

    simulator = ReelSetSimulator(num_iterations=MONTE_CARLO_SPINS)

    t0      = time.time()
    hrv     = simulator.run(sim_params)
    elapsed = time.time() - t0

    print(f"Monte Carlo: {elapsed:.1f}s  ({MONTE_CARLO_SPINS/elapsed:,.0f} spins/sec)\n")

    labels = (["avg_win($)", "hit_rate"] +
              [f"sym_{i}_rate" for i in range(evolved_rm.min_val, evolved_rm.max_val + 1)])

    print(f"{'Metric':<18} {'Target':>10} {'Measured':>10} {'Delta':>10} {'Match':>8}")
    print("-" * 60)
    for i, label in enumerate(labels):
        if i >= len(TARGET):
            break
        t     = TARGET[i]
        m     = float(hrv.data[i])
        delta = m - t
        pct   = abs(delta / t) * 100 if t != 0 else 0
        match = "✓" if pct < 10 else "✗"
        print(f"{label:<18} {t:>10.4f} {m:>10.4f} {delta:>+10.4f} {match:>6} ({pct:.1f}%)")

    print()
    print(f"Final HitRateVector:\n{hrv}")


def main():
    rm = ReelsMatrix("reels_base.csv")
    run_ga(rm)
    run_monte_carlo()


if __name__ == "__main__":
    main()