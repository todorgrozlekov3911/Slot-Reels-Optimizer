import numpy as np
from reels_csv_matrix import ReelsMatrix
from models import PayTable
from tools.reel_set_simulator.win_mechanics.cluster_win import ClusterWinMechanic
from simulator import SimulationParams
from tools.reel_set_simulator.optimization_algos.base_ga_generator import GeneticOptimizer, GAParams


def main():
    rm = ReelsMatrix("reels_base.csv")

    pt = PayTable({
        0: [0, 0, 0, 0, 10, 20, 30, 40, 50],
        1: [0, 0, 0, 0, 10, 20, 30, 40, 50],
        2: [0, 0, 0, 0, 10, 20, 30, 40, 50],
        3: [0, 0, 0, 0, 20, 40, 60, 80, 100],
        4: [0, 0, 0, 0, 20, 40, 60, 80, 100],
        5: [0, 0, 0, 0, 20, 40, 60, 80, 100],
        6: [0, 0, 0, 0, 30, 60, 90, 120, 150],
        7: [0, 0, 0, 0, 30, 60, 90, 120, 150],
        8: [0, 0, 0, 0, 50, 100, 200, 400, 800],
    })

    mech = ClusterWinMechanic()

    sim_params = SimulationParams(
        matrix=rm.matrix.copy(),
        win_mechanic=mech,
        pay_table=pt,
        board_reels=5,
        board_rows=5,
    )

    target_hrv = np.array([
        20.0,
        0.10,
        0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
    ])

    weights = np.array([
        1.0,
        10.0,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ])

    ga_params = GAParams(
        population_size=50,
        elite_count=2,
        parent_count=10,
        mutations_per_parent=5,
        max_generations=50,
        target_fitness=0.05,
        stagnation_limit=10,
        num_iterations=50_000,
    )

    print("=" * 60)
    print("GENETIC OPTIMIZER")
    print("=" * 60)

    optimizer = GeneticOptimizer(
        sim_params=sim_params,
        target_hrv=target_hrv,
        weights=weights,
        ga_params=ga_params,
    )

    best = optimizer.run(verbose=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best fitness: {best.fitness:.6f}")
    print(f"Best HRV: {best.hit_rate_vector}")

    optimizer.save_best("evolved_reels.csv")


if __name__ == "__main__":
    main()