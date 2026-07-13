from dataclasses import dataclass
from typing import Optional
import random as rand
import numpy as np

from models import Chromosome
from simulator import ReelSetSimulator, SimulationParams


@dataclass
class GAParams:
    population_size: int = 50
    elite_count: int = 2
    parent_count: int = 10
    mutations_per_parent: int = 5
    max_generations: int = 100
    target_fitness: float = 0.01
    stagnation_limit: int = 15
    num_iterations: int = 100_000


class GeneticOptimizer:

    def run(self, describe: bool = True) -> Chromosome:
        stagnation_counter = 0

        for gen in range(self.params.max_generations):
            self.run_generation()

            current_best = self.population[0].fitness

            if describe and gen % 5 == 0:
                print(f"Gen {gen:3d}: best_fitness = {current_best:.6f}")

            if current_best <= self.params.target_fitness:
                if describe:
                    print(f"Target fitness reached at generation {gen}")
                break

            if gen > 0:
                if current_best >= self.best_fitness_history[-2] * 0.9999:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0

            if stagnation_counter >= self.params.stagnation_limit:
                if describe:
                    print(f"Stagnation detected at generation {gen}")
                break

        if describe:
            print(f"\nOptimization complete after {self.generation} generations")
            print(f"Best fitness: {self.population[0].fitness:.6f}")


        if self.save_path is not None:
            self.save_best(self.set_save_path)

        return self.get_best()

        

    
    def mutate(self, chromosome: Chromosome):
        matrix = chromosome.reel_matrix

        reel_pick = rand.randrange(0, self.num_reels)
        val_i, val_j = rand.sample(range(self.min_val, self.max_val + 1), 2)

        reel_data = matrix[:, reel_pick]
        count_j = np.sum(reel_data == val_j)

        if count_j == 0:
            return

        count_swap = rand.randint(1, max(1, count_j // 2 + 1))
        count_swap = min(count_swap, count_j)

        if count_swap == 0:
            return

        positions = np.where(reel_data == val_j)[0]
        chosen = np.random.choice(positions, size=count_swap, replace=False)
        matrix[chosen, reel_pick] = val_i

        chromosome.hit_rate_vector = None
        chromosome.fitness = float('inf')

    def evaluate(self, chromosome: Chromosome):
        if chromosome.hit_rate_vector is not None:
            return #znachi e elit

        sim_params = SimulationParams(
            matrix       =chromosome.reel_matrix,
            win_mechanic =self.win_mechanic,
            pay_table    =self.pay_table,
            board_reels  =self.board_reels,
            board_rows   =self.board_rows,
        )

        hrv = self.simulator.run(sim_params)

        chromosome.hit_rate_vector = hrv
        chromosome.fitness = self._weighted_distance(hrv.data, self.target_hrv)

    def _weighted_distance(self, hrv: np.ndarray, target: np.ndarray) -> float:
        min_len = min(len(hrv), len(target), len(self.weights))
        errors = (hrv[:min_len] - target[:min_len]) ** 2
        weighted_errors = self.weights[:min_len] * errors
        return float(np.sqrt(np.sum(weighted_errors)))

    def run_generation(self):
        for chrom in self.population:
            self.evaluate(chrom)

        self.population.sort(key=lambda x: x.fitness)
        self.best_fitness_history.append(self.population[0].fitness)

        new_population: list[Chromosome] = []

        for i in range(self.params.elite_count):
            new_population.append(self.population[i].copy())

        parents = self.population[:self.params.parent_count]

        children_needed     = self.params.population_size - self.params.elite_count
        children_per_parent = children_needed // self.params.parent_count
        extra_children      = children_needed % self.params.parent_count

        for i, parent in enumerate(parents):
            num_children = children_per_parent + (1 if i < extra_children else 0)

            for _ in range(num_children):
                child = parent.copy()
                self.mutate(child)
                new_population.append(child)

        self.population = new_population
        self.generation += 1


    def get_best(self) -> Chromosome:
        return min(self.population, key=lambda x: x.fitness)

    def save_best(self, filepath: str):
        best = self.get_best()
        np.savetxt(filepath, best.reel_matrix, delimiter=",", fmt="%d")

    def __init__(
        self,
        sim_params: SimulationParams,
        target_hrv: np.ndarray,
        weights:    np.ndarray,
        save_path: Optional[str] = None,
        ga_params:  Optional[GAParams] = None,

    ):
        self.target_hrv = target_hrv
        self.params     = ga_params or GAParams()
        self.weights    = weights if weights is not None else np.ones(len(target_hrv))

        self.win_mechanic = sim_params.win_mechanic
        self.pay_table    = sim_params.pay_table
        self.board_reels  = sim_params.board_reels
        self.board_rows   = sim_params.board_rows

        base_matrix    = sim_params.matrix
        self.num_rows  = base_matrix.shape[0]
        self.num_reels = base_matrix.shape[1]
        self.min_val   = int(base_matrix.min())
        self.max_val   = int(base_matrix.max())

        self.simulator = ReelSetSimulator(self.params.num_iterations)

        self.population: list[Chromosome] = []
        self.initialize_population(base_matrix)

        self.generation = 0
        self.best_fitness_history: list[float]= []

        self.save_path = save_path

    def initialize_population(self, base_matrix: np.ndarray):
        self.population.append(Chromosome(
            reel_matrix=base_matrix.copy(),
            fitness=float('inf'),
            hit_rate_vector=None
        ))

        for _ in range(self.params.population_size - 1):
            chrom = Chromosome(
                reel_matrix=base_matrix.copy(),
                fitness=float('inf'),
                hit_rate_vector=None
            )
            self.mutate(chrom)
            self.population.append(chrom)

    def set_save_path(self, new_save_path:str):
        self.save_path = new_save_path
