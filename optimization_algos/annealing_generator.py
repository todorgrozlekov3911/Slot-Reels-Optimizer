
from reel_set_simulator.optimization_algos.base_ga_generator import GeneticOptimizer

class AnnealingGenerator(GeneticOptimizer):

    def mutate(self, chromosome):

        aggresion = max(1, round(3 - 2 * self.progress()))
        for _ in range(aggresion):
            super().mutate(chromosome)

    def progress(self) -> float:
        gen = getattr(self, "generation", 0)
        return min(1.0, gen / max(1, self.params.max_generations))