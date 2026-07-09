from dataclasses import dataclass
from reels_csv_matrix import ReelsMatrix
from win_mechanics import WinMechanic
from models import Board, HitRateVector, PayTable
from classification_head import ClassificationHead
import numpy as np

@dataclass
class SimulationResult:
    state: ClassificationHead
    hit_rate_vector: HitRateVector

class ReelSetSimulator:
    def __init__(self, csv_obj: ReelsMatrix, 
                 win_mechanic:WinMechanic, pay_table: PayTable,
                 board_reels: int, board_rows: int,
                 num_iterations: int = 1000):
        
        self.win_mechanic        = win_mechanic 
        self.pay_table           = pay_table
        self.board               = Board(board_reels, board_rows, csv_obj)
        self.classification_head = ClassificationHead.from_reels_matrix(self.board.csv_representation)
        self.num_iterations      = num_iterations
        self.hit_rate_vector     = HitRateVector(np.zeros(self.board.csv_representation.get_numb_symbols() + 2, dtype=np.float64))

    def run(self) -> SimulationResult:

        while self.hit_rate_vector.games < self.num_iterations:
            self.board.generate_board()
            self.hit_rate_vector += self.win_mechanic.play(self.board, self.pay_table)

        self.hit_rate_vector.finalize()
        return SimulationResult(self.classification_head, self.hit_rate_vector)
    


