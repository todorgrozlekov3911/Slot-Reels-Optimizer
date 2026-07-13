from dataclasses import dataclass
from numpy.typing import NDArray
from win_mechanics import WinMechanic
from models import Board, HitRateVector, PayTable
import numpy as np


@dataclass
class SimulationParams:
    matrix: NDArray
    win_mechanic: WinMechanic
    pay_table: PayTable
    board_reels: int
    board_rows: int


class ReelSetSimulator:
    def __init__(self, num_iterations: int = 100_000):
        self.num_iterations = num_iterations

    def run(self, params: SimulationParams) -> HitRateVector:
        matrix = params.matrix
        min_val, max_val = int(matrix.min()), int(matrix.max())
        hrv_size = max_val - min_val + 1 + 2

        board = Board.from_matrix(params.board_reels, params.board_rows, matrix)
        hrv = HitRateVector(np.zeros(hrv_size, dtype=np.float64))

        for _ in range(self.num_iterations):
            board.generate_board()
            hrv += params.win_mechanic.play(board, params.pay_table)

        hrv.finalize()
        return hrv
    


