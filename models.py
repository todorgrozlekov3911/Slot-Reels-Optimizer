
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import random as random
from reels_csv_matrix import ReelsMatrix
from numba import jit

class Board:
    def __init__(self, num_reels, num_rows, reels_matrix: ReelsMatrix):
        self.board = np.full((num_reels, num_rows), -1) # Board is represented the same as inside the Engine -> board[i] == i-th reel
                                                        # -> board[i][j] == (ith reel, j-th row)

        if not num_reels == reels_matrix.num_reels:
            raise RuntimeError("Invalid Board: Number of reels must match the actuall csv reels")
    

        self.reels_count =num_reels
        self.rows_count = num_rows
        self.csv_representation = reels_matrix

    
    def generate_board(self):
        generate_jit_board(self.board, self.csv_representation.matrix, self.reels_count, self.rows_count)

@jit(nopython = True)
def generate_jit_board(board: NDArray, csv_matrix: NDArray, reels_cout:int, rows_count:int):

    for reel in range(reels_cout):
        row_start_index = np.random.randint(0, csv_matrix.shape[0])
        indices = (row_start_index + np.arange(rows_count)) % csv_matrix.shape[0]
        board[reel] = csv_matrix[indices, reel]




# rm:ReelsMatrix = ReelsMatrix("reels_base.csv")
# board_test:Board = Board(5, 5, rm )
# board_test.generate_board()
# print(board_test.board)
# board_test.generate_board()
# print(board_test.board)

# board_test_fail:Board = Board(10, 5, rm )



@dataclass
class PayTable:
    
    pay_table: dict[int, list[int]]

    def __post_init__(self):
        for symbol, payout_list in self.pay_table.items():
            for val in payout_list:
                if val < 0:
                    raise RuntimeError(f"Invalid PayTable: negative payout for symbol {symbol}")

    def get_pay(self, symbol: int, cluster_size: int) -> int:
        if symbol not in self.pay_table:
            return 0
        entries = self.pay_table[symbol]
        
        idx = min(cluster_size, len(entries)) - 1
        if idx < 0:
            return 0
        return entries[idx]
    
@dataclass
class WinInfo:
    total_win:     int
    win_breakdown: list[dict]

class HitRateVector: # represent a result vector [arv_payout, win_hit_rate, ... {val_i avrg hit rate : val_i € [0, max_val]}]

    def __iadd__(self, outcome: WinInfo):
        self.data[0] += outcome.total_win
        self.data[1] += int(outcome.total_win > 0)

        for entry in outcome.win_breakdown:
            self.data[2 + entry["symbol"]] += entry["win"]
        self.games += 1

        return self
    
    def finalize(self):
        hit_count     = self.data[1]                                       
        self.data[0]  = self.data[0] / hit_count if hit_count > 0 else 0.0 
        self.data[1]  = hit_count / self.games                              
        self.data[2:] = self.data[2:] / self.games   

    def avg_payout(self):
        return self.data[0]
    
    def win_hit_rate(self):
        return self.data[1]
    
    def symbol_hit_rate(self, symbol: int) -> float:
        if not 0 <= symbol <= self.data.size - 2:
            raise IndexError(f"Symbol {symbol} out of range [0, {self.max_symbol}]")
        return self.data[2 + symbol]
    
    def __init__(self, data:NDArray[np.float64]):
        if data.size < 2:
            raise RuntimeError("Invalid HitRateVector: must at least be of size 2 to give avr payout + win hit rate")
        
        self.data  = data
        self.games = 0


    
    @classmethod
    def from_list(cls, float_list):
        numpy_vector = np.array(float_list, dtype=np.float64)
        return cls(numpy_vector)

    def __repr__(self) -> str:
        sym_rates = {s: round(self.symbol_hit_rate(s), 4) for s in range(self.data.size - 2 )}
        return (
            f"HitRateVector(\n"
            f"  avg_win  = {self.avg_payout():.4f}\n"
            f"  hit_rate = {self.win_hit_rate():.4f}\n"
            f"  per_sym  = {sym_rates}\n"
            f")"
        )





