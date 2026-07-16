
from dataclasses import dataclass
from typing import Optional
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

    @classmethod
    def from_matrix(cls, num_reels: int, num_rows: int, matrix: NDArray) -> "Board":
        obj = cls.__new__(cls)
        obj.board = np.full((num_reels, num_rows), -1)
        obj.reels_count = num_reels
        obj.rows_count = num_rows
        obj.csv_representation = type('MatrixWrapper', (), {
            'matrix': matrix,
            'num_reels': matrix.shape[1],
            'num_rows': matrix.shape[0],
        })()
        return obj

    def generate_board(self):
        generate_jit_board(self.board, self.csv_representation.matrix, self.reels_count, self.rows_count)

@jit(nopython = True)
def generate_jit_board(board: NDArray, csv_matrix: NDArray, reels_cout:int, rows_count:int):

    for reel in range(reels_cout):
        row_start_index = np.random.randint(0, csv_matrix.shape[0])
        indices = (row_start_index + np.arange(rows_count)) % csv_matrix.shape[0]
        board[reel] = csv_matrix[indices, reel]





PAYOT_MULTIPLIER = 100 # standartly we represent money/payout in cents -> 1$ == 100 as numeric val
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
            self.data[2 + entry["symbol"]] += 1
        self.games += 1

        return self
    
    def copy(self)->"HitRateVector":
        new = HitRateVector(self.data.copy())
        new.games = self.games
        return new
    
    def finalize(self):
        hit_count     = self.data[1]                                       
        self.data[0]  = self.data[0] / hit_count / PAYOT_MULTIPLIER if hit_count > 0 else 0.0 
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

@dataclass
class Chromosome:
    reel_matrix: NDArray
    fitness: float = float('inf')
    hit_rate_vector: Optional[HitRateVector] = None
    eval_count: int = 0


    def copy(self)->"Chromosome":
        return Chromosome(
            reel_matrix = self.reel_matrix.copy(),
            fitness=self.fitness,
            hit_rate_vector=self.hit_rate_vector.copy() if self.hit_rate_vector else None,
            eval_count= self.eval_count
        )

    def invalidate(self):
        self.fitness = float('inf')
        self.hit_rate_vector= None
        self.eval_count = 0




