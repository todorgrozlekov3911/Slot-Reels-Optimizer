from dataclasses import dataclass
from tools.reel_set_simulator.models import Board, PayTable, WinInfo
from tools.reel_set_simulator.win_mechanics.win_mechanic_base import WinMechanic
from numpy.typing import NDArray
import numpy as np
from numba import jit


@dataclass
class LinesData:
    possible_lines: NDArray

    def __post_init__(self):
        if self.possible_lines.ndim != 2:
            raise RuntimeError("The used line configurations must be all from the same len, if NOT -> numpy fallbacks to 1D vector of Python objects")
        
    
    @classmethod
    def from_list(cls, data: list[list[int]])->"LinesData":
        lines_array = np.array(data, dtype=np.int32)
        return cls(possible_lines = lines_array)

class LineWinMechanic(WinMechanic):
    def __init__(self, lines_data: LinesData):
        self.lines_data = lines_data

    def play(self, board: Board, pay_table: PayTable):
        
        total_win, win_symbols, win_lengths, win_payouts = jit_line_wins(board.board, self.lines_data.possible_lines,
                                                                         self.build_numpy_lines_pay_table(pay_table))
        
        win_log = []
        for i in range(len(win_symbols)):
            if win_symbols[i] == -1:
                continue
            
            win_log.append({
                "symbol": int(win_symbols[i]),
                "len"   : int(win_lengths[i]),
                "pay"   : int(win_payouts[i]),
                "line_type" : i
            })


        return WinInfo(total_win, win_log)
                
    def build_numpy_lines_pay_table(self, pay_table:PayTable):
        line_len = max(len(v) for v in pay_table.pay_table.values())

        arr = np.zeros((len(pay_table.pay_table), line_len), dtype=np.int32)
        for symbol, symbol_pay in pay_table.pay_table.items():
            for idx, val in enumerate(symbol_pay):
                arr[symbol, idx] = val

        return arr



@jit(nopython = True)
def jit_line_wins(board:NDArray, lines:NDArray, paytable: NDArray):

    num_lines        = lines.shape[0]
    max_len_of_lines = lines.shape[1]
    max_pay_len      = paytable.shape[1]

    win_symbols = np.full(num_lines, -1, dtype=np.int32)
    win_lengths = np.zeros(num_lines, dtype=np.int32)
    win_payouts = np.zeros(num_lines, dtype=np.int32)

    total_win = 0
    for line_type in range(num_lines):
        origin_symbol = board[0,lines[line_type, 0]]
        count = 1
        for reel in range(1,max_len_of_lines):
            curr_symbol = board[reel, lines[line_type, reel]]
            count += 1 * (curr_symbol == origin_symbol)
            if not curr_symbol == origin_symbol:
                break

        pay_idx = min(count, max_pay_len) - 1
        pay     = paytable[origin_symbol, pay_idx]

        if pay > 0:
            win_symbols[line_type] = origin_symbol  
            win_lengths[line_type] = count
            win_payouts[line_type] = pay
            total_win += pay


    return total_win, win_symbols, win_lengths, win_payouts
