


from abc import ABC, abstractmethod

from tools.reel_set_simulator.models import Board, PayTable, WinInfo


class WinMechanic(ABC):

    @abstractmethod
    def play(self, board: Board, pay_table: PayTable) -> WinInfo:
        # returns (total_win, list of {symbol, size, positions, win} per cluster)
        ...