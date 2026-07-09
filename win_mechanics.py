from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque

from models import Board, PayTable, WinInfo

class WinMechanic(ABC):

    @abstractmethod
    def play(self, board: Board, pay_table: PayTable) -> WinInfo:
        # returns (total_win, list of {symbol, size, positions, win} per cluster)
        ...

class ClusterWinMechanic(WinMechanic):

    def play(self, board: Board, pay_table: PayTable) -> WinInfo:
        visited = set()
        total_win = 0
        win_breakdown = []

        for reel in range(board.reels_count):
            for row in range(board.rows_count):
                if (reel, row) in visited:
                    continue

                symbol = board.board[reel, row]
                cluster = self.bfs(board, reel, row, symbol, visited)
                payout = pay_table.get_pay(symbol, len(cluster))

                if payout > 0:
                    total_win += payout
                    win_breakdown.append({
                        "symbol":    symbol,
                        "size":      len(cluster),
                        "positions": cluster,
                        "win":       payout,
                    })

        return WinInfo(total_win, win_breakdown)

    def bfs(
        self,
        board: Board,
        start_reel: int,
        start_row: int,
        symbol: int,
        visited: set,
    ) -> list[tuple[int, int]]:
        cluster = []
        queue = deque([(start_reel, start_row)])

        while queue:
            reel, row = queue.popleft()

            if (reel, row) in visited:
                continue
            if board.board[reel, row] != symbol:
                continue

            visited.add((reel, row))
            cluster.append((reel, row))

            for neighbor in self._get_neighbors(board, reel, row):
                if neighbor not in visited:
                    queue.append(neighbor)

        return cluster

    def _get_neighbors(
        self,
        board: Board,
        reel: int,
        row: int,
    ) -> list[tuple[int, int]]:
        candidates = [
            (reel + 1, row),  # right
            (reel - 1, row),  # left
            (reel, row + 1),  # down
            (reel, row - 1),  # up
        ]
        return [
            (r, ro)
            for r, ro in candidates
            if 0 <= r < board.reels_count and 0 <= ro < board.rows_count
        ]