from abc import ABC, abstractmethod
from numpy.typing import NDArray
from collections import deque
from numba import jit
import numpy as np

from models import Board, PayTable, WinInfo

class WinMechanic(ABC):

    @abstractmethod
    def play(self, board: Board, pay_table: PayTable) -> WinInfo:
        # returns (total_win, list of {symbol, size, positions, win} per cluster)
        ...

class ClusterWinMechanic(WinMechanic):

    def play(self, board: Board, pay_table: PayTable) -> WinInfo:
        visited:NDArray = np.zeros((board.board.shape[0] , board.board.shape[1]), dtype = np.bool_)
        total_win = 0
        win_breakdown = []

        for reel in range(board.reels_count):
            for row in range(board.rows_count):
                if visited[reel, row]:
                    continue

                symbol = board.board[reel, row]
                cluster = self.bfs(board, reel, row, symbol, visited)
                payout = pay_table.get_pay(symbol, cluster)

                if payout > 0:
                    total_win += payout
                    win_breakdown.append({
                        "symbol":    symbol,
                        "size":      cluster,
                        "win":       payout,
                    })

        return WinInfo(total_win, win_breakdown)    

    def bfs(self,
        board: Board,
        start_reel: int,
        start_row: int,
        symbol: int,
        visited: NDArray,):
        return jit_bfs(board.board, start_reel, start_row, symbol, visited) 

@jit(nopython=True)
def jit_bfs(board: NDArray, start_reel: int, start_row: int, symbol: int, visited: NDArray):
    queue = np.empty((board.shape[0] * board.shape[1], 2), dtype=np.int32)
    head, tail = 0, 0

   
    queue[tail, 0] = start_reel
    queue[tail, 1] = start_row
    visited[start_reel, start_row] = True
    tail += 1
    cluster_size = 0

    while head != tail:
        curr_reel = queue[head, 0]
        curr_row = queue[head, 1]
        head += 1
        cluster_size += 1

        
        if curr_reel + 1 < board.shape[0] and not visited[curr_reel + 1, curr_row] and board[curr_reel + 1, curr_row] == symbol:
            visited[curr_reel + 1, curr_row] = True
            queue[tail, 0] = curr_reel + 1
            queue[tail, 1] = curr_row
            tail += 1

 
        if curr_reel - 1 >= 0 and not visited[curr_reel - 1, curr_row] and board[curr_reel - 1, curr_row] == symbol:
            visited[curr_reel - 1, curr_row] = True
            queue[tail, 0] = curr_reel - 1
            queue[tail, 1] = curr_row
            tail += 1

  
        if curr_row + 1 < board.shape[1] and not visited[curr_reel, curr_row + 1] and board[curr_reel, curr_row + 1] == symbol:
            visited[curr_reel, curr_row + 1] = True
            queue[tail, 0] = curr_reel
            queue[tail, 1] = curr_row + 1
            tail += 1

      
        if curr_row - 1 >= 0 and not visited[curr_reel, curr_row - 1] and board[curr_reel, curr_row - 1] == symbol:
            visited[curr_reel, curr_row - 1] = True
            queue[tail, 0] = curr_reel
            queue[tail, 1] = curr_row - 1
            tail += 1

    return cluster_size
    
            


