from reels_csv_matrix import ReelsMatrix
from models import Board, PayTable, HitRateVector
from win_mechanics import ClusterWinMechanic
from classification_head import ClassificationHead
from simulator import ReelSetSimulator
import time

rm  = ReelsMatrix("reels_base.csv")
pt  = PayTable({
    0: [0, 0, 0, 0, 10, 20,  30,  40,  50],
    1: [0, 0, 0, 0, 10, 20,  30,  40,  50],
    2: [0, 0, 0, 0, 10, 20,  30,  40,  50],
    3: [0, 0, 0, 0, 20, 40,  60,  80, 100],
    4: [0, 0, 0, 0, 20, 40,  60,  80, 100],
    5: [0, 0, 0, 0, 20, 40,  60,  80, 100],
    6: [0, 0, 0, 0, 30, 60,  90, 120, 150],
    7: [0, 0, 0, 0, 30, 60,  90, 120, 150],
    8: [0, 0, 0, 0, 50, 100, 200, 400, 800],
})

mech = ClusterWinMechanic()
spins = 1000000

sim  = ReelSetSimulator(
    csv_obj       = rm,
    win_mechanic  = mech,
    pay_table     = pt,
    board_reels   = 5,
    board_rows    = 5,
    num_iterations= spins,
)

start  = time.time()
result = sim.run()
elapsed = time.time() - start

print(f"{spins} : spins in {elapsed:.3f}s  ({spins/elapsed:.0f} spins/sec)")
print(result.hit_rate_vector)