
import numpy as np
from numpy.typing import NDArray
from reels_csv_matrix import ReelsMatrix



class ClassificationHead: #A matrix (K,M) where K_j represents j-th reel & M_i represent the count of occurances of the symbol i on reel j

    def __init__(self, count_matrix: NDArray):
        if not count_matrix.ndim == 2:
            raise RuntimeError("Invalid Classification Head input: it must be represented via a 2D matrix")
        
        self.classification_head = count_matrix

        self.row_count  = self.classification_head.shape[0]
        self.reel_count = self.classification_head.shape[1]

    @classmethod
    def from_reels_matrix(cls, reel_matrix: ReelsMatrix)-> "ClassificationHead": #forward declarirane v python , beshe mi zabavno da go napisha

        init_head = np.zeros((reel_matrix.get_numb_symbols(), reel_matrix.num_reels), dtype=np.int32)

        for i in range(reel_matrix.num_rows):
            for j in range(reel_matrix.num_reels):
                val = reel_matrix.matrix[i, j]
                init_head[val, j] +=1

        return cls(init_head)
    

    



