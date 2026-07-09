
import os
import pandas as pd
import numpy as np

#Read a csv base file of a reels set and create a wrapper for it represented via numpy matrix
class ReelsMatrix:
    def __init__(self, base_csv_path: str):

        reels_csv_reader = pd.read_csv(base_csv_path, header=None)
        self.matrix      = reels_csv_reader.values

        self.num_reels = self.matrix.shape[1]
        self.num_rows  = self.matrix.shape[0]

        self.min_val = int(self.matrix.min())
        if self.min_val < 0:
            raise RuntimeError("Invalid csv: vals must start from 0")
        self.max_val = int(self.matrix.max())

        expected_vals = np.arange(self.min_val, self.max_val+1)
        actual_vals   = np.unique(self.matrix)
        if not np.array_equal(expected_vals, actual_vals):
            raise RuntimeError("Invalid csv: missing values between min and max boundary")
        
    def get_reel(self,i: int):
            return self.matrix[: , i]
    
    def get_numb_symbols(self):
         return self.max_val - self.min_val + 1
    
    def serialize(self, write_full_path: str):
        os.makedirs(os.path.dirname(write_full_path), exist_ok=True)
        np.savetxt(write_full_path, self.matrix, delimiter=",", fmt="%d")
    


#rm:ReelsMatrix = ReelsMatrix("reels_base.csv")
#print(rm.get_reel(0))

# print(rm.max_val)
# print(rm.min_val)
# print(rm.matrix[9, 3]) 





