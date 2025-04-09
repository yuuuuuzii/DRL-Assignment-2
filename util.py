import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = np.zeros((4, 16, 16, 16, 16, 16, 16), dtype = float)
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)
    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        def rotate(x, y, N, i):
            if i == 0:
                return (x, y)
            elif i == 1:
                return (y, N - 1 - x)
            elif i == 2:
                return (N - 1 - x, N - 1 - y)
            elif i == 3:
                return (N - 1 - y, x)
            
        def mirro(x, y, N):
            return (x, N - 1 - y)
        
        arr = []

        for i in range(4):
            block = []
            mir = []
            for j in range(len(pattern)):
                block.append(rotate(pattern[j][0], pattern[j][1], 4, i))
                temp = mirro(pattern[j][0],pattern[j][1], 4)
                mir.append(rotate(temp[0], temp[1], 4, i))
            arr.append(block)
            arr.append(mir)

        return arr

    def tile_to_index(self, tile): ## 1024轉成10
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords): ##coords 為一個 pattern 
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for (x , y) in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        total = 0.0
        self.features = [self.get_feature(board, coords) for coords in self.symmetry_patterns]
        for i, feature in enumerate(self.features):
            group_idx = i // 8
            total += self.weights[group_idx][tuple(feature)]
        return total
    
    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        bonus = alpha * delta/len(self.symmetry_patterns)
        self.features = [self.get_feature(board, coords) for coords in self.symmetry_patterns]
        for i, feature in enumerate(self.features):
            group_idx = i // 8
            self.weights[group_idx][tuple(feature)] += bonus

def save_approximator(approximator, filename='approximator.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(approximator, f)

def load_approximator(filename='approximator.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)