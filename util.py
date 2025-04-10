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
        self.weights = np.zeros((6, 16, 16, 16, 16, 16, 16), dtype = float)
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
    

class TD_MCTS_Node:
    def __init__(self, env ,state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.gamma = 0.99
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=5.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        best_score = -float('inf')
        best_children = None
        for action, children in node.children.items():
            if children.visits == 0:
                return children
            
            score = self.approximator.value(children.state) + self.c * np.sqrt(2*np.log(node.visits)/children.visits)
            
            if score > best_score:
                best_score = score
                best_children = children

        return best_children
    # def rollout(self, sim_env, depth):
    #     # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
    #     # TODO: Use the approximator to evaluate the final state.
    #     prev_score = sim_env.score
    #     score = sim_env.score
    #     for _ in range(depth):
    #         legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
    #         if not legal_moves:
    #             break
    #         # TODO: Use your N-Tuple approximator to play 2048
    #         action = np.random.choice(legal_moves) 
    #         _, score, done, _  = sim_env.step(action)
    #         if done:
    #             break
        
    #     return (score - prev_score) + self.approximator.value(sim_env.board)
    
    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while(node != None):
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
       
        while(node.fully_expanded() and node.children): # 沒有動作可以選過才要往下 然後 要有小孩才會往下 不然沒小孩要expand
            node = self.select_child(node)

            _,_,_,_ = sim_env.step(node.action)
            
        # TODO: Expansion: if the node has untried actions, expand one.
        
        action = np.random.choice(node.untried_actions) ## 隨便選一個number
        
        board,score,_,afterstate = sim_env.step(action)

        child = TD_MCTS_Node(self.env, afterstate, score, node, action)
        
        node.untried_actions.remove(action)
        node.children[action] = child
        # rollout_reward = self.rollout(copy.deepcopy(sim_env),self.rollout_depth)
        self.backpropagate(node, self.approximator.value(afterstate))


    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution