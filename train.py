import copy
import random
import math
import numpy as np
from collections import defaultdict
import pickle
from util import NTupleApproximator ,save_approximator
from student_agent import Game2048Env
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def td_learning(env,approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    final_scores = []
    success_flags = []
    Avg_score = []
    step_array = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            best_action = legal_moves[0]
            best_value = -float('inf')
            
            for action in legal_moves:
                env_copy = copy.deepcopy(env)
                _,score,_,board = env_copy.step(action)
                value = (score-previous_score) + gamma * approximator.value(board)
                if value > best_value:
                    best_action = action
                    best_value  = value

            current_state = copy.deepcopy(state)  
           
            next_state, new_score, done, _ = env.step(best_action)
 
            incremental_reward = new_score - previous_score
            
            max_tile = max(max_tile, np.max(next_state))
      
            trajectory.append((current_state, copy.deepcopy(next_state),incremental_reward))

            previous_score = new_score
            state = next_state
        trajectory.append((copy.deepcopy(state), None, 0))
        for (state, next_state, reward) in trajectory:
            if next_state is not None:
                delta = reward + gamma * approximator.value(next_state) - approximator.value(state)
            else:
                delta = reward - approximator.value(state)
                
            approximator.update(state, delta, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            Avg_score.append(avg_score)
            step_array.append(episode)
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
            #print(count)

        if (episode + 1) % 5000 == 0:
            save_approximator(approximator, f"checkpoint_ep{episode+1}.pkl")
            print(f"✅ Checkpoint saved at episode {episode + 1}")
    return final_scores ,Avg_score, step_array

patterns = [[(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)], [(1, 1), (2, 1), (3, 1),(1, 2), (2, 2), (3, 2)],[(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (3, 1)], [(0, 1), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2)],[(0, 2), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3)]]

#approximator = load_approximator(f"checkpoint_ep{start_episode}.pkl")
approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

td_learning(env,approximator, num_episodes=80000, alpha=0.1, gamma=0.99, epsilon=0.1)