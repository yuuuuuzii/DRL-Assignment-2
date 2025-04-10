from util import TD_MCTS, TD_MCTS_Node, load_approximator
from student_agent import Game2048Env
import copy

env = Game2048Env()
approximator = load_approximator(f"checkpoint_ep{20000}.pkl")
td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=7.41, rollout_depth=10, gamma=0.99)

state = env.reset()
env.render()

done = False
round = 0
while not done:
    # Create the root node from the current state
    root = TD_MCTS_Node(env, state, env.score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    #print("TD-MCTS selected action:", best_act)

    # Execute the selected action and update the state
    state, reward, done, _ = env.step(best_act)
    round += 1

    td_mcts.env = env

    if(round % 100 == 0):
        print(f"reward: at {round} is {reward}")
    #env.render(action=best_act)

print("Game over, final score:", env.score)