import math
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import display, clear_output

## PARAMETERS
GRID_SIZE = 5
NUM_ACTIONS = 4
GOAL_REWARD = 1
OBSTACLE_REWARD = -1
EPSILON = 1 ## Will be reduced every step
ALPHA = 0.2
GAMMA = 0.9
NUM_EPISODES = 500

## INITIALIZE Q-VALUES FOR EACH STATE-ACTION PAIR
Q = np.zeros((GRID_SIZE, GRID_SIZE, NUM_ACTIONS))

## DEFINE ACTIONS (UP, DOWN, LEFT, RIGHT)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

## DEFINE THE GRID
grid_world = np.zeros((GRID_SIZE, GRID_SIZE))

grid_world[1, 3] = OBSTACLE_REWARD
grid_world[3, 2] = OBSTACLE_REWARD

grid_world[:, 0] = OBSTACLE_REWARD
grid_world[:, -1] = OBSTACLE_REWARD
grid_world[0, :] = OBSTACLE_REWARD
grid_world[-1, :] = OBSTACLE_REWARD

goal_position = np.random.randint(0, GRID_SIZE, size=2)
grid_world[GRID_SIZE-2, GRID_SIZE-2] = GOAL_REWARD

## FUNCTION TO CHECK IF THE STATE IS TERMINAL (GOAL OR OBSTACLE)
report = []
def is_terminal(state, steps):
    if grid_world[state] == GOAL_REWARD:
        report.append((1,steps))
    elif grid_world[state] == OBSTACLE_REWARD:
        report.append((0,steps))
    return grid_world[state] in [GOAL_REWARD, OBSTACLE_REWARD]

## EPSILON-GREEDY POLICY
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def custom_decay(epsilon, episode, num_episodes):
    midpoint = num_episodes / 2
    steepness = 10  # Adjust steepness for the sigmoid function if you guys want ;)
    return epsilon * sigmoid(-(steepness * (episode - midpoint) / num_episodes))

x = [i for i in range(NUM_EPISODES)]

plt.style.use('dark_background')
decay = [custom_decay(EPSILON, i, NUM_EPISODES) for i in x]

plt.plot(x, decay, color='red')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Over Time/Episodes')
plt.show()
    
def epsilon_greedy_policy(state, Q, episode):
    if np.random.rand() < custom_decay(EPSILON, episode, NUM_EPISODES):
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(Q[state])

## FUNCTION TO TAKE AN ACTION AND GET THE NEXT STATE + REWARD
def take_action(state, action):
    row, col = state
    row_change, col_change = ACTIONS[action]
    new_row, new_col = row + row_change, col + col_change

    new_row = max(0, min(new_row, GRID_SIZE - 1))
    new_col = max(0, min(new_col, GRID_SIZE - 1))

    next_state = (new_row, new_col)
    reward = grid_world[next_state[0], next_state[1]]

    return next_state, reward


def visualize_grid_with_agent(ax, grid, agent_position):
    cmap = plt.cm.colors.ListedColormap(['red', 'green', 'blue'])

    bounds = [OBSTACLE_REWARD, 0, GOAL_REWARD, np.max(grid)]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.clear()

    ax.matshow(grid, cmap=cmap, norm=norm)
    ax.text(agent_position[1], agent_position[0], 'A', color='black', ha='center', va='center', fontsize=15)
    
## Q-LEARNING ALGO
for episode in tqdm(range(NUM_EPISODES)):
    state = (1, 1)  # Starting state

    fig, ax = plt.subplots()

    reward = 0
    steps = 0
    while not is_terminal(state, steps):
        action = epsilon_greedy_policy(state, Q, episode)
        previous_state = state
        next_state, action_reward = take_action(state, action)

        ## UPDATE Q-VALUE
        Q[state][action] = (1 - ALPHA) * Q[state][action] + ALPHA * (action_reward + GAMMA * np.max(Q[next_state]))

        state = next_state

        if ((episode+1)%50 == 0 or episode == 0) and previous_state != state:
            visualize_grid_with_agent(ax, grid_world, previous_state)
            plt.pause(0.5)
            visualize_grid_with_agent(ax, grid_world, state)
            clear_output(wait=True)
            display(fig)
        steps += 1
    
    plt.close()

## RESULTS

# Epsilon + Success
success = [i[0] for i in report]
plt.scatter([i for i in range(len(success))], success, color='blue')
plt.scatter(x, decay, color='red')
plt.xlabel('Episode')
plt.ylabel('Success or Not')
plt.title('Success Plot')
plt.show()

# Success Rate
success_rate = [sum(success[max(len(success[:i+1])-100,0):i+1]) / len(success[max(len(success[:i+1])-100,0):i+1]) for i in range(len(success))]
plt.plot(range(len(report)), success_rate, linestyle='-')
plt.xlabel('Time (Episodes)')
plt.ylabel('Success Rate of Last 100')
plt.title('Success Rate Over Time/Episodes')
plt.ylim(0, 1)
plt.show()

# N Steps
n_steps = [i[1] for i in report]
plt.plot(range(len(report)), n_steps, linestyle='-')
plt.xlabel('Time (Episodes)')
plt.ylabel('N Steps')
plt.title('N Steps every Episode')
plt.show()

# All
plt.subplot(2,2,1)
plt.plot(x, decay, color='red')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.subplot(2,2,2)
plt.scatter([i for i in range(len(success))], success, color='blue')
plt.xlabel('Episode')
plt.ylabel('Success or Not')
plt.subplot(2,2,3)
plt.plot(range(len(report)), success_rate, linestyle='-')
plt.xlabel('Time (Episodes)')
plt.ylabel('Success Rate of Last 100')
plt.ylim(0, 1)
plt.subplot(2,2,4)
plt.plot(range(len(report)), n_steps, linestyle='-')
plt.xlabel('Time (Episodes)')
plt.ylabel('N Steps')
plt.show()


print("Learned Q-values:")
print(Q)
