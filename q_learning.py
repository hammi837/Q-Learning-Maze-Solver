"""
Lesson 2: First Q-Learning Project — Maze Solver
A small maze solved with tabular Q-Learning (stdlib only).
"""

import random
import sys

# Windows terminals often default to cp1252; allow ↑↓←→ to print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------------------
# 1. Maze setup
# ------------------------------------------------------------
ROWS = 5
COLS = 5

START = (0, 0)
GOAL = (4, 4)
WALLS = {(1, 1), (2, 1), (3, 3), (1, 3)}

# Actions: 0=up, 1=down, 2=left, 3=right
ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

# ------------------------------------------------------------
# 2. Hyperparameters
# ------------------------------------------------------------
EPISODES = 1000
ALPHA = 0.1       # learning rate
GAMMA = 0.9       # discount factor
EPSILON = 0.2     # exploration rate

# ------------------------------------------------------------
# 3. Q-table: state -> [Q(up), Q(down), Q(left), Q(right)]
# ------------------------------------------------------------
q_table = {
    (r, c): [0.0, 0.0, 0.0, 0.0]
    for r in range(ROWS)
    for c in range(COLS)
    if (r, c) not in WALLS
}


def is_valid(pos):
    r, c = pos
    return 0 <= r < ROWS and 0 <= c < COLS and pos not in WALLS


def take_action(state, action):
    """Move the agent; return (next_state, reward, done)."""
    dr, dc = ACTIONS[action]
    next_state = (state[0] + dr, state[1] + dc)

    if not is_valid(next_state):
        # Hit wall or boundary — stay put, small penalty
        return state, -1, False

    if next_state == GOAL:
        return next_state, 10, True

    # Small step cost encourages shorter paths
    return next_state, -0.1, False


def choose_action(state):
    """Epsilon-greedy action selection."""
    if random.random() < EPSILON:
        return random.randint(0, 3)
    return max(range(4), key=lambda a: q_table[state][a])


# ------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------
print(f"Training for {EPISODES} episodes...")

for episode in range(EPISODES):
    state = START

    for _ in range(100):  # max steps per episode
        action = choose_action(state)
        next_state, reward, done = take_action(state, action)

        # Q-learning update
        old_q = q_table[state][action]
        next_max = max(q_table[next_state]) if next_state in q_table else 0.0
        q_table[state][action] = old_q + ALPHA * (reward + GAMMA * next_max - old_q)

        state = next_state
        if done:
            break

    if (episode + 1) % 200 == 0:
        print(f"  Episode {episode + 1}/{EPISODES} complete")

print("Training finished.\n")

# ------------------------------------------------------------
# 5. Print learned maze policy
# ------------------------------------------------------------
arrows = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}

print("\nLearned Maze Policy:\n")
for row in range(ROWS):
    line = []
    for col in range(COLS):
        state = (row, col)
        if state == START:
            line.append(" S ")
        elif state == GOAL:
            line.append(" G ")
        elif state in WALLS:
            line.append(" # ")
        else:
            best_action = max(
                range(4),
                key=lambda a: q_table[state][a],
            )
            line.append(f" {arrows[best_action]} ")
    print(" ".join(line))

# ------------------------------------------------------------
# 6. Test the trained agent
# ------------------------------------------------------------
print("\nTesting trained agent:")
state = START
path = [state]

for _ in range(30):
    if state == GOAL:
        break
    action = max(
        range(4),
        key=lambda a: q_table[state][a],
    )
    state, reward, done = take_action(state, action)
    path.append(state)
    if done:
        break

print("Agent path:", path)
if state == GOAL:
    print("Success! Agent reached the goal.")
else:
    print("Agent did not reach the goal.")
