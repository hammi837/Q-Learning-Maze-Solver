"""
Lesson 4: Improving Q-Learning — Maze Solver
Epsilon decay, reward tracking, and Matplotlib visualization.
"""

import random
import sys

import matplotlib.pyplot as plt

# Windows terminals often default to cp1252; allow ↑↓←→ to print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =====================================
# 1. MAZE SETUP (same 5×5 environment)
# =====================================
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

ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}

# =====================================
# 2. HYPERPARAMETERS
# =====================================
episodes = 1000
max_steps = 100
alpha = 0.1       # learning rate
gamma = 0.9       # discount factor

# Epsilon decay schedule
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

# =====================================
# 3. Q-TABLE
# =====================================
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
        # Hit wall or boundary
        return state, -2, False

    if next_state == GOAL:
        return next_state, 10, True

    # Normal movement
    return next_state, -1, False


def choose_action(state, epsilon):
    """Epsilon-greedy action selection."""
    if random.random() < epsilon:
        return random.randint(0, 3)

    values = q_table[state]
    best_value = max(values)
    best_actions = [
        a for a, value in enumerate(values)
        if value == best_value
    ]
    return random.choice(best_actions)


# =====================================
# 4. TRAINING
# =====================================
episode_rewards = []
epsilon_history = []

print(f"Training for {episodes} episodes...\n")

for episode in range(episodes):
    state = START
    total_reward = 0

    for _ in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done = take_action(state, action)

        # Q-learning update
        old_q = q_table[state][action]
        next_max = max(q_table[next_state]) if next_state in q_table else 0.0
        q_table[state][action] = old_q + alpha * (
            reward + gamma * next_max - old_q
        )

        total_reward += reward
        state = next_state

        if done:
            break

    episode_rewards.append(total_reward)
    epsilon_history.append(epsilon)

    # Decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(
            f"Episode {episode + 1}/{episodes} | "
            f"Reward: {total_reward} | "
            f"Epsilon: {epsilon:.3f}"
        )

print("\nTraining finished!")

# =====================================
# 5. MATPLOTLIB VISUALIZATION
# =====================================
window = 50
moving_avg = []
for i in range(len(episode_rewards)):
    start = max(0, i - window + 1)
    moving_avg.append(
        sum(episode_rewards[start:i + 1]) / (i - start + 1)
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Episode rewards
axes[0].plot(episode_rewards, alpha=0.35, label="Episode reward")
axes[0].plot(moving_avg, linewidth=2, label=f"{window}-episode moving avg")
axes[0].set_title("Episode Rewards")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Epsilon decay
axes[1].plot(epsilon_history, color="orange")
axes[1].set_title("Epsilon Decay")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Epsilon")
axes[1].set_ylim(0, 1.05)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("q_learning_results.png", dpi=150)
print("\nSaved graph: q_learning_results.png")
plt.close()

# =====================================
# 6. LEARNED POLICY
# =====================================
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
            line.append(f" {ARROWS[best_action]} ")
    print(" ".join(line))

# =====================================
# 7. TEST TRAINED AGENT
# =====================================
print("\nTesting trained agent:")
state = START
path = [state]

for step in range(max_steps):
    if state == GOAL:
        break

    values = q_table[state]
    best_value = max(values)
    best_actions = [
        a for a, value in enumerate(values)
        if value == best_value
    ]
    action = random.choice(best_actions)

    state, reward, done = take_action(state, action)
    path.append(state)

    if done:
        break

print("Agent path:", path)
if state == GOAL:
    print("Success! Agent reached the goal.")
else:
    print("Agent did not reach the goal.")
