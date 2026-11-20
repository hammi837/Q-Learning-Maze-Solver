# Q-Learning Maze Solver

A small maze solved with tabular Q-Learning using only the Python standard library.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Run

```powershell
python q_learning.py
```

## What you'll see

- 1,000 training episodes
- A learned maze policy (arrows show the best action in each cell)
- The agent's final path from start (`S`) to goal (`G`)
- A success message if the agent reaches the goal

The exact policy and path can vary because the agent explores randomly during training.

## How it works

1. The agent explores a 5×5 maze with walls.
2. Q-values are updated with the standard Q-learning rule.
3. After training, a greedy policy is printed and tested.
