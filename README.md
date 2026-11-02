# Emergency Drone RL

A 2D emergency-response drone simulation built with Python and Pygame. A Q-learning agent learns to navigate a grid from its starting point to a rescue target while avoiding buildings and fire hazards.

## Simulation

<img width="1475" height="1010" alt="image" src="https://github.com/user-attachments/assets/74ee309f-cd22-4042-b6e2-e72e7d7940bf" />


## Features

- Manually fly the drone or train a Q-learning agent.
- Run the agent's learned policy in the simulation.
- Save the Q-table, training history, and charts.

## Requirements

- Python
- Pygame
- NumPy
- Matplotlib

## Setup and run

Install the packages:

```bash
python -m pip install pygame numpy matplotlib
```

Start the simulation from the project folder:

```bash
python main.py
```

## Controls

| Key | Action |
| --- | --- |
| Arrow keys | Move the drone manually |
| T | Train the Q-learning agent |
| A | Run the learned policy |
| R | Reset the mission |
| S | Save results |
| Esc | Quit |


Training results and charts are saved in the `results/` folder. See [Report.pdf](Report.pdf) for the project report.
