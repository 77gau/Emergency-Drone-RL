
import csv
import os
import pickle
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import GRID_SIZE


class QLearningDrone:

    # Action order must remain consistent throughout training.
    actions = [
        (0, -1),  # Up
        (0, 1),   # Down
        (-1, 0),  # Left
        (1, 0),   # Right
        (0, 0),   # Hover
    ]

    action_names = [
        "up", "down", "left", "right", "hover"
    ]

    def __init__(self, environment, episodes=6000, max_steps=250):
        self.environment = environment
        self.episodes = episodes
        self.max_steps = max_steps

        self.alpha = 0.15
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.04
        self.epsilon_decay = 0.997

        self.q_table = np.zeros(
            (GRID_SIZE, GRID_SIZE, len(self.actions)),
            dtype=float
        )

        self.rewards_history = []
        self.steps_history = []
        self.success_history = []

        self.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results"
        )
        os.makedirs(self.results_dir, exist_ok=True)

        self.trained = False
        self.demo_active = False
        self.demo_steps = 0

    def distance(self, x, y):
        # Drone center is one cell down/right from its anchor.
        cx = x + 1
        cy = y + 1
        tx, ty = self.environment.target

        return abs(cx - tx) + abs(cy - ty)

    def reached_target(self, x, y):
        # Use the exact same target condition as environment.py.
        return self.environment.reached_target(x, y)

    def choose_action(self, x, y):
        if random.random() < self.epsilon:
            return random.randrange(len(self.actions))

        return int(np.argmax(self.q_table[y, x]))


    def step(self, x, y, action_index):
        dx, dy = self.actions[action_index]
        nx = x + dx
        ny = y + dy

        # Check building collisions first.
        if not self.environment.can_move_to(nx, ny):
            return x, y, -10.0, False

        # Fire collision destroys the drone and ends the episode.
        if self.environment.hazard_at(nx, ny):
            self.environment.last_event = "DRONE DESTROYED - FIRE COLLISION"
            return nx, ny, -100.0, True

        reward = -1.0

        old_distance = self.distance(x, y)
        new_distance = self.distance(nx, ny)

        if new_distance < old_distance:
            reward += 1.5
        elif new_distance > old_distance:
            reward -= 0.25

        done = self.reached_target(nx, ny)

        if done:
            reward += 100.0

        return nx, ny, reward, done



    def train(self):
        print("Starting Q-learning training...")

        self.environment.last_event = "Training started"
        self.destroyed_history = []

        for episode in range(self.episodes):
            x, y = self.environment.start

            total_reward = 0.0
            success = False
            destroyed = False
            steps_taken = 0

            for step_number in range(self.max_steps):
                action = self.choose_action(x, y)

                nx, ny, reward, done = self.step(x, y, action)

                # A fire collision ends the episode with a penalty.
                fire_collision = (
                    done
                    and reward <= -100.0
                )

                old_q = self.q_table[y, x, action]

                if done:
                    future_q = 0.0
                else:
                    future_q = float(np.max(self.q_table[ny, nx]))

                self.q_table[y, x, action] = old_q + self.alpha * (
                    reward + self.gamma * future_q - old_q
                )

                x, y = nx, ny
                total_reward += reward
                steps_taken = step_number + 1

                if fire_collision:
                    destroyed = True
                    break

                if done:
                    success = True
                    break

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            self.rewards_history.append(total_reward)
            self.steps_history.append(steps_taken)
            self.success_history.append(int(success))
            self.destroyed_history.append(int(destroyed))

            if (episode + 1) % 500 == 0:
                recent = self.success_history[-100:]
                rate = 100 * sum(recent) / len(recent)
                fires = sum(self.destroyed_history[-100:])

                print(
                    f"Episode {episode + 1}/{self.episodes} | "
                    f"Success rate: {rate:.1f}% | "
                    f"Fire destructions (last 100): {fires} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        self.trained = True
        self.environment.last_event = "Training complete"

        self.save_results()

        print("Training complete.")
        print(f"Results saved to: {self.results_dir}")

        return self.success_history


    def save_results(self):
        self.save_q_table()
        self.save_training_history()
        self.save_graphs()
        self.save_q_table_png()

    def save_q_table(self):
        # Save the full numerical Q-table.
        pkl_path = os.path.join(
            self.results_dir, "q_table.pkl"
        )

        with open(pkl_path, "wb") as file:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "target": self.environment.target,
                    "start": self.environment.start,
                    "actions": self.actions,
                },
                file
            )

        # Save a readable CSV.
        csv_path = os.path.join(
            self.results_dir, "q_table.csv"
        )

        with open(
            csv_path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow(["x", "y"] + self.action_names)

            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    writer.writerow(
                        [x, y] + list(self.q_table[y, x])
                    )

    def save_training_history(self):
        path = os.path.join(
            self.results_dir, "training_history.csv"
        )

        with open(
            path, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "episode", "total_reward", "steps", "success"
            ])

            for i, values in enumerate(
                zip(
                    self.rewards_history,
                    self.steps_history,
                    self.success_history
                ),
                start=1
            ):
                reward, steps, success = values

                writer.writerow([
                    i, reward, steps, success
                ])

    def save_graphs(self):
        if not self.rewards_history:
            return

        # Reward per episode.
        plt.figure(figsize=(9, 4))
        plt.plot(self.rewards_history, linewidth=0.8)
        plt.title("Q-learning: Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.results_dir, "episode_rewards.png"
            ),
            dpi=150
        )
        plt.close()

        # Rolling success rate.
        window = min(100, len(self.success_history))
        rolling_rates = []

        for i in range(len(self.success_history)):
            start = max(0, i - window + 1)
            recent = self.success_history[start:i + 1]

            rolling_rates.append(
                100 * sum(recent) / len(recent)
            )

        plt.figure(figsize=(9, 4))
        plt.plot(rolling_rates, linewidth=1.2)
        plt.title("Success Rate (Rolling Window)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 100)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.results_dir, "success_rate.png"
            ),
            dpi=150
        )
        plt.close()

        # Steps per episode.
        plt.figure(figsize=(9, 4))
        plt.plot(self.steps_history, linewidth=0.8)
        plt.title("Steps per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.results_dir, "episode_steps.png"
            ),
            dpi=150
        )
        plt.close()



    def save_q_table_png(self):
        import matplotlib.pyplot as plt
        import numpy as np

        best_actions = np.argmax(self.q_table, axis=2)
        best_values = np.max(self.q_table, axis=2)

        arrows = ["↑", "↓", "←", "→", "•"]

        fig, ax = plt.subplots(figsize=(18, 15))

        # Color cells according to their learned Q-values.
        image = ax.imshow(
            best_values,
            cmap="RdYlGn",
            interpolation="nearest",
            aspect="equal"
        )

        fig.colorbar(image, ax=ax, label="Best learned Q-value")

        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.set_xticklabels(range(GRID_SIZE))
        ax.set_yticklabels(range(GRID_SIZE))

        ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Label every cell with its coordinate, best action and Q-value.
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                action = int(best_actions[y, x])
                q_value = float(best_values[y, x])

                ax.text(
                    x, y,
                    f"({x},{y})\n{arrows[action]}\nQ={q_value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black"
                )

        sx, sy = self.environment.start
        tx, ty = self.environment.target

        ax.scatter(
            sx, sy, marker="o", s=180,
            facecolors="none", edgecolors="blue",
            linewidths=2, label=f"Start ({sx},{sy})",
            zorder=5
        )

        ax.scatter(
            tx, ty, marker="*", s=260,
            color="red", edgecolors="black",
            label=f"Target ({tx},{ty})",
            zorder=6
        )

        ax.set_title("Emergency Drone — Learned Q-Values and Policy")
        ax.set_xlabel("Grid X coordinate")
        ax.set_ylabel("Grid Y coordinate")
        ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1))

        plt.tight_layout()

        path = os.path.join(self.results_dir, "q_table.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Q-table image saved: {path}")
        print(
            f"Q-table range: {self.q_table.min():.2f} "
            f"to {self.q_table.max():.2f}"
        )



    def load(self):
        path = os.path.join(
            self.results_dir, "q_table.pkl"
        )

        if not os.path.isfile(path):
            print("No saved Q-table found.")
            return False

        try:
            with open(path, "rb") as file:
                saved = pickle.load(file)

            table = saved.get("q_table")

            if table is None or table.shape != self.q_table.shape:
                print("Saved Q-table has an incompatible shape.")
                return False

            if tuple(saved.get("target", ())) != tuple(
                self.environment.target
            ):
                print("Saved Q-table target does not match this map.")
                return False

            self.q_table = table
            self.trained = True

            self.environment.last_event = "Q-table loaded"
            return True

        except (
            OSError, pickle.PickleError, EOFError,
            AttributeError, ValueError
        ) as error:
            print(f"Could not load Q-table: {error}")
            return False

    def start_demo(self):
        self.environment.reset()
        self.demo_active = True
        self.demo_steps = 0

    def demo_step(self):
        if not self.demo_active:
            return

        env = self.environment

        x, y = env.drone_x, env.drone_y
        action = int(np.argmax(self.q_table[y, x]))

        dx, dy = self.actions[action]
        nx, ny = x + dx, y + dy

        if not env.can_move_to(nx, ny):
            env.last_event = (
                "Learned action blocked. Retrain the agent."
            )
            self.demo_active = False
            return

        env.move_drone(dx, dy)
        self.demo_steps += 1

        if (
            env.mission_complete
            or env.destroyed
            or self.demo_steps >= self.max_steps
        ):
            self.demo_active = False
