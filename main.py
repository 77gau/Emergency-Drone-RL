import os
import pygame

from environment import (
    DroneEnvironment,
    WIDTH,
    HEIGHT
)

from q_learning import QLearningDrone


def main():
    pygame.init()

    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT)
    )

    pygame.display.set_caption(
        "2D Emergency Response Simulation - Q-learning Drone"
    )

    clock = pygame.time.Clock()

    environment = DroneEnvironment(screen)
    agent = QLearningDrone(environment)

    automatic_mode = False
    running = True

    print("CONTROLS")
    print("Arrow keys = move drone manually")
    print("T = train Q-learning agent")
    print("A = run learned policy")
    print("R = reset mission")
    print("S = save results")
    print("ESC = quit")

    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    automatic_mode = False
                    agent.demo_active = False
                    environment.reset()

                elif event.key == pygame.K_t:
                    automatic_mode = False
                    agent.demo_active = False

                    agent.train()

                    environment.reset()

                elif event.key == pygame.K_a:

                    if agent.load():
                        agent.start_demo()
                        automatic_mode = True
                    else:
                        environment.last_event = (
                            "No saved Q-table. Press T to train."
                        )

                        print(
                            "Train first by pressing T."
                        )

                elif event.key == pygame.K_s:
                    agent.save_results()

                    environment.last_event = (
                        "Training results saved"
                    )

                elif not automatic_mode:

                    moves = {
                        pygame.K_UP: (0, -1),
                        pygame.K_DOWN: (0, 1),
                        pygame.K_LEFT: (-1, 0),
                        pygame.K_RIGHT: (1, 0),
                    }

                    if event.key in moves:
                        environment.move_drone(
                            *moves[event.key]
                        )

        # Automatic learned-policy demonstration
        if automatic_mode:
            agent.demo_step()

            if not agent.demo_active:
                automatic_mode = False

        # Draw the map and learned path when trained
        environment.draw(
            screen,
            q_agent=agent if agent.trained else None
        )

        pygame.display.flip()

        clock.tick(8 if automatic_mode else 30)

    pygame.quit()


if __name__ == "__main__":
    main()