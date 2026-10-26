
import math
import os
import pygame

GRID_SIZE = 20
CELL_SIZE = 32

MAP_WIDTH = GRID_SIZE * CELL_SIZE
MAP_HEIGHT = GRID_SIZE * CELL_SIZE
PANEL_WIDTH = 340

WIDTH = MAP_WIDTH + PANEL_WIDTH
HEIGHT = MAP_HEIGHT

# Muted terrain colors
BACKGROUND = (126, 151, 119)
GRID_COLOR = (105, 127, 100)
PANEL_COLOR = (22, 40, 57)
TEXT_COLOR = (235, 244, 250)


class DroneEnvironment:

    def __init__(self, screen=None):
        self.screen = screen

        self.assets_dir = os.path.join(
            os.path.dirname(__file__), "assets"
        )

        self.images = {}
        self.load_assets()

        # Drone starts bottom-left.
        self.start = (1, 17)

        # Rescue target at the top-right.
        self.target = (18, 2)

        self.targets = [{
            "x": self.target[0],
            "y": self.target[1],
            "found": False
        }]

        # Building rectangles: x, y, width, height.
        # Buildings are the only solid obstacles.
        self.buildings = [
            (3, 1, 3, 4),
            (10, 2, 3, 2),
            (14, 6, 2, 4),
            (5, 8, 2, 3),
            (12, 14, 3, 3),
        ]

        # Rocks are visual scenery: the drone can fly over them.
        self.rock_piles = [
            (7, 5, 3, 2),
            (10, 12, 3, 2),
        ]

          # Tree clusters: repositioned to make the route more challenging
        self.tree_clusters = [
            (8, 0, 2, 2),
            (11, 0, 2, 2),
            (0, 10, 2, 3),
            (7, 4, 2, 2),
            (9, 7, 2, 2),
            (17, 5, 2, 3),
            (8, 16, 3, 2),
            (16, 16, 3, 3),
            (4, 13, 2, 2),
            (10, 10, 2, 2),
]

# Fire hazard zones: moved closer to useful travel routes
        self.hazards = [
            (5.0, 7.0, 2.0),
            (15.5, 9.0, 2.2),
            (11.0, 5.0, 1.8),
]

        self.drone_x, self.drone_y = self.start

        self.mission_complete = False
        self.destroyed = False
        self.steps = 0
        self.last_event = "Ready for mission"

    def load_assets(self):
        for name in [
            "drone", "building", "house",
            "rock", "trees", "fire"
        ]:
            path = os.path.join(
                self.assets_dir, f"{name}.png"
            )

            if os.path.isfile(path):
                try:
                    self.images[name] = (
                        pygame.image.load(path).convert_alpha()
                    )
                except pygame.error:
                    print(f"Could not load asset: {path}")

    def reset(self):
        self.drone_x, self.drone_y = self.start
        self.mission_complete = False
        self.destroyed = False
        self.steps = 0
        self.last_event = "Mission reset"

        for target in self.targets:
            target["found"] = False

        return self.drone_x, self.drone_y

    # Drone position is its top-left logical grid cell.
    # Collision footprint is 2 x 2 cells.
    def can_move_to(self, x, y):
        if (
            x < 0 or y < 0
            or x + 1 >= GRID_SIZE
            or y + 1 >= GRID_SIZE
        ):
            return False

        drone_rect = pygame.Rect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            2 * CELL_SIZE,
            2 * CELL_SIZE
        )

        # Only buildings block flight.
        for bx, by, bw, bh in self.buildings:
            building_rect = pygame.Rect(
                bx * CELL_SIZE,
                by * CELL_SIZE,
                bw * CELL_SIZE,
                bh * CELL_SIZE
            )

            if drone_rect.colliderect(building_rect):
                return False

        # Rocks and trees do not block the drone.
        return True

    def hazard_at(self, x, y):
        # Check the drone's center against each circular fire zone.
        cx = x + 1.0
        cy = y + 1.0

        for hx, hy, radius in self.hazards:
            if math.hypot(cx - hx, cy - hy) <= radius:
                return True

        return False

    def check_hazard(self, x=None, y=None):
        if x is None:
            x = self.drone_x

        if y is None:
            y = self.drone_y

        return self.hazard_at(x, y)

    def reached_target(self, x, y):
        # Target coordinates refer to its center cell.
        # The drone center must reach the target cell.
        return (x + 1, y + 1) == self.target

    def move_drone(self, dx, dy):
        if self.mission_complete or self.destroyed:
            return False

        nx = self.drone_x + dx
        ny = self.drone_y + dy

        if not self.can_move_to(nx, ny):
            self.last_event = "Building collision: move blocked"
            return False

        self.steps += 1
        self.drone_x, self.drone_y = nx, ny

        if self.check_hazard():
            self.last_event = "Warning: entered fire zone"
        else:
            self.last_event = "Drone moved"

        if self.reached_target(nx, ny):
            self.mission_complete = True
            self.targets[0]["found"] = True
            self.last_event = "Target reached"

        return True

    def draw_image(self, screen, name, rect, fallback_color):
        image = self.images.get(name)

        if image is None:
            pygame.draw.rect(
                screen, fallback_color, rect,
                border_radius=5
            )
            return

        image = pygame.transform.smoothscale(
            image, (rect.width, rect.height)
        )

        # No extra border is drawn around the sprite.
        screen.blit(image, rect.topleft)

    def draw(self, screen=None, q_agent=None):
        if screen is None:
            screen = self.screen

        if screen is None:
            return

        screen.fill(PANEL_COLOR)

        # Base terrain.
        pygame.draw.rect(
            screen, BACKGROUND,
            (0, 0, MAP_WIDTH, MAP_HEIGHT)
        )

        # Very subtle grid: thin and low contrast.
        grid_surface = pygame.Surface(
            (MAP_WIDTH, MAP_HEIGHT),
            pygame.SRCALPHA
        )

        grid_color = (*GRID_COLOR, 65)

        for x in range(GRID_SIZE + 1):
            pygame.draw.line(
                grid_surface, grid_color,
                (x * CELL_SIZE, 0),
                (x * CELL_SIZE, MAP_HEIGHT), 1
            )

        for y in range(GRID_SIZE + 1):
            pygame.draw.line(
                grid_surface, grid_color,
                (0, y * CELL_SIZE),
                (MAP_WIDTH, y * CELL_SIZE), 1
            )

        screen.blit(grid_surface, (0, 0))

        # Soft circular fire-zone glow.
        glow = pygame.Surface(
            (MAP_WIDTH, MAP_HEIGHT),
            pygame.SRCALPHA
        )

        for hx, hy, radius in self.hazards:
            center = (
                int(hx * CELL_SIZE),
                int(hy * CELL_SIZE)
            )

            pygame.draw.circle(
                glow, (255, 70, 40, 35),
                center, int(radius * CELL_SIZE)
            )

            pygame.draw.circle(
                glow, (255, 70, 40, 40),
                center, int(radius * CELL_SIZE * 0.55)
            )

        screen.blit(glow, (0, 0))

        # Fire symbols.
        for hx, hy, radius in self.hazards:
            center = (
                int(hx * CELL_SIZE),
                int(hy * CELL_SIZE)
            )

            if "fire" in self.images:
                size = int(CELL_SIZE * 0.8)
                fire_image = pygame.transform.smoothscale(
                    self.images["fire"], (size, size)
                )

                screen.blit(
                    fire_image,
                    (
                        center[0] - size // 2,
                        center[1] - size // 2
                    )
                )
            else:
                pygame.draw.circle(
                    screen, (195, 54, 35),
                    center, 10
                )

        # Trees are decorative and can be flown over.
        for tx, ty, tw, th in self.tree_clusters:
            tree_positions = [
                (0.25, 0.25),
                (0.65, 0.3),
                (0.45, 0.65)
            ]

            for ox, oy in tree_positions:
                px = int((tx + ox * tw) * CELL_SIZE)
                py = int((ty + oy * th) * CELL_SIZE)

                if "trees" in self.images:
                    size = int(CELL_SIZE * 1.2)

                    tree_image = pygame.transform.smoothscale(
                        self.images["trees"], (size, size)
                    )

                    screen.blit(
                        tree_image,
                        (px - size // 2, py - size // 2)
                    )
                else:
                    pygame.draw.circle(
                        screen, (50, 100, 49),
                        (px, py), 12
                    )

        # Buildings: multi-cell sprites without added outlines.
        for bx, by, bw, bh in self.buildings:
            rect = pygame.Rect(
                bx * CELL_SIZE,
                by * CELL_SIZE,
                bw * CELL_SIZE,
                bh * CELL_SIZE
            )

            name = (
                "building"
                if "building" in self.images
                else "house"
            )

            self.draw_image(
                screen, name, rect, (65, 76, 83)
            )

        # Rock piles are visual only; drone can fly over them.
        for rx, ry, rw, rh in self.rock_piles:
            rect = pygame.Rect(
                rx * CELL_SIZE,
                ry * CELL_SIZE,
                rw * CELL_SIZE,
                rh * CELL_SIZE
            )

            self.draw_image(
                screen, "rock", rect, (83, 88, 91)
            )

        # Learned path.
        if q_agent is not None and getattr(
            q_agent, "trained", False
        ):
            self.draw_policy_path(screen, q_agent)

        # Target: one marker at the top-right.
        tx, ty = self.target
        center = (
            (tx + 0.5) * CELL_SIZE,
            (ty + 0.5) * CELL_SIZE
        )
        center = (int(center[0]), int(center[1]))

        pygame.draw.circle(
            screen, (218, 55, 38), center, 17
        )

        pygame.draw.line(
            screen, (255, 255, 255),
            (center[0] - 9, center[1]),
            (center[0] + 9, center[1]), 4
        )

        pygame.draw.line(
            screen, (255, 255, 255),
            (center[0], center[1] - 9),
            (center[0], center[1] + 9), 4
        )

        font = pygame.font.SysFont(
            "arial", 16, bold=True
        )

        screen.blit(
            font.render("TARGET", True, (255, 255, 255)),
            (center[0] - 30, center[1] - 35)
        )

        # Start label only; no decorative circle.
        sx, sy = self.start

        screen.blit(
            font.render("START", True, (255, 255, 255)),
            (sx * CELL_SIZE, (sy + 2) * CELL_SIZE)
        )

        # Drone sprite: larger than one cell, without a border.
        drone_rect = pygame.Rect(
            self.drone_x * CELL_SIZE,
            self.drone_y * CELL_SIZE,
            int(CELL_SIZE * 1.5),
            int(CELL_SIZE * 1.5)
        )

        self.draw_image(
            screen, "drone", drone_rect,
            (30, 145, 210)
        )

        # Information panel.
        panel_x = MAP_WIDTH

        pygame.draw.rect(
            screen, PANEL_COLOR,
            (panel_x, 0, PANEL_WIDTH, HEIGHT)
        )

        title_font = pygame.font.SysFont(
            "arial", 21, bold=True
        )

        body_font = pygame.font.SysFont(
            "arial", 15
        )

        screen.blit(
            title_font.render(
                "Emergency Drone RL",
                True, TEXT_COLOR
            ),
            (panel_x + 16, 20)
        )

        status = (
            "TARGET REACHED"
            if self.mission_complete
            else "IN MISSION"
        )

        lines = [
            f"Grid: {GRID_SIZE} x {GRID_SIZE}",
            f"Drone: ({self.drone_x}, {self.drone_y})",
            f"Target: {self.target}",
            f"Steps: {self.steps}",
            f"Status: {status}",
            "",
            "CONTROLS",
            "Arrow keys: move drone",
            "T: train Q-learning",
            "A: run learned policy",
            "R: reset mission",
            "S: save results",
            "ESC: quit",
            "",
            "REWARDS",
            "+100: reach target",
            "-10: building collision",
            "-1: each step",
            "-5: enter fire zone",
            "",
            "LATEST EVENT",
            self.last_event[:32],
        ]

        y = 62

        for line in lines:
            color = (
                (126, 199, 239)
                if line in (
                    "CONTROLS", "REWARDS", "LATEST EVENT"
                )
                else TEXT_COLOR
            )

            screen.blit(
                body_font.render(line, True, color),
                (panel_x + 16, y)
            )

            y += 24

        pygame.display.set_caption(
            "2D Emergency Response Simulation - Q-learning Drone"
        )

    def draw_policy_path(self, screen, agent):
        x, y = self.start
        visited = set()

        points = [
            (
                int((x + 1) * CELL_SIZE),
                int((y + 1) * CELL_SIZE)
            )
        ]

        for _ in range(GRID_SIZE * GRID_SIZE):
            if self.reached_target(x, y):
                break

            if (x, y) in visited:
                break

            visited.add((x, y))

            action = int(agent.q_table[y, x].argmax())
            dx, dy = agent.actions[action]

            nx, ny = x + dx, y + dy

            if not self.can_move_to(nx, ny):
                break

            x, y = nx, ny

            points.append(
                (
                    int((x + 1) * CELL_SIZE),
                    int((y + 1) * CELL_SIZE)
                )
            )

        if len(points) > 1:
            for i in range(len(points) - 1):
                pygame.draw.line(
                    screen, (30, 100, 230),
                    points[i], points[i + 1], 3
                )
