import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# ==============================
# Multi-Agent Delivery Environment
# ==============================
class DroneDeliveryEnv:
    def __init__(self, grid_size=10, num_agents=2, num_packages=3, max_steps=200):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_packages = num_packages
        self.max_steps = max_steps

        # Agent positions
        self.agent_pos = None
        self.agent_has = None  # Which package the agent carries (-1 = none)

        # Packages: (pickup_pos, delivery_pos, picked, delivered)
        self.packages = None

        self.step_count = 0

        # Rendering
        self.fig = None
        self.ax = None

    # -----------------------------
    # Reset environment
    # -----------------------------
    def reset(self):
        self.step_count = 0

        # Place agents randomly
        self.agent_pos = [self._random_empty_cell([]) for _ in range(self.num_agents)]
        self.agent_has = [-1 for _ in range(self.num_agents)]

        # Generate packages
        self.packages = []
        used = self.agent_pos.copy()
        for _ in range(self.num_packages):
            pick = self._random_empty_cell(used)
            used.append(pick)
            drop = self._random_empty_cell(used)
            used.append(drop)
            self.packages.append([pick, drop, False, False])

        return self._get_obs()

    # -----------------------------
    # Random empty cell generator
    # -----------------------------
    def _random_empty_cell(self, occupied):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in occupied:
                return pos

    # -----------------------------
    # Observation
    # -----------------------------
    def _get_obs(self):
        """Returns list[agent] = dict(state)"""
        obs = []
        for i in range(self.num_agents):
            agent_state = {
                "agent_pos": self.agent_pos[i],
                "carrying": self.agent_has[i],
                "packages": self.packages,
                "other_agent": self.agent_pos[1-i],
            }
            obs.append(agent_state)
        return obs

    # -----------------------------
    # Step function
    # actions: list of integer actions for each agent
    # Action mapping:
    # 0 stay
    # 1 up
    # 2 down
    # 3 left
    # 4 right
    # 5 pick
    # 6 deliver
    # -----------------------------
    def step(self, actions):
        rewards = [0, 0]
        done = False

        # Move agents
        for i in range(self.num_agents):
            a = actions[i]
            x, y = self.agent_pos[i]

            if a == 1:  # up
                x = max(0, x - 1)
            elif a == 2:  # down
                x = min(self.grid_size - 1, x + 1)
            elif a == 3:  # left
                y = max(0, y - 1)
            elif a == 4:  # right
                y = min(self.grid_size - 1, y + 1)

            self.agent_pos[i] = (x, y)

        # Collision penalty
        if self.agent_pos[0] == self.agent_pos[1]:
            rewards[0] -= 5
            rewards[1] -= 5

        # Pick / Deliver actions
        for i in range(self.num_agents):
            pos = self.agent_pos[i]
            carry = self.agent_has[i]
            action = actions[i]

            if carry == -1:
                # 荷物を持っていない時：最も近い「未回収」の荷物への距離を報酬にする
                undelivered_pkgs = [p for p in self.packages if not p[2]]
                if undelivered_pkgs:
                    dists = [np.abs(pos[0]-p[0][0]) + np.abs(pos[1]-p[0][1]) for p in undelivered_pkgs]
                    # 距離が近いほど報酬（最大0.1程度になるよう調整）
                    rewards[i] += 0.01 * (10 - min(dists)) 
            else:
                # 荷物を持っている時：その荷物の目的地への距離を報酬にする
                drop_pos = self.packages[carry][1]
                dist_to_drop = np.abs(pos[0]-drop_pos[0]) + np.abs(pos[1]-drop_pos[1])
                rewards[i] += 0.01 * (10 - dist_to_drop)
            # pick
            if action == 5 and carry == -1:
                for pid, pack in enumerate(self.packages):
                    pick, drop, picked, delivered = pack
                    if not picked and pos == pick:
                        self.agent_has[i] = pid
                        pack[2] = True  # mark picked
                        rewards[i] += 1
                        break

            # deliver
            if action == 6 and carry != -1:
                pid = carry
                pick, drop, picked, delivered = self.packages[pid]
                if pos == drop and picked and not delivered:
                    self.agent_has[i] = -1
                    self.packages[pid][3] = True
                    rewards[i] += 10

        # Check if all delivered
        if all(p[3] for p in self.packages):
            done = True
            rewards = [r + 5 for r in rewards]

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return self._get_obs(), rewards, done, {}

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()

        # Grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.ax.add_patch(patches.Rectangle((y, self.grid_size-1-x),
                                                    1, 1,
                                                    fill=False, edgecolor='gray'))

        # Draw packages
        for pid, (pick, drop, picked, delivered) in enumerate(self.packages):
            px, py = pick
            dx, dy = drop

            px = (py, self.grid_size - 1 - px)
            dx = (dy, self.grid_size - 1 - dx)

            # pickup point
            if not picked:
                self.ax.add_patch(patches.Circle(px, 0.3, color="red"))
            # carry state (do nothing)
            # delivery point
            if not delivered:
                self.ax.add_patch(patches.Circle(dx, 0.3, color="green"))

        # Draw agents
        colors = ["blue", "orange"]
        for i in range(self.num_agents):
            x, y = self.agent_pos[i]
            cx, cy = y, self.grid_size - 1 - x
            self.ax.add_patch(patches.Rectangle((cx, cy), 1, 1,
                                                color=colors[i], alpha=0.8))

            if self.agent_has[i] != -1:
                self.ax.text(cx+0.3, cy+0.3, "P", color="white", fontsize=12)

        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")
        plt.pause(0.01)

