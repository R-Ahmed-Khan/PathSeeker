import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import ast
import os

class Animate():
    
    def __init__(self):
        base_path = os.getcwd()
        artifacts_path = os.path.join(base_path, 'artifacts')
        self.obstacle_file_path = os.path.join(artifacts_path, 'obstacles.csv')
        self.grid_file_path = os.path.join(artifacts_path, 'grid_size.csv')

    def get_obstacles(self):
        data = []
        with open(self.obstacle_file_path, 'r') as file:
            for line in file:
                row = [float(value.replace(',', '')) for value in line.split()]
                data.append(row)
        self.obstacles = [tuple(inner_list) for inner_list in data]

    def get_grid(self):
        with open(self.grid_file_path, 'r') as file:
            for line in file:
                self.grid_size = float(line)

    def animate_trajectory(self, path, target_positions, resolution_Hz=3, duration=None, fig_size=5):
        
        x_path = [p[0] for p in path]
        y_path = [p[1] for p in path]
        theta_path = [p[2] for p in path]

        target_positions= ast.literal_eval(target_positions)
        plt.ioff()

        trajectory = []
        for point in path:
            trajectory.append((point[0], point[1]))

        trajectory = np.array(trajectory)
        if len(trajectory.shape) == 1:
            trajectory = trajectory.reshape(trajectory.size, 1)
        if trajectory.shape == (trajectory.size, 1):
            trajectory = np.stack((trajectory.T[0], np.zeros(trajectory.size))).T
        if duration == None:
            frames = range(trajectory.shape[0])
        else:
            frames = range(int(duration * resolution_Hz))

        fig, ax = plt.subplots()

        ax.set_xlim([-0.5, self.grid_size])
        ax.set_ylim([-0.5, self.grid_size])
        ax.set_aspect("equal")
        ax.grid(True)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')

        # Plot the start position
        ax.scatter(x_path[0], y_path[0], color='b', s=100, label='Start')  # Start position

        arrow_length = 0.5
        ax.scatter([], [], color='r', s=100, label='Obstacle')

        # Plot the obstacles
        for obs in self.obstacles:
            ax.scatter(obs[0], obs[1], color='r', s=100)

        # Create a point for the target position (bolder and larger)
        target_point, = ax.plot([], [], marker="*", color="g", markersize=15, label="Target")

        # Create a line object to show the path
        path_line, = ax.plot([], [], color='b', label='Path Taken')  # Empty line to update

        # Create a point to visualize UGV's movement
        point, = ax.plot([], [], marker="o", color="b", markersize=10)

        ax.legend(loc='upper right', bbox_to_anchor=(1.34, 1))

        arrows = []

        # Updating function, to be repeatedly called by the animation
        def update(val):
            # Obtain UGV point coordinates
            x, y = trajectory[int(val) % trajectory.shape[0]]

            # Update the path line (including previous points)
            path_line.set_data(trajectory[:int(val)+1, 0], trajectory[:int(val)+1, 1])

            # Plot the current point (UGV's movement)
            point.set_data([x], [y])

            # Update the target point (the target position at the current step)
            if val >= len(target_positions):
                current_target = target_positions[-1]
            else:
                current_target = target_positions[int(val) % len(target_positions)]
            target_point.set_data([current_target[0]], [current_target[1]])

            # Update the arrows
            if val < len(theta_path):
                # Remove the previous arrow
                if len(arrows) > 0:
                    arrows[-1].remove()

                # Get the current arrow's direction (dx, dy)
                dx = arrow_length * np.cos(theta_path[int(val)] / 57.3)
                dy = arrow_length * np.sin(theta_path[int(val)] / 57.3)

                # Plot the current arrow at the current position with orientation defined by theta
                arrow = ax.arrow(x_path[int(val)], y_path[int(val)], dx, dy, head_width=0.20, head_length=0.1, fc='red', ec='red')
                arrows.append(arrow)

            return path_line, point, target_point

        # Set up the animation
        ani = FuncAnimation(fig, update, frames=max(len(trajectory), len(target_positions)), interval=1000 / resolution_Hz, blit=True)

        plt.ion()
        return ani

def main_animate(path, target_positions):
    animate = Animate()
    animate.get_obstacles()
    animate.get_grid()
    ani = animate.animate_trajectory(path, target_positions=target_positions, resolution_Hz=1)
    base_path = os.getcwd()
    analysis_folder_path = os.path.join(base_path, 'analysis/simulation.gif')
    ani.save(analysis_folder_path, writer='pillow', fps=1)
    print('The simulation gif has been saved in the analysis folder')
