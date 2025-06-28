"""
Entry point for simulating and visualizing a 2D vehicle's trajectory.

This module initializes a Vehicle instance, applies a predefined sequence
of movement vectors, and uses matplotlib to plot the resulting path.

Functions:
    main(): Runs the simulation and displays the vehicle's trajectory.
"""

import matplotlib.pyplot as plt

from simulator.vehicle import Vehicle


def main() -> None:
    """
    Simulates a series of movements for a vehicle and plots its trajectory.

    This function initializes a Vehicle at the origin, applies a sequence of
    movement vectors, and then visualizes the path taken using matplotlib.

    Returns:
        None
    """
    # Define the environment's initial state
    vehicle: Vehicle = Vehicle(position=(0, 0))

    # Execute operations upon the environment
    vehicle.move(vector=(1, 0))
    vehicle.move(vector=(0, 1))
    vehicle.move(vector=(0, 1))
    vehicle.move(vector=(1, 0))
    vehicle.move(vector=(1, 0))
    vehicle.move(vector=(0, -1))
    vehicle.move(vector=(-1, 0))

    # Output the results of the environment
    xs, ys = zip(*vehicle.trajectory)
    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
