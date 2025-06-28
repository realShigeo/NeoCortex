"""
Defines the Vehicle class used for simulating movement in 2D space.

This module provides a simple interface for updating a vehicle's position
based on movement vectors and tracking its trajectory over time.

Classes:
    Vehicle: Represents an object that moves in a 2D plane.
"""


class Vehicle:
    """
    A simple 2D vehicle that tracks its movement over time.

    The Vehicle maintains its current position and a history of all past
    positions in a trajectory list. Movement is performed by applying a
    vector to the current position.

    Attributes:
        position (tuple[float, float]): The current (x, y) coordinates of the
                                        vehicle.
        trajectory (list[tuple[float, float]]): A list of all positions the
                                                vehicle has occupied.
    """

    def __init__(self, position: tuple[float, float]):
        self.position: tuple[float, float] = position
        self.trajectory: list[tuple[float, float]] = [position]

    def move(self, vector: tuple[float, float]) -> None:
        """
        Updates the vehicle's position by applying a movement vector and
        records the new position in the trajectory.

        Args:
            vector (tuple[float, float]): The movement vector as (dx, dy),
                                          where dx and dy are the changes in
                                          the x and y coordinates respectively.

        Returns:
            None
        """
        position_x = self.position[0] + vector[0]
        position_y = self.position[1] + vector[1]
        self.position = (position_x, position_y)
        self.trajectory.append(self.position)
