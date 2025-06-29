class Vehicle:
    def __init__(self, position: tuple[float, float]):
        self.position: tuple[float, float] = position
        self.trajectory: list[tuple[float, float]] = [position]

    def move(self, vector: tuple[float, float]) -> None:
        position_x = self.position[0] + vector[0]
        position_y = self.position[1] + vector[1]
        self.position = (position_x, position_y)
        self.trajectory.append(self.position)
