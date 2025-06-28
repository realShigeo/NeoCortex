class Vehicle:
    def __init__(self, position: tuple[float, float]):
        self.position: tuple[float, float] = position
        self.trajectory: list[tuple[float, float]] = [position]

    def move(self, direction: str) -> None:
        match direction:
            case "up":
                self.position = (self.position[0], self.position[1] + 1)
            case "down":
                self.position = (self.position[0], self.position[1] - 1)
            case "right":
                self.position = (self.position[0] + 1, self.position[1])
            case "left":
                self.position = (self.position[0] - 1, self.position[1])

        self.trajectory.append(self.position)
