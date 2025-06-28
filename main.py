import matplotlib.pyplot as plt

from simulator.vehicle import Vehicle


def main() -> None:
    # Define the environment's initial state
    vehicle: Vehicle = Vehicle(position=(0, 0))

    # Execute operations upon the environment
    vehicle.move("right")
    vehicle.move("up")
    vehicle.move("up")
    vehicle.move("right")
    vehicle.move("right")
    vehicle.move("down")
    vehicle.move("left")

    # Output the results of the environment
    xs, ys = zip(*vehicle.trajectory)
    plt.plot(xs, ys)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
