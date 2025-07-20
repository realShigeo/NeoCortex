import torch
from torch import Tensor, nn


def simulate(
    model: nn.Sequential,
    max_length: float,
    vehicle_velocity: float,
    distance_to_travel: float,
):
    norm_distance_to_travel: float = distance_to_travel / max_length

    model.eval()
    with torch.no_grad():
        model_input: Tensor = torch.tensor(
            [norm_distance_to_travel], dtype=torch.float32
        )

        motor_on_time: Tensor = model(model_input)

    distance_traveled: float = vehicle_velocity * motor_on_time.item()
    percent_error: float = (
        abs((distance_traveled - distance_to_travel) / distance_to_travel)
        * 100
    )

    print(f"Analytical Distance to Travel: {distance_to_travel} meters")
    print(f"Actual Distance Traveled: {distance_traveled} meters")
    print(f"Percent Error: {percent_error:.3f}%")
