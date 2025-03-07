"""
Contains the ZeroVibrationStreamGenerator2D class for re-use in other code.

Run the file directly to test the class functionality.
"""

import math
from enum import Enum
from dataclasses import dataclass
import numpy as np
from plant import Plant
from zero_vibration_stream_generator import ZeroVibrationStreamGenerator, ShaperType, StreamSegment, trapezoidal_motion_generator, calculate_acceleration_convolution, AccelPoint


@dataclass
class StreamSegment2D:
    """A class that contains information for a single segment of the stream trajectory."""

    x_position: float
    y_position: float
    speed_limit: float
    accel: float
    duration: float


@dataclass
class AccelPoint2D:
    """Acceleration points used to define trajectories in 2D."""

    time: float
    x_acceleration: float
    y_acceleration: float

@dataclass
class Impulse2D:
    """Combined 2D impulses."""

    impulse_times: list[float]
    x_impulses: list[float]
    y_impulses: list[float]

def create_stream_trajectory_2d(x_trajectory: list[AccelPoint], y_trajectory: list[AccelPoint]
                             ) -> list[StreamSegment2D]:
    """
    Compute information needed to execute trajectory through streams.

    Returns list of StreamSegment objects.
    The final acceleration must be 0.

    :param x_trajectory: List of acceleration points for x-axis to create trajectory from
    :param y_trajectory: List of acceleration points for y-axis to create trajectory from
    """
    trajectory_time = [x.time for x in x_trajectory]
    trajectory_x_acceleration = [x.acceleration for x in x_trajectory]
    trajectory_y_acceleration = [x.acceleration for x in y_trajectory]

    stream_segments = []
    # trajectory is one row less than list of accelerations since first row would be the
    # initial position (zeros)
    previous_x_position = 0.0
    previous_x_velocity = 0.0
    previous_y_position = 0.0
    previous_y_velocity = 0.0
    previous_velocity = 0.0
    for n in range(0, len(trajectory_time) - 1):
        # Calculate position and velocity at end of each segment using equations for constant
        # acceleration since acceleration changes are steps.
        dt = trajectory_time[n + 1] - trajectory_time[n]  # dt

        current_x_accel = trajectory_x_acceleration[n]
        current_x_velocity = previous_x_velocity + trajectory_x_acceleration[n] * dt  # velocity
        current_x_position = (
            previous_x_position + (current_x_velocity + previous_x_velocity) / 2 * dt
        )  # position

        current_y_accel = trajectory_y_acceleration[n]
        current_y_velocity = previous_y_velocity + trajectory_y_acceleration[n] * dt  # velocity
        current_y_position = (
            previous_y_position + (current_y_velocity + previous_y_velocity) / 2 * dt
        )  # position

        current_velocity = math.sqrt(current_x_velocity ** 2 + current_y_velocity ** 2)
        current_accel = math.sqrt(current_x_accel ** 2 + current_y_accel ** 2)

        stream_segments.append(
            StreamSegment2D(
                current_x_position,
                current_y_position,
                max([abs(current_velocity), abs(previous_velocity)]),
                abs(current_accel),
                dt,
            )
        )

        # Record current values and previous positions for next step
        previous_velocity = current_velocity
        previous_x_position = current_x_position
        previous_x_velocity = current_x_velocity
        previous_y_position = current_y_position
        previous_y_velocity = current_y_velocity

    return stream_segments

class ZeroVibrationStreamGenerator2D:
    """A class for creating stream motion with zero vibration input shaping theory."""

    def __init__(self, x_plant: Plant, y_plant: Plant, shaper_type: ShaperType = ShaperType.ZV
                 ) -> None:
        """
        Initialize the class.

        :param x_plant: The Plant instance defining the system that the shaper is targeting.
        :param y_plant: The Plant instance defining the system that the shaper is targeting.
        :param shaper_type: Type of input shaper to use to generate impulses.
        """

        self.x_shaper = ZeroVibrationStreamGenerator(x_plant, shaper_type)
        self.y_shaper = ZeroVibrationStreamGenerator(y_plant, shaper_type)

    def get_2D_impulses(self) -> Impulse2D:
        x_impulses = self.x_shaper.get_impulse_amplitudes()
        x_impulse_times = self.x_shaper.get_impulse_times()
        y_impulses = self.y_shaper.get_impulse_amplitudes()
        y_impulse_times = self.y_shaper.get_impulse_times()

        for t in y_impulse_times:
            if not any(time == t for time in x_impulse_times):
                x_impulse_times.append(t)
                x_impulses.append(0)

        for t in x_impulse_times:
            if not any(time == t for time in y_impulse_times):
                y_impulse_times.append(t)
                y_impulses.append(0)

        x_impulse_times, x_impulses = zip(*sorted(zip(x_impulse_times, x_impulses)))
        y_impulse_times, y_impulses = zip(*sorted(zip(y_impulse_times, y_impulses)))

        return Impulse2D(x_impulse_times, x_impulses, y_impulses)

    def shape_trapezoidal_motion(
        self, x_distance: float, y_distance: float, acceleration: float, deceleration: float, max_speed_limit: float
    ) -> list[StreamSegment2D]:
        """
        Create stream points for zero vibration trapezoidal motion.

        All distance, speed, and accel units must be consistent.

        :param x_distance: The trajectory distance in the x direction.
        :param y_distance: The trajectory distance in the y direction.
        :param acceleration: The trajectory acceleration.
        :param deceleration: The trajectory deceleration.
        :param max_speed_limit: An optional limit to place on maximum trajectory speed in the
        output motion.
        """
        # Get time and magnitude of the impulses used for shaping
        impulses = self.get_2D_impulses()

        # Calculate magnitude of x and y component relative to total
        total_distance = math.sqrt(x_distance ** 2 + y_distance ** 2)
        x_ratio = x_distance / total_distance  # cos(theta)
        y_ratio = y_distance / total_distance  # sin(theta)

        # Generate trapezoidal profile for combined move
        unshaped_trajectory = trapezoidal_motion_generator(
            total_distance,
            acceleration,
            deceleration,
            max_speed_limit,
        )

        # Split into X and Y components
        unshaped_x_trajectory = []
        unshaped_y_trajectory = []
        for trajectory_point in unshaped_trajectory:
            unshaped_x_trajectory.append(
                AccelPoint(trajectory_point.time, trajectory_point.acceleration * x_ratio))
            unshaped_y_trajectory.append(
                AccelPoint(trajectory_point.time, trajectory_point.acceleration * y_ratio))

        # Apply shaper
        shaped_x_trajectory = calculate_acceleration_convolution(
            impulses.impulse_times, impulses.x_impulses, unshaped_x_trajectory
        )
        shaped_y_trajectory = calculate_acceleration_convolution(
            impulses.impulse_times, impulses.y_impulses, unshaped_y_trajectory
        )

        # Create stream segments
        stream_segments = create_stream_trajectory_2d(shaped_x_trajectory, shaped_y_trajectory)

        # make sure end point position is exactly on target
        stream_segments[-1].x_position = x_distance
        stream_segments[-1].y_position = y_distance

        return stream_segments


# Example code for using the class.
if __name__ == "__main__":
    x_plant_var = Plant(3, 0.04)
    y_plant_var = Plant(8, 0.04)
    shaper = ZeroVibrationStreamGenerator2D(x_plant_var, y_plant_var, ShaperType.ZV)

    shaper.get_2D_impulses()

    X_DIST = 600
    Y_DIST = 300
    ACCEL = 2100
    MAX_SPEED = 1000

    trajectory_points = shaper.shape_trapezoidal_motion(X_DIST, Y_DIST, ACCEL, ACCEL, MAX_SPEED)
    print("Tangential Accel, X Position, Y Position, Max Speed, Time")
    for point in trajectory_points:
        print(*[point.accel, point.x_position, point.y_position, point.speed_limit, point.duration],
              sep=", ")

    print("")
    print(
        f"Shaped Move: "
        f"Max Speed: {max(np.abs(point.speed_limit) for point in trajectory_points):.2f}, "
        f"Total Time: {sum((point.duration for point in trajectory_points)):.2f}, "
    )
