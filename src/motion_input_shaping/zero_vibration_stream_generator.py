"""
Contains the ZeroVibrationShaper class for re-use in other code.

Run the file directly to test the class functionality.
"""

# pylint: disable=invalid-name
# Allow short variable names

import math
from enum import Enum
from dataclasses import dataclass
import numpy as np


class ShaperType(Enum):
    """Enumeration for different input shaper types."""

    ZV = 1
    ZVD = 2
    ZVDD = 3


@dataclass
class StreamSegment:
    """A class that contains information for a single segment of the trajectory."""

    position: float
    speed_limit: float
    accel: float
    duration: float


def trapezoidal_motion_generator(
    distance: float, acceleration: float, deceleration: float, max_speed_limit: float
) -> list[list[float]]:
    """
    Produce acceleration profile for basic trapezoidal motion.

    Returns time and acceleration at points where acceleration changes
    All acceleration changes are step changes for trapezoidal motion.

    :param distance: The trajectory distance.
    :param acceleration: The trajectory acceleration.
    :param deceleration: The trajectory deceleration.
    :param max_speed_limit: Maximum trajectory speed in the output motion.
    """
    direction = np.sign(distance)

    acceleration_distance = max_speed_limit**2 / (2 * acceleration)
    deceleration_distance = max_speed_limit**2 / (2 * deceleration)
    max_speed_distance = abs(distance) - acceleration_distance - deceleration_distance

    if max_speed_distance <= 0:
        # Max speed is not reached.
        max_speed = np.sqrt(abs(distance) / (1 / (2 * acceleration) + 1 / (2 * deceleration)))
        acceleration_duration = max_speed / acceleration
        deceleration_duration = max_speed / deceleration

        acceleration_endtime = acceleration_duration
        deceleration_endtime = acceleration_duration + deceleration_duration

        return [
            [0, acceleration_endtime, deceleration_endtime],
            [acceleration * direction, deceleration * (-1 * direction), 0],
        ]

    # Trapezoidal Motion
    max_speed = max_speed_limit

    acceleration_duration = max_speed_limit / acceleration
    max_speed_duration = max_speed_distance / max_speed_limit
    deceleration_duration = max_speed_limit / deceleration

    acceleration_endtime = acceleration_duration
    max_speed_endtime = acceleration_endtime + max_speed_duration
    deceleration_endtime = max_speed_endtime + deceleration_duration

    return [
        [0, acceleration_endtime, max_speed_endtime, deceleration_endtime],
        [acceleration * direction, 0, deceleration * (-1 * direction), 0],
    ]


def calculate_acceleration_convolution(
    impulse_times: list[float],
    impulses: list[float],
    unshaped_time: list[float],
    unshaped_acceleration: list[float],
) -> list[list[float]]:
    """
    Perform the shaping by computing convolution of acceleration with the shaper impulses.

    This algorithm computes the result of the convolution without actually performing a convolution
    on a full timeseries dataset. This reduces the amount of computation and number of points
    generated and is also more accurate since it is not constrained by fixed timestep size.
    This algorithm only works if the acceleration changes are steps changes.
    Returns shaped segment times and accelerations.

    :param impulse_times: List of shaper impulse times
    :param impulses: List of shaper impulse magnitudes
    :param unshaped_time: List of times when acceleration changes in unshaped trajectory
    :param unshaped_acceleration: Acceleration of each segment of the unshaped trajectory
    """
    unshaped_accel_changes = np.diff(
        np.concatenate([np.array([0]), np.array(unshaped_acceleration)])
    )  # append a 0 to start of list of accelerations and take diff to get changes
    num_rows = len(unshaped_acceleration)

    # Create extra points based on number of impulses
    shaped_time = np.zeros(len(unshaped_acceleration) * len(impulses))
    accel_changes = np.zeros(len(unshaped_acceleration) * len(impulses))
    # for each impulse create a copy of the acceleration change delayed and scaled by the
    # impulse time and magnitude
    for n, impulse in enumerate(impulses):
        shaped_time[n * num_rows : (num_rows + n * num_rows)] = (
            np.array([unshaped_time]) + impulse_times[n]
        )
        accel_changes[n * num_rows : (num_rows + n * num_rows)] = (
            np.array([unshaped_accel_changes]) * impulse
        )

    # sort acceleration changes by time
    sort_index = shaped_time.argsort()
    shaped_time = shaped_time[sort_index]
    accel_changes = accel_changes[sort_index]

    # Final trajectory acceleration is cumulative sum of acceleration steps which gives the
    # superposition of the contribution from each impulse and is equivalent to the convolution
    shaped_acceleration = np.cumsum(accel_changes)  # Acceleration

    return [shaped_time.tolist(), shaped_acceleration.tolist()]


def create_stream_trajectory(
    trajectory_time: list[float], trajectory_acceleration: list[float]
) -> list[StreamSegment]:
    """
    Compute information needed to execute trajectory through streams.

    Returns list of StreamSegment objects.
    The final acceleration must be 0.

    :param trajectory_time: Start time of each constant acceleration segment
    :param trajectory_acceleration: Acceleration of each segment
    """
    stream_segments = []
    # trajectory is one row less than list of accelerations since first row would be the
    # initial position (zeros)
    previous_position = 0.0
    previous_velocity = 0.0
    for n in range(0, len(trajectory_acceleration) - 1):
        # Calculate position and velocity at end of each segment using equations for constant
        # acceleration since acceleration changes are steps.
        current_accel = trajectory_acceleration[n]
        dt = trajectory_time[n + 1] - trajectory_time[n]  # dt
        current_velocity = previous_velocity + trajectory_acceleration[n] * dt  # velocity
        current_position = (
            previous_position + (current_velocity + previous_velocity) / 2 * dt
        )  # position

        stream_segments.append(
            StreamSegment(
                current_position,
                max([abs(current_velocity), abs(previous_velocity)]),
                abs(current_accel),
                dt,
            )
        )

        # Record current values and previous positions for next step
        previous_position = current_position
        previous_velocity = current_velocity

    return stream_segments


class ZeroVibrationStreamGenerator:
    """A class for creating stream motion with zero vibration input shaping theory."""

    # pylint: disable=too-many-instance-attributes
    # Ignore number of attributes warning because calling property setters in init is being counted
    # as additional attributes

    def __init__(
        self,
        resonant_frequency: float,
        damping_ratio: float,
        shaper_type: ShaperType = ShaperType.ZV,
    ) -> None:
        """
        Initialize the class.

        :param resonant_frequency: The target vibration resonant frequency in Hz.
        :param damping_ratio: The target vibration damping ratio.
        :param damping_ratio: Minimum timestep in pvt sequence in seconds.
        """
        self.resonant_frequency = resonant_frequency
        self.damping_ratio = damping_ratio
        self.shaper_type = shaper_type

        # Initialize impulses as a single impulse with no delay for no shaping
        self._impulse_times = [0.0]
        self._impulses = [1.0]
        self._impulses_updated = False
        self._update_shaper_impulses()  # Update impulses

    @property
    def resonant_frequency(self) -> float:
        """Get the target resonant frequency for input shaping in Hz."""
        return self._resonant_frequency

    @resonant_frequency.setter
    def resonant_frequency(self, value: float) -> None:
        """Set the target resonant frequency for input shaping in Hz."""
        if value <= 0:
            raise ValueError(f"Invalid resonant frequency: {value}. Value must be greater than 0.")
        self._resonant_frequency = value
        self._impulses_updated = False

    @property
    def resonant_period(self) -> float:
        """Get the target resonant period for input shaping in s."""
        return 1.0 / self.resonant_frequency

    @property
    def damping_ratio(self) -> float:
        """Get the target damping ratio for input shaping."""
        return self._damping_ratio

    @damping_ratio.setter
    def damping_ratio(self, value: float) -> None:
        """Set the target damping ratio for input shaping."""
        if value < 0:
            raise ValueError(
                f"Invalid damping ratio: {value}. Value must be greater than or equal to 0."
            )
        self._damping_ratio = value
        self._impulses_updated = False

    @property
    def shaper_type(self) -> ShaperType:
        """Get input shaper type."""
        return self._shaper_type

    @shaper_type.setter
    def shaper_type(self, value: ShaperType) -> None:
        """Set input shaper type."""
        self._shaper_type = value
        self._impulses_updated = False

    def get_impulses(self) -> list[float]:
        """Get shaper impulse magnitudes."""
        if self._impulses_updated is False:
            self._update_shaper_impulses()
        return self._impulses

    def get_impulse_times(self) -> list[float]:
        """Get shaper impulse times."""
        if self._impulses_updated is False:
            self._update_shaper_impulses()
        return self._impulse_times

    def _update_shaper_impulses(self) -> None:
        """
        Calculate times and unitless magnitude of impulses to perform the input shaping.

        The sum of all impulses should total to 1 to maintain the same final state.
        """
        k = math.exp(
            (-1 * math.pi * self.damping_ratio) / math.sqrt(1 - self.damping_ratio**2)
        )  # Decay factor

        match self.shaper_type:
            case ShaperType.ZV:
                a0 = 1 / (1 + k)
                a1 = k / (1 + k)
                t0 = 0
                t1 = self.resonant_period / 2
                self._impulses = [a0, a1]
                self._impulse_times = [t0, t1]
            case ShaperType.ZVD:
                a0 = 1 / (1 + 2 * k + k**2)
                a1 = 2 * k / (1 + 2 * k + k**2)
                a2 = k**2 / (1 + 2 * k + k**2)
                t0 = 0
                t1 = self.resonant_period / 2
                t2 = self.resonant_period
                self._impulses = [a0, a1, a2]
                self._impulse_times = [t0, t1, t2]
            case ShaperType.ZVDD:
                a0 = 1 / (1 + 3 * k + 3 * k**2 + k**3)
                a1 = 3 * k / (1 + 3 * k + 3 * k**2 + k**3)
                a2 = 3 * k**2 / (1 + 3 * k + 3 * k**2 + k**3)
                a3 = k**3 / (1 + 3 * k + 3 * k**2 + k**3)
                t0 = 0
                t1 = self.resonant_period / 2
                t2 = self.resonant_period
                t3 = self.resonant_period * 3 / 2
                self._impulses = [a0, a1, a2, a3]
                self._impulse_times = [t0, t1, t2, t3]
            case _:
                raise ValueError(f"Shaper type {self.shaper_type} is not valid.")
        self._impulses_updated = True

    def shape_trapezoidal_motion(
        self, distance: float, acceleration: float, deceleration: float, max_speed_limit: float
    ) -> list[StreamSegment]:
        """
        Create stream points for zero vibration trapezoidal motion.

        All distance, speed, and accel units must be consistent.

        :param distance: The trajectory distance.
        :param acceleration: The trajectory acceleration.
        :param deceleration: The trajectory deceleration.
        :param max_speed_limit: An optional limit to place on maximum trajectory speed in the
        output motion.
        """
        # Get time and magnitude of the impulses used for shaping
        impulses = self.get_impulses()
        impulse_times = self.get_impulse_times()

        (unshaped_time, unshaped_acceleration) = trapezoidal_motion_generator(
            distance,
            acceleration,
            deceleration,
            max_speed_limit,
        )

        (shaped_time, shaped_acceleration) = calculate_acceleration_convolution(
            impulse_times, impulses, unshaped_time, unshaped_acceleration
        )

        stream_segments = create_stream_trajectory(shaped_time, shaped_acceleration)

        # make sure end point position is exactly on target
        stream_segments[-1].position = distance

        return stream_segments


# Example code for using the class.
if __name__ == "__main__":
    shaper = ZeroVibrationStreamGenerator(4.64, 0.04, ShaperType.ZV)

    DIST = 600
    ACCEL = 2100
    MAX_SPEED = 1000

    trajectory_points = shaper.shape_trapezoidal_motion(DIST, ACCEL, ACCEL, MAX_SPEED)
    print("Accel, Position, Max Speed, Time")
    for point in trajectory_points:
        print(*[point.accel, point.position, point.speed_limit, point.duration], sep=", ")

    print("")
    print(
        f"Shaped Move: "
        f"Max Speed: {max(np.abs(point.speed_limit) for point in trajectory_points):.2f}, "
        f"Total Time: {sum((point.duration for point in trajectory_points)):.2f}, "
    )
