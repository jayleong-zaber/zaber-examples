"""
Contains the ZeroVibrationShaper class for re-use in other code.

Run the file directly to test the class functionality.
"""

# pylint: disable=invalid-name
# Allow short variable names

import math
import numpy as np

class ZeroVibrationTrajectoryGenerator:
    """A class for implementing zero vibration input shaping theory."""

    def __init__(self, resonant_frequency: float, damping_ratio: float, min_timestep: float = 0.001) -> None:
        """
        Initialize the class.

        :param resonant_frequency: The target vibration resonant frequency in Hz.
        :param damping_ratio: The target vibration damping ratio.
        :param damping_ratio: Minimum timestep in pvt sequence in seconds.
        """
        self.resonant_frequency = resonant_frequency
        self.damping_ratio = damping_ratio
        self.min_timestep = min_timestep

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

    @property
    def min_timestep(self) -> float:
        """Get the minimum time step for shaped pvt sequence in seconds."""
        return self._min_timestep

    @min_timestep.setter
    def min_timestep(self, value: float) -> None:
        """Get the minimum time step for shaped pvt sequence in seconds."""
        if value <= 0:
            raise ValueError(f"Invalid minimum timestep: {value}. Value must be greater than 0.")

        self._min_timestep = value

    def get_impulse_amplitudes(self) -> list[float]:
        """Get the unitless magnitude of both impulses to perform the input shaping."""
        k = math.exp(
            (-1 * math.pi * self.damping_ratio) / math.sqrt(1 - self.damping_ratio**2)
        )  # Decay factor at half period

        a1 = 1 / (1 + k)
        a2 = k / (1 + k)

        return [a1, a2]

    def basic_trapezoidal_motion_generator(
        self, distance: float, acceleration: float, deceleration: float, max_speed_limit: float):
        """
                Produce acceleration profile for basic trapezoidal motion
                Returns time and acceleration at points where acceleration changes in nx2 array where column 1 is time
                and column 2 is acceleration
                All acceleration changes are step changes for trapezoidal motion.

                :param distance: The trajectory distance.
                :param acceleration: The trajectory acceleration.
                :param deceleration: The trajectory deceleration.
                :param max_speed_limit: Maximum trajectory speed in the output motion.
        """
        direction = np.sign(distance)

        acceleration_distance = max_speed_limit ** 2 / (2 * acceleration)
        deceleration_distance = max_speed_limit ** 2 / (2 * deceleration)
        max_speed_distance = abs(distance) - acceleration_distance - deceleration_distance

        if max_speed_distance > 0:
            # Trapezoidal Motion
            max_speed = max_speed_limit

            acceleration_duration = max_speed_limit / acceleration
            max_speed_duration = max_speed_distance / max_speed_limit
            deceleration_duration = max_speed_limit / deceleration

            acceleration_endtime = acceleration_duration
            max_speed_endtime = acceleration_endtime + max_speed_duration
            deceleration_endtime = max_speed_endtime + deceleration_duration

            return np.array([[0, acceleration * direction],
                             [acceleration_endtime, 0],
                             [max_speed_endtime, deceleration * (-1 * direction)],
                             [deceleration_endtime, 0]]
                            )
        else:
            # Max speed is not reached.
            max_speed = np.sqrt(abs(distance)/(1/(2*acceleration)+1/(2*deceleration)))
            acceleration_duration = max_speed / acceleration
            deceleration_duration = max_speed / deceleration

            acceleration_endtime = acceleration_duration
            deceleration_endtime = acceleration_duration + deceleration_duration

            return np.array([[0, acceleration * direction],
                             [acceleration_endtime, deceleration * (-1 * direction)],
                             [deceleration_endtime, 0]]
                            )

    def shape_trapezoidal_motion(
        self, distance: float, acceleration: float, deceleration: float, max_speed_limit: float
    ) -> list[list[float]]:
        """
        Create pvt points for zero vibration.

        Impulses are based on target resonant frequency and damping ratio. Sum of impulses is 1.
        The shaped acceleration is the basic acceleration convolved with the impulses.
        This algorithm calculates creates the output without actually performing a convolution on a full timeseries
        dataset which reduces the number of points generated and is also more accurate since it is not constrained by
        fixed timestep size. This algorithm only works if acceleration changes are steps.
        All distance units must be the same.

        :param distance: The trajectory distance.
        :param acceleration: The trajectory acceleration.
        :param deceleration: The trajectory deceleration.
        :param max_speed_limit: An optional limit to place on maximum trajectory speed in the
        output motion.
        """
        # Get time and magnitude of the impulses used for shaping
        impulse_amplitudes = self.get_impulse_amplitudes()
        impulses = np.array([[0, impulse_amplitudes[0]], [self.resonant_period/2, impulse_amplitudes[1]]])

        basic_trajectory_acceleration = (
            self.basic_trapezoidal_motion_generator(distance, acceleration, deceleration, max_speed_limit))
        basic_accel_changes = np.diff(
            np.concatenate([np.array([0]), basic_trajectory_acceleration[:, 1]])
        )  # append a 0 to start of list of accelerations and take diff to get changes
        num_rows = basic_trajectory_acceleration.shape[0]

        accel_changes = np.zeros([basic_trajectory_acceleration.shape[0] * len(impulses), 2])
        # for each impulse create a copy of the acceleration change delayed and scaled by the impulse time and magnitude
        for n in range(len(impulses)):
            accel_changes[n*num_rows:(num_rows+n*num_rows), 0] = basic_trajectory_acceleration[:, 0] + impulses[n, 0]
            accel_changes[n*num_rows:(num_rows+n*num_rows), 1] = basic_accel_changes[:] * impulses[n, 1]

        accel_changes = accel_changes[accel_changes[:, 0].argsort()]  # sort acceleration changes by time

        # Final trajectory acceleration is cumulative sum of acceleration steps which gives the superposition of the
        # contribution from each impulse and is equivalent to the convolution
        trajectory_acceleration = np.zeros([basic_trajectory_acceleration.shape[0] * len(impulses),
                                            basic_trajectory_acceleration.shape[1]])
        trajectory_acceleration[:, 0] = accel_changes[:, 0]  # Time
        trajectory_acceleration[:, 1] = np.cumsum(accel_changes[:, 1])  # Acceleration

        # trajectory is one row less since first row would be initial position (zeros)
        pvt_trajectory = np.zeros([trajectory_acceleration.shape[0] - 1, 3])
        for n in range(0, len(pvt_trajectory)):
            # Calculate position and velocity at each point using equations for constant acceleration since acceleration
            # changes are steps.
            if n > 0:
                prev_x = pvt_trajectory[n - 1, 0]
                prev_v = pvt_trajectory[n - 1, 1]
            else:
                prev_x = 0
                prev_v = 0

            # dt[n] = t[n] - t[n-1]
            # v[n] = v[n-1] + a[n] * dt[n]
            # p[n] = x[n-1] + (v[n] + v[n-1])/2 * dt[n]
            pvt_trajectory[n, 2] = trajectory_acceleration[n + 1, 0] - trajectory_acceleration[n, 0]  # dt
            pvt_trajectory[n, 1] = prev_v + trajectory_acceleration[n, 1] * pvt_trajectory[n, 2]  # velocity
            pvt_trajectory[n, 0] = prev_x + (pvt_trajectory[n, 1] + prev_v) / 2 * pvt_trajectory[n, 2]  # position

        # make sure end point velocity is 0 and position is exactly on target
        pvt_trajectory[-1, 0] = distance
        pvt_trajectory[-1, 1] = 0

        return pvt_trajectory.tolist()

# Example code for using the class.
if __name__ == "__main__":
    shaper = ZeroVibrationTrajectoryGenerator(4.64, 0.04)

    DIST = 680
    ACCEL = 2100
    MAX_SPEED = 1000

    pvt_points = shaper.shape_trapezoidal_motion(DIST, ACCEL, ACCEL, MAX_SPEED)
    print('Position, Velocity, Time')
    for point in pvt_points:
        print(*point, sep=',')

    print("")
    print(
        f"Shaped Move: Max Speed: {np.max(np.abs(np.array(pvt_points)[:,1])):.2f}, "
        f"Total Time: {np.sum(np.array(pvt_points)[:, 2]):.2f}, "
    )
