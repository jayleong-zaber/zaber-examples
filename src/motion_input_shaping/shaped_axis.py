"""
This file contains the ShapedAxis class, which is to be re-used in your code.

Run the file directly to test the class out with a Zaber Device.
"""

# pylint: disable=too-many-arguments

import numpy as np
from zaber_motion import Units
from zaber_motion.ascii import Axis, Lockstep
from zero_vibration_shaper import ZeroVibrationShaper
from plant import Plant
from axis_lockstep_helper import AxisLockstep


class ShapedAxis:
    """A Zaber device axis that performs moves with input shaping vibration reduction theory."""

    def __init__(
        self,
        zaber_axis: Axis | Lockstep,
        plant: Plant,
    ) -> None:
        """
        Initialize the class for the specified axis.

        :param zaber_axis: The Zaber Motion Axis or Lockstep object
        :param plant: The Plant instance defining the system that the shaper is targeting.
        """
        self._axis_lockstep = AxisLockstep(zaber_axis)
        self.shaper = ZeroVibrationShaper(plant)

        self._max_speed_limit = -1.0

        # Grab the current deceleration so we can reset it back to this value later if we want.
        self._original_deceleration = self._axis_lockstep.settings.get_all(
            "motion.decelonly", Units.NATIVE
        )

        # Set the speed limit to the device's current maxspeed so it will never be exceeded
        self.reset_max_speed_limit()

    @property
    def axis(self) -> Axis | Lockstep:
        # Return Axis or Lockstep instance from hidden AxisLockstep class
        return self._axis_lockstep.axis

    def is_homed(self) -> bool:
        """Check if all axes in lockstep group are homed."""
        return self._axis_lockstep.is_homed()

    def get_max_speed_limit(self, unit: Units = Units.NATIVE) -> float:
        """
        Get the current velocity limit for which shaped moves will not exceed.

        :param unit: The value will be returned in these units.
        :return: The velocity limit.
        """
        return self._axis_lockstep.settings.convert_from_native_units(
            "maxspeed", self._max_speed_limit, unit
        )

    def set_max_speed_limit(self, value: float, unit: Units = Units.NATIVE) -> None:
        """
        Set the velocity limit for which shaped moves will not exceed.

        :param value: The velocity limit.
        :param unit: The units of the velocity limit value.
        """
        self._max_speed_limit = self._axis_lockstep.settings.convert_to_native_units(
            "maxspeed", value, unit
        )

    def reset_max_speed_limit(self) -> None:
        """Reset the velocity limit for shaped moves to the device's existing maxspeed setting."""
        self.set_max_speed_limit(self._axis_lockstep.settings.get("maxspeed"))

    def reset_deceleration(self) -> None:
        """Reset the trajectory deceleration to the value stored when the class was created."""
        self._axis_lockstep.settings.set(
            "motion.decelonly", self._original_deceleration, Units.NATIVE
        )

    def move_relative(
        self,
        position: float,
        unit: Units = Units.NATIVE,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped relative move for the target resonant frequency and damping ratio.

        :param position: The amount to move.
        :param unit: The units for the position value.
        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        # Convert all to values to the same units
        position_native = self._axis_lockstep.settings.convert_to_native_units(
            "pos", position, unit
        )
        accel_native: float | list[float]
        accel_native = self._axis_lockstep.settings.convert_to_native_units(
            "accel", acceleration, acceleration_unit
        )

        if acceleration == 0:  # Get the acceleration if it wasn't specified
            accel_native = self._axis_lockstep.settings.get("accel", Units.NATIVE)

        position_mm = self._axis_lockstep.settings.convert_from_native_units(
            "pos", position_native, Units.LENGTH_MILLIMETRES
        )
        accel_mm = self._axis_lockstep.settings.convert_from_native_units(
            "accel", accel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )

        # Apply the input shaping with all values of the same units
        deceleration_mm, max_speed_mm = self.shaper.shape_trapezoidal_motion(
            position_mm,
            accel_mm,
            self.get_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND),
        )

        # Check if the target deceleration is different from the current value
        deceleration_native = round(
            self._axis_lockstep.settings.convert_to_native_units(
                "accel", deceleration_mm, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
            )
        )

        if (
            self._axis_lockstep.settings.get("motion.decelonly", Units.NATIVE)
            != deceleration_native
        ):
            self._axis_lockstep.settings.set("motion.decelonly", deceleration_native, Units.NATIVE)

        # Perform the move
        self._axis_lockstep.axis.move_relative(
            position,
            unit,
            wait_until_idle,
            max_speed_mm,
            Units.VELOCITY_MILLIMETRES_PER_SECOND,
            accel_mm,
            Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED,
        )

    def move_absolute(
        self,
        position: float,
        unit: Units = Units.NATIVE,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped absolute move for the target resonant frequency and damping ratio.

        :param position: The position to move to.
        :param unit: The units for the position value.
        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        current_position = self._axis_lockstep.axis.get_position(unit)
        self.move_relative(
            position - current_position, unit, wait_until_idle, acceleration, acceleration_unit
        )

    def move_max(
        self,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped move to the max limit for the target resonant frequency and damping ratio.

        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        largest_possible_move = self._axis_lockstep.get_max_relative_move()
        self.move_relative(
            largest_possible_move, Units.NATIVE, wait_until_idle, acceleration, acceleration_unit
        )

    def move_min(
        self,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped move to the min limit for the target resonant frequency and damping ratio.

        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        largest_possible_move = self._axis_lockstep.get_min_relative_move()
        self.move_relative(
            largest_possible_move, Units.NATIVE, wait_until_idle, acceleration, acceleration_unit
        )
