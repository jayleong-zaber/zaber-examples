"""
This file contains the ShapedAxisStream2D class, which is to be re-used in your code.

Run the file directly to test the class out with a Zaber Device.
"""

# pylint: disable=too-many-arguments

import numpy as np
from zaber_motion import Units, Measurement
from zaber_motion.ascii import Axis, Lockstep, StreamAxisDefinition, StreamAxisType
from zero_vibration_stream_generator import ShaperType
from zero_vibration_stream_generator_2d import ZeroVibrationStreamGenerator2D
from plant import Plant


def check_zaber_axis(axis: Axis | Lockstep):
    """Check whether axis or lockstep group is valid"""
    if isinstance(axis, Axis):
        # Sanity check if the passed x_axis has a higher number than the number of axes on the
        # device.
        if axis.axis_number > axis.device.axis_count or axis is None:
            raise TypeError("Invalid Axis class was used to initialized ShapedAxisStream2D.")
    elif isinstance(axis, Lockstep):
        # Sanity check if the passed lockstep group number exceeds than the number of
        # lockstep groups on the device.
        if (
            axis.lockstep_group_id > axis.device.settings.get("lockstep.numgroups")
            or axis is None
        ):
            raise TypeError("Invalid Lockstep class was used to initialized ShapedAxisStream2D.")


class ShapedAxisStream2D:
    """A Zaber device x_axis that performs streamed moves with input shaping vibration reduction."""

    def __init__(
        self,
        x_zaber_axis: Axis | Lockstep,
        y_zaber_axis: Axis | Lockstep,
        x_plant: Plant,
        y_plant: Plant,
        shaper_type: ShaperType = ShaperType.ZV,
        stream_id: int = 1,
    ) -> None:
        """
        Initialize the class for the specified pair of axes.

        :param x_zaber_axis: The Zaber Motion Axis or Lockstep object for X-axis
        :param y_zaber_axis: The Zaber Motion Axis or Lockstep object for Y-axis
        :param x_plant: The Plant instance defining the system that the x-axis shaper is targeting
        :param y_plant: The Plant instance defining the system that the y-axis shaper is targeting
        :shaper_type: Type of input shaper to use
        :stream_id: Stream number on device to use to perform moves
        """
        check_zaber_axis(x_zaber_axis)
        check_zaber_axis(y_zaber_axis)

        self.x_axis = x_zaber_axis
        self.y_axis = y_zaber_axis

        if isinstance(self.x_axis, Lockstep):
            # Get x_axis numbers that are used so that settings can be changed
            self._x_lockstep_axes = []
            for axis_number in self.x_axis.get_axis_numbers():
                self._x_lockstep_axes.append(self.x_axis.device.get_axis(axis_number))
            self._x_primary_axis = self._x_lockstep_axes[0]
        else:
            self._x_primary_axis = self.x_axis

        if isinstance(self.y_axis, Lockstep):
            # Get x_axis numbers that are used so that settings can be changed
            self._y_lockstep_axes = []
            for axis_number in self.y_axis.get_axis_numbers():
                self._y_lockstep_axes.append(self.y_axis.device.get_axis(axis_number))
            self._y_primary_axis = self._y_lockstep_axes[0]
        else:
            self._y_primary_axis = self.y_axis

        self.shaper = ZeroVibrationStreamGenerator2D(x_plant, y_plant, shaper_type)
        self.stream = x_zaber_axis.device.streams.get_stream(stream_id)

        self._max_speed_limit = -1.0

        # Set the speed limit to the device's current maxspeed so it will never be exceeded
        self.reset_max_speed_limit()

    def get_max_speed_limit(self, unit: Units = Units.NATIVE) -> float:
        """
        Get the current velocity limit for which shaped moves will not exceed.

        :param unit: The value will be returned in these units.
        :return: The velocity limit.
        """
        return self._x_primary_axis.settings.convert_from_native_units(
            "maxspeed", self._max_speed_limit, unit
        )

    def set_max_speed_limit(self, value: float, unit: Units = Units.NATIVE) -> None:
        """
        Set the velocity limit for which shaped moves will not exceed.

        :param value: The velocity limit.
        :param unit: The units of the velocity limit value.
        """
        self._max_speed_limit = self._x_primary_axis.settings.convert_to_native_units(
            "maxspeed", value, unit
        )

    def reset_max_speed_limit(self) -> None:
        """Reset the velocity limit for shaped moves to the device's existing maxspeed setting."""
        if isinstance(self.x_axis, Lockstep):
            self.set_max_speed_limit(min(self.get_setting_from_x_lockstep_axes("maxspeed")))
        else:
            self.set_max_speed_limit(self.x_axis.settings.get("maxspeed"))

    def is_homed(self) -> bool:
        """Check if all axes in lockstep group are homed."""
        if isinstance(self.x_axis, Lockstep):
            for axis in self._x_lockstep_axes:
                if not axis.is_homed():
                    return False
        else:
            if not self.x_axis.is_homed():
                return False

        if isinstance(self.y_axis, Lockstep):
            for axis in self._y_lockstep_axes:
                if not axis.is_homed():
                    return False
        else:
            if not self.x_axis.is_homed():
                return False

        return True

    def get_setting_from_x_lockstep_axes(
        self, setting: str, unit: Units = Units.NATIVE
    ) -> list[float]:
        """
        Get setting values from axes in the lockstep group.

        :param setting: The name of setting
        :param unit: The values will be returned in these units.
        :return: A list of setting values
        """
        values = []
        for axis in self._x_lockstep_axes:
            values.append(axis.settings.get(setting, unit))
        return values

    def get_setting_from_y_lockstep_axes(
        self, setting: str, unit: Units = Units.NATIVE
    ) -> list[float]:
        """
        Get setting values from axes in the lockstep group.

        :param setting: The name of setting
        :param unit: The values will be returned in these units.
        :return: A list of setting values
        """
        values = []
        for axis in self._y_lockstep_axes:
            values.append(axis.settings.get(setting, unit))
        return values

    def get_lockstep_x_axes_positions(self, unit: Units = Units.NATIVE) -> list[float]:
        """
        Get positions from axes in the lockstep group.

        :param unit: The positions will be returned in these units.
        :return: A list of setting values
        """
        positions = []
        for axis in self._x_lockstep_axes:
            positions.append(axis.get_position(unit))
        return positions

    def get_lockstep_y_axes_positions(self, unit: Units = Units.NATIVE) -> list[float]:
        """
        Get positions from axes in the lockstep group.

        :param unit: The positions will be returned in these units.
        :return: A list of setting values
        """
        positions = []
        for axis in self._y_lockstep_axes:
            positions.append(axis.get_position(unit))
        return positions

    def move_relative(
        self,
        x_position: float,
        y_position: float,
        unit: Units = Units.NATIVE,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped relative move for the target resonant frequency and damping ratio.

        :param x_position: The amount to move x-axis.
        :param y_position: The amount to move y-axis.
        :param unit: The units for the position value.
        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        # Convert all to values to the same units
        x_position_native = self._x_primary_axis.settings.convert_to_native_units(
            "pos", x_position, unit)
        y_position_native = self._y_primary_axis.settings.convert_to_native_units(
            "pos", y_position, unit)

        x_accel_native = self._x_primary_axis.settings.convert_to_native_units(
            "accel", acceleration, acceleration_unit
        )
        x_decel_native = x_accel_native
        y_accel_native = self._y_primary_axis.settings.convert_to_native_units(
            "accel", acceleration, acceleration_unit
        )
        y_decel_native = y_accel_native

        if acceleration == 0:  # Get the acceleration and deceleration if it wasn't specified
            if isinstance(self.x_axis, Lockstep):
                x_accel_native = min(self.get_setting_from_x_lockstep_axes("accel", Units.NATIVE))
                x_decel_native = min(
                    self.get_setting_from_x_lockstep_axes("motion.decelonly", Units.NATIVE)
                )
            else:
                x_accel_native = self.x_axis.settings.get("accel", Units.NATIVE)
                x_decel_native = self.x_axis.settings.get("motion.decelonly", Units.NATIVE)
            if isinstance(self.y_axis, Lockstep):
                y_accel_native = min(self.get_setting_from_y_lockstep_axes("accel", Units.NATIVE))
                y_decel_native = min(
                    self.get_setting_from_y_lockstep_axes("motion.decelonly", Units.NATIVE)
                )
            else:
                y_accel_native = self.y_axis.settings.get("accel", Units.NATIVE)
                y_decel_native = self.y_axis.settings.get("motion.decelonly", Units.NATIVE)

        x_position_mm = self._x_primary_axis.settings.convert_from_native_units(
            "pos", x_position_native, Units.LENGTH_MILLIMETRES
        )
        y_position_mm = self._y_primary_axis.settings.convert_from_native_units(
            "pos", y_position_native, Units.LENGTH_MILLIMETRES
        )
        x_accel_mm = self._x_primary_axis.settings.convert_from_native_units(
            "accel", x_accel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )
        y_accel_mm = self._y_primary_axis.settings.convert_from_native_units(
            "accel", y_accel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )
        x_decel_mm = self._x_primary_axis.settings.convert_from_native_units(
            "accel", x_decel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )
        y_decel_mm = self._y_primary_axis.settings.convert_from_native_units(
            "accel", y_decel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )

        accel_mm = min(x_accel_mm, y_accel_mm)
        decel_mm = min(x_decel_native, y_decel_mm)

        x_start_position = self.x_axis.get_position(Units.LENGTH_MILLIMETRES)
        y_start_position = self.y_axis.get_position(Units.LENGTH_MILLIMETRES)

        stream_segments = self.shaper.shape_trapezoidal_motion(
            x_position_mm,
            y_position_mm,
            accel_mm,
            decel_mm,
            self.get_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND),
        )

        self.stream.disable()
        if isinstance(self.x_axis, Lockstep):
            x_axis_definition = StreamAxisDefinition(self.x_axis.lockstep_group_id,
                                                     StreamAxisType.LOCKSTEP)
        else:
            x_axis_definition = StreamAxisDefinition(self.x_axis.axis_number,
                                                     StreamAxisType.PHYSICAL)
        if isinstance(self.y_axis, Lockstep):
            y_axis_definition = StreamAxisDefinition(self.y_axis.lockstep_group_id,
                                                     StreamAxisType.LOCKSTEP)
        else:
            y_axis_definition = StreamAxisDefinition(self.y_axis.axis_number,
                                                     StreamAxisType.PHYSICAL)
        self.stream.setup_live_composite(x_axis_definition, y_axis_definition)

        for segment in stream_segments:
            # Set acceleration making sure it is greater than zero by comparing 1 native accel unit
            if (
                (self._x_primary_axis.settings.convert_to_native_units(
                    "accel", segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
                ) > 1)
                & (self._y_primary_axis.settings.convert_to_native_units(
                "accel", segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
            ) > 1)
            ):
                self.stream.set_max_tangential_acceleration(
                    segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
                )
            else:
                self.stream.set_max_tangential_acceleration(1, Units.NATIVE)

            # Set max speed making sure that it is at least 1 native speed unit
            if (
                (self._x_primary_axis.settings.convert_to_native_units(
                    "maxspeed", segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND
                ) > 1)
                & (self._y_primary_axis.settings.convert_to_native_units(
                "maxspeed", segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND
            ) > 1)
            ):
                self.stream.set_max_speed(
                    segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND
                )
            else:
                self.stream.set_max_speed(1, Units.NATIVE)

            # set position for the end of the segment
            self.stream.line_absolute(
                Measurement(segment.x_position + x_start_position, Units.LENGTH_MILLIMETRES),
                Measurement(segment.y_position + y_start_position, Units.LENGTH_MILLIMETRES)
            )
        self.stream.uncork()

        if wait_until_idle:
            self.stream.wait_until_idle()

    def move_absolute(
        self,
        x_position: float,
        y_position: float,
        unit: Units = Units.NATIVE,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped absolute move for the target resonant frequency and damping ratio.

        :param x_position: The position to move x-axis to.
        :param y_position: The position to move y-axis to.
        :param unit: The units for the position value.
        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        x_current_position = self.x_axis.get_position(unit)
        y_current_position = self.y_axis.get_position(unit)
        self.move_relative(
            x_position - x_current_position,
            y_position - y_current_position,
            unit,
            wait_until_idle,
            acceleration,
            acceleration_unit
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
        if isinstance(self.x_axis, Lockstep):
            x_current_axis_positions = self.get_lockstep_x_axes_positions(Units.NATIVE)
            x_end_positions = self.get_setting_from_x_lockstep_axes("limit.max", Units.NATIVE)
            # Move will be positive so find min relative move
            x_largest_possible_move = np.min(np.subtract(x_end_positions, x_current_axis_positions))
        else:
            x_current_position = self.x_axis.get_position(Units.NATIVE)
            x_end_position = self.x_axis.settings.get("limit.max", Units.NATIVE)
            x_largest_possible_move = x_end_position - x_current_position

        if isinstance(self.y_axis, Lockstep):
            y_current_axis_positions = self.get_lockstep_y_axes_positions(Units.NATIVE)
            y_end_positions = self.get_setting_from_y_lockstep_axes("limit.max", Units.NATIVE)
            # Move will be positive so find min relative move
            y_largest_possible_move = np.min(np.subtract(y_end_positions, y_current_axis_positions))
        else:
            y_current_position = self.y_axis.get_position(Units.NATIVE)
            y_end_position = self.y_axis.settings.get("limit.max", Units.NATIVE)
            y_largest_possible_move = y_end_position - y_current_position

        self.move_relative(
            x_largest_possible_move,
            y_largest_possible_move,
            Units.NATIVE,
            wait_until_idle,
            acceleration,
            acceleration_unit
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
        if isinstance(self.x_axis, Lockstep):
            x_current_axis_positions = self.get_lockstep_x_axes_positions(Units.NATIVE)
            x_end_positions = self.get_setting_from_x_lockstep_axes("limit.min", Units.NATIVE)
            # Move will be negative so find max relative move
            x_largest_possible_move = np.max(np.subtract(x_end_positions, x_current_axis_positions))
        else:
            x_current_position = self.x_axis.get_position(Units.NATIVE)
            x_end_position = self.x_axis.settings.get("limit.min", Units.NATIVE)
            x_largest_possible_move = x_end_position - x_current_position

        if isinstance(self.y_axis, Lockstep):
            y_current_axis_positions = self.get_lockstep_y_axes_positions(Units.NATIVE)
            y_end_positions = self.get_setting_from_y_lockstep_axes("limit.min", Units.NATIVE)
            # Move will be negative so find max relative move
            y_largest_possible_move = np.max(np.subtract(y_end_positions, y_current_axis_positions))
        else:
            y_current_position = self.y_axis.get_position(Units.NATIVE)
            y_end_position = self.y_axis.settings.get("limit.min", Units.NATIVE)
            y_largest_possible_move = y_end_position - y_current_position

        self.move_relative(
            x_largest_possible_move,
            y_largest_possible_move,
            Units.NATIVE,
            wait_until_idle,
            acceleration,
            acceleration_unit
        )
