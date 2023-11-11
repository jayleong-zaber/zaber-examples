"""
This file contains the ShapedAxisStream class, which is to be re-used in your code.

Run the file directly to test the class out with a Zaber Device.
"""

# pylint: disable=too-many-arguments

import numpy as np
from zaber_motion import Units, Measurement
from zaber_motion.ascii import Axis, Lockstep, StreamAxisDefinition, StreamAxisType
from zero_vibration_stream_generator import ZeroVibrationStreamGenerator, ShaperType
from plant import Plant
from axis_lockstep_helper import AxisLockstep


class ShapedAxisStream:
    """A Zaber device axis that performs streamed moves with input shaping vibration reduction."""

    def __init__(
        self,
        zaber_axis: Axis | Lockstep,
        plant: Plant,
        shaper_type: ShaperType = ShaperType.ZV,
        stream_id: int = 1,
    ) -> None:
        """
        Initialize the class for the specified axis.

        :param zaber_axis: The Zaber Motion Axis or Lockstep object
        :param plant: The Plant instance defining the system that the shaper is targeting
        :shaper_type: Type of input shaper to use
        :stream_id: Stream number on device to use to perform moves
        """
        self._axis_lockstep = AxisLockstep(zaber_axis)
        self.shaper = ZeroVibrationStreamGenerator(plant, shaper_type)
        self.stream = zaber_axis.device.get_stream(stream_id)

        self._max_speed_limit = -1.0

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
        position_native = self._axis_lockstep.settings.convert_to_native_units("pos", position, unit)
        accel_native = self._axis_lockstep.settings.convert_to_native_units(
            "accel", acceleration, acceleration_unit
        )
        decel_native = accel_native

        if acceleration == 0:  # Get the acceleration and deceleration if it wasn't specified
            accel_native = self._axis_lockstep.settings.get("accel", Units.NATIVE)
            decel_native = self._axis_lockstep.settings.get("motion.decelonly", Units.NATIVE)

        position_mm = self._axis_lockstep.settings.convert_from_native_units(
            "pos", position_native, Units.LENGTH_MILLIMETRES
        )
        accel_mm = self._axis_lockstep.settings.convert_from_native_units(
            "accel", accel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )
        decel_mm = self._axis_lockstep.settings.convert_from_native_units(
            "accel", decel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )

        start_position = self._axis_lockstep.get_position(Units.LENGTH_MILLIMETRES)

        stream_segments = self.shaper.shape_trapezoidal_motion(
            position_mm,
            accel_mm,
            decel_mm,
            self.get_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND),
        )

        self.stream.disable()
        if isinstance(self._axis_lockstep.axis, Lockstep):
            self.stream.setup_live_composite(
                StreamAxisDefinition(self._axis_lockstep.axis.lockstep_group_id,
                                     StreamAxisType.LOCKSTEP)
            )
        else:
            self.stream.setup_live(self._axis_lockstep.axis.axis_number)
        self.stream.cork()
        for segment in stream_segments:
            # Set acceleration making sure it is greater than zero by comparing 1 native accel unit
            if (
                self._axis_lockstep.settings.convert_to_native_units(
                    "accel", segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
                )
                > 1
            ):
                self.stream.set_max_tangential_acceleration(
                    segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
                )
            else:
                self.stream.set_max_tangential_acceleration(1, Units.NATIVE)

            # Set max speed making sure that it is at least 1 native speed unit
            if (
                self._axis_lockstep.settings.convert_to_native_units(
                    "maxspeed", segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND
                )
                > 1
            ):
                self.stream.set_max_speed(
                    segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND
                )
            else:
                self.stream.set_max_speed(1, Units.NATIVE)

            # set position for the end of the segment
            self.stream.line_absolute(
                Measurement(segment.position + start_position, Units.LENGTH_MILLIMETRES)
            )
        self.stream.uncork()

        if wait_until_idle:
            self.stream.wait_until_idle()

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
