"""
This file contains the ShapedAxisStream2D class, which is to be re-used in your code.

Run the file directly to test the class out with a Zaber Device.
"""

# pylint: disable=too-many-arguments

import sys
import time
import math
from zaber_motion import Units, Measurement, CommandFailedException, LockstepNotEnabledException
from zaber_motion.ascii import Axis, Lockstep, StreamAxisDefinition, StreamAxisType, Connection
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

        self._x_max_speed_limit = -1.0
        self._y_max_speed_limit = -1.0

        # Set the speed limit to the device's current maxspeed so it will never be exceeded
        self.reset_max_speed_limits()

    def get_x_max_speed_limit(self, unit: Units = Units.NATIVE) -> float:
        """
        Get the current velocity limit for which shaped moves will not exceed.

        :param unit: The value will be returned in these units.
        :return: The velocity limit.
        """
        return self._x_primary_axis.settings.convert_from_native_units(
            "maxspeed", self._x_max_speed_limit, unit
        )

    def get_y_max_speed_limit(self, unit: Units = Units.NATIVE) -> float:
        """
        Get the current velocity limit for which shaped moves will not exceed.

        :param unit: The value will be returned in these units.
        :return: The velocity limit.
        """
        return self._y_primary_axis.settings.convert_from_native_units(
            "maxspeed", self._y_max_speed_limit, unit
        )

    def set_x_max_speed_limit(self, value: float, unit: Units = Units.NATIVE) -> None:
        """
        Set the velocity limit for which shaped moves will not exceed.

        :param value: The velocity limit.
        :param unit: The units of the velocity limit value.
        """
        self._x_max_speed_limit = self._x_primary_axis.settings.convert_to_native_units(
            "maxspeed", value, unit
        )

    def set_y_max_speed_limit(self, value: float, unit: Units = Units.NATIVE) -> None:
        """
        Set the velocity limit for which shaped moves will not exceed.

        :param value: The velocity limit.
        :param unit: The units of the velocity limit value.
        """
        self._y_max_speed_limit = self._y_primary_axis.settings.convert_to_native_units(
            "maxspeed", value, unit
        )

    def reset_max_speed_limits(self) -> None:
        """Reset the velocity limit for shaped moves to the device's existing maxspeed setting."""
        if isinstance(self.x_axis, Lockstep):
            self.set_x_max_speed_limit(min(self.get_setting_from_x_lockstep_axes("maxspeed")))
        else:
            self.set_x_max_speed_limit(self.x_axis.settings.get("maxspeed"))

        if isinstance(self.y_axis, Lockstep):
            self.set_y_max_speed_limit(min(self.get_setting_from_y_lockstep_axes("maxspeed")))
        else:
            self.set_y_max_speed_limit(self.y_axis.settings.get("maxspeed"))

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
            if not self.y_axis.is_homed():
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
        if (x_position == 0.0) & (y_position == 0.0):
            # Travel distance is 0. No movement.
            return

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

        x_maxspeed_mm = self.get_x_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND)
        y_maxspeed_mm = self.get_y_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND)

        # Calculate accel and max speed along trajectory direction
        distance = math.sqrt(x_position ** 2 + y_position ** 2)
        x_ratio = abs(x_position) / distance  # cos(theta)
        y_ratio = abs(y_position) / distance  # sin(theta)
        if x_ratio == 0:
            accel_mm = y_accel_mm / y_ratio
            decel_mm = y_decel_mm / y_ratio
            maxspeed_mm = y_maxspeed_mm / y_ratio
        elif y_ratio == 0:
            accel_mm = x_accel_mm / x_ratio
            decel_mm = x_decel_mm / x_ratio
            maxspeed_mm = x_maxspeed_mm / x_ratio
        else:
            accel_mm = min(x_accel_mm / x_ratio, y_accel_mm / y_ratio)
            decel_mm = min(x_decel_mm / x_ratio, y_decel_mm / y_ratio)
            maxspeed_mm = min(x_maxspeed_mm / x_ratio, y_maxspeed_mm / y_ratio)

        x_start_position = self.x_axis.get_position(Units.LENGTH_MILLIMETRES)
        y_start_position = self.y_axis.get_position(Units.LENGTH_MILLIMETRES)

        stream_segments = self.shaper.shape_trapezoidal_motion(
            x_position_mm,
            y_position_mm,
            accel_mm,
            decel_mm,
            maxspeed_mm,
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

        # Disable centripetal accel limit by setting to 0 so that it doesn't alter the trajectory
        self.stream.set_max_centripetal_acceleration(0, Units.NATIVE)

        for segment in stream_segments:
            # Set acceleration making sure it is greater than zero by comparing 1 native accel unit
            if (
                (self._x_primary_axis.settings.convert_to_native_units(
                    "accel", segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED) > 1)
                & (self._y_primary_axis.settings.convert_to_native_units(
                "accel", segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED) > 1)
            ):
                self.stream.set_max_tangential_acceleration(
                    segment.accel, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
                )
            else:
                self.stream.set_max_tangential_acceleration(1, Units.NATIVE)

            # Set max speed making sure that it is at least 1 native speed unit
            if (
                (self._x_primary_axis.settings.convert_to_native_units(
                    "maxspeed", segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND) > 1)
                & (self._y_primary_axis.settings.convert_to_native_units(
                "maxspeed", segment.speed_limit, Units.VELOCITY_MILLIMETRES_PER_SECOND) > 1)
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


# Example code for using the class.
if __name__ == "__main__":
    COM_PORT = "COMX"  # The COM port with the connected Zaber device.
    DEVICE_INDEX = 0  # The Zaber device index to test.
    X_AXIS_INDEX = 1  # The x-axis index to test.
    Y_AXIS_INDEX = 3  # The y-axis index to test.
    X_RESONANT_FREQUENCY = 6  # Input shaping resonant frequency in Hz.
    X_DAMPING_RATIO = 0.1  # Input shaping damping ratio.
    Y_RESONANT_FREQUENCY = 2.5  # Input shaping resonant frequency in Hz.
    Y_DAMPING_RATIO = 0.1  # Input shaping damping ratio.
    X_MOVE_DISTANCE = 200  # The move distance in mm to use in relative moves.
    Y_MOVE_DISTANCE = 100  # The move distance in mm to use in relative moves.
    PAUSE_TIME = 1  # Amount of time in seconds to pause between moves.
    STREAM_SHAPER_TYPE = ShaperType.ZV  # Input shaper type to use for ShapedAxisStream.
    LIMITED_MOVE_SPEED = 200  # Speed limit in mm/s to use in demo of moves with speed limit

    with Connection.open_serial_port(COM_PORT) as connection:
        # Get all the devices on the connection
        device_list = connection.detect_devices()
        print(f"Found {len(device_list)} devices.")

        if len(device_list) < 1:
            print("No devices, exiting.")
            sys.exit(0)

        device = device_list[0]  # Get the first device on the port

        # Check if axis is part of lockstep group
        X_LOCKSTEP_INDEX = 0
        Y_LOCKSTEP_INDEX = 0
        try:
            num_lockstep_groups_possible = device.settings.get("lockstep.numgroups")
            for group_num in range(1, int(num_lockstep_groups_possible) + 1):
                try:
                    axis_nums = device.get_lockstep(group_num).get_axis_numbers()
                    if X_AXIS_INDEX in axis_nums:
                        print(f"Axis {X_AXIS_INDEX} is part of Lockstep group {group_num}.")
                        X_LOCKSTEP_INDEX = group_num
                    if Y_AXIS_INDEX in axis_nums:
                        print(f"Axis {Y_AXIS_INDEX} is part of Lockstep group {group_num}.")
                        Y_LOCKSTEP_INDEX = group_num
                except LockstepNotEnabledException:
                    pass
        except CommandFailedException:
            # Unable to get lockstep.numgroups settings meaning device is not capable of lockstep.
            pass

        x_zaber_object: Axis | Lockstep
        if X_LOCKSTEP_INDEX == 0:
            x_zaber_object = device.get_axis(X_AXIS_INDEX)
        else:
            x_zaber_object = device.get_lockstep(X_LOCKSTEP_INDEX)

        y_zaber_object: Axis | Lockstep
        if Y_LOCKSTEP_INDEX == 0:
            y_zaber_object = device.get_axis(Y_AXIS_INDEX)
        else:
            y_zaber_object = device.get_lockstep(Y_LOCKSTEP_INDEX)

        # Initialize a Plant class with the frequency and damping ratio
        x_plant_var = Plant(X_RESONANT_FREQUENCY, X_DAMPING_RATIO)
        y_plant_var = Plant(Y_RESONANT_FREQUENCY, Y_DAMPING_RATIO)

        shaped_axis = ShapedAxisStream2D(
            x_zaber_object, y_zaber_object, x_plant_var, y_plant_var, STREAM_SHAPER_TYPE)

        if not shaped_axis.is_homed():
            raise Exception("Devices are not homed! Home devices before running script.")

        print("Performing shaped moves.")
        shaped_axis.move_relative(X_MOVE_DISTANCE, Y_MOVE_DISTANCE, Units.LENGTH_MILLIMETRES, True)
        time.sleep(PAUSE_TIME)
        shaped_axis.move_relative(-X_MOVE_DISTANCE, -Y_MOVE_DISTANCE, Units.LENGTH_MILLIMETRES, True)
        time.sleep(PAUSE_TIME)

        print("Performing shaped moves with speed limit.")
        shaped_axis.set_x_max_speed_limit(LIMITED_MOVE_SPEED, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        shaped_axis.set_y_max_speed_limit(LIMITED_MOVE_SPEED, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        shaped_axis.move_relative(X_MOVE_DISTANCE, Y_MOVE_DISTANCE, Units.LENGTH_MILLIMETRES, True)
        time.sleep(PAUSE_TIME)
        shaped_axis.move_relative(-X_MOVE_DISTANCE, -Y_MOVE_DISTANCE, Units.LENGTH_MILLIMETRES, True)
        time.sleep(PAUSE_TIME)
        shaped_axis.reset_max_speed_limits()
