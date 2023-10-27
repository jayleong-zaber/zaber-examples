"""
This file contains the ShapedAxis class, which is to be re-used in your code.

Run the file directly to test the class out with a Zaber Device.
"""

# pylint: disable=too-many-arguments

import time
import sys
from zaber_motion import Units, Measurement
from zaber_motion.ascii import Connection, Axis, Lockstep
from zero_vibration_shaper import ZeroVibrationShaper
from zero_vibration_stream_generator import ZeroVibrationStreamGenerator, ShaperType
from shaper_config import ShaperConfig, ShaperMode


class ShapedAxis:
    """
    A Zaber device axis with extra functions.

    Used for performing moves with input shaping vibration reduction theory.
    """

    def __init__(
        self,
        zaber_axis: Axis,
        resonant_frequency: float,
        damping_ratio: float,
        shaper_config: ShaperConfig,
    ) -> None:
        """
        Initialize the class for the specified axis.

        Set a max speed limit to the current maxspeed setting
        so the input shaping algorithm won't exceed that value.

        :param zaber_axis: The Zaber Motion Axis or Lockstep object
        :param resonant_frequency: The target resonant frequency for shaped moves [Hz]
        :param damping_ratio: The target damping ratio for shaped moves
        :shaper_config: ShaperConfig object containing settings for the shaper
        """

        # Sanity check if the passed axis has a higher number than the number of axes on the device.
        if zaber_axis.axis_number > zaber_axis.device.axis_count or zaber_axis is None:
            raise TypeError("Invalid Axis class was used to initialized ShapedAxis.")

        self.axis = zaber_axis

        match shaper_config.shaper_mode:
            case ShaperMode.DECEL:
                self.shaper = ZeroVibrationShaper(resonant_frequency, damping_ratio)
            case ShaperMode.STREAM:
                self.shaper = ZeroVibrationStreamGenerator(
                    resonant_frequency,
                    damping_ratio,
                    shaper_type=shaper_config.settings["shaper_type"],
                )
                self.stream = zaber_axis.device.get_stream(shaper_config.settings["stream_id"])

        self._max_speed_limit = -1.0

        # Grab the current deceleration so we can reset it back to this value later if we want.
        self._original_deceleration = self.axis.settings.get("motion.decelonly", Units.NATIVE)

        # Set the speed limit to the device's current maxspeed so it will never be exceeded
        self.reset_max_speed_limit()

    def get_max_speed_limit(self, unit: Units = Units.NATIVE) -> float:
        """
        Get the current velocity limit for which shaped moves will not exceed.

        :param unit: The value will be returned in these units.
        :return: The velocity limit.
        """
        return self.axis.settings.convert_from_native_units("maxspeed", self._max_speed_limit, unit)

    def set_max_speed_limit(self, value: float, unit: Units = Units.NATIVE) -> None:
        """
        Set the velocity limit for which shaped moves will not exceed.

        :param value: The velocity limit.
        :param unit: The units of the velocity limit value.
        """
        self._max_speed_limit = self.axis.settings.convert_to_native_units("maxspeed", value, unit)

    def reset_max_speed_limit(self) -> None:
        """Reset the velocity limit for shaped moves to the device's existing maxspeed setting."""
        self.set_max_speed_limit(self.axis.settings.get("maxspeed"))

    def reset_deceleration(self) -> None:
        """Reset the trajectory deceleration to the value stored when the class was created."""
        self.axis.settings.set("motion.decelonly", self._original_deceleration, Units.NATIVE)

    def move_relative(
        self,
        position: float,
        unit: Units = Units.NATIVE,
        wait_until_idle: bool = True,
        acceleration: float = 0,
        acceleration_unit: Units = Units.NATIVE,
    ) -> None:
        """
        Input-shaped relative move using function for specific shaper mode.

        :param position: The amount to move.
        :param unit: The units for the position value.
        :param wait_until_idle: If true the command will hang until the device reaches idle state.
        :param acceleration: The acceleration for the move.
        :param acceleration_unit: The units for the acceleration value.
        """
        if isinstance(self.shaper, ZeroVibrationShaper):
            self._move_relative_decel(position, unit, wait_until_idle, acceleration, acceleration_unit)
        elif isinstance(self.shaper, ZeroVibrationStreamGenerator):
            self._move_relative_stream(position, unit, wait_until_idle, acceleration, acceleration_unit)
        else:
            raise TypeError("Invalid shaper type.")

    def _move_relative_decel(
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
        position_native = self.axis.settings.convert_to_native_units("pos", position, unit)
        accel_native = self.axis.settings.convert_to_native_units(
            "accel", acceleration, acceleration_unit
        )

        if acceleration == 0:  # Get the acceleration if it wasn't specified
            accel_native = self.axis.settings.get("accel", Units.NATIVE)

        position_mm = self.axis.settings.convert_from_native_units(
            "pos", position_native, Units.LENGTH_MILLIMETRES
        )
        accel_mm = self.axis.settings.convert_from_native_units(
            "accel", accel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )

        if isinstance(self.shaper, ZeroVibrationShaper):
            # Apply the input shaping with all values of the same units
            deceleration_mm, max_speed_mm = self.shaper.shape_trapezoidal_motion(
                position_mm, accel_mm,
                self.get_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND)
            )
        else:
            raise TypeError("_move_relative_decel method requires a shaper to be an instance of "
                            "ZeroVibrationShaper class.")


        # Check if the target deceleration is different from the current value
        deceleration_native = round(
            self.axis.settings.convert_to_native_units(
                "accel", deceleration_mm, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
            )
        )

        if self.axis.settings.get("motion.decelonly", Units.NATIVE) != deceleration_native:
            self.axis.settings.set("motion.decelonly", deceleration_native, Units.NATIVE)

        # Perform the move
        self.axis.move_relative(
            position,
            unit,
            wait_until_idle,
            max_speed_mm,
            Units.VELOCITY_MILLIMETRES_PER_SECOND,
            accel_mm,
            Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED,
        )

    def _move_relative_stream(
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
        position_native = self.axis.settings.convert_to_native_units("pos", position, unit)
        accel_native = self.axis.settings.convert_to_native_units(
            "accel", acceleration, acceleration_unit
        )

        if acceleration == 0:  # Get the acceleration if it wasn't specified
            accel_native = self.axis.settings.get("accel", Units.NATIVE)

        position_mm = self.axis.settings.convert_from_native_units(
            "pos", position_native, Units.LENGTH_MILLIMETRES
        )
        accel_mm = self.axis.settings.convert_from_native_units(
            "accel", accel_native, Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED
        )

        start_position = self.axis.get_position(Units.LENGTH_MILLIMETRES)

        if isinstance(self.shaper, ZeroVibrationStreamGenerator):
            stream_segments = self.shaper.shape_trapezoidal_motion(
                position_mm,
                accel_mm,
                accel_mm,
                self.get_max_speed_limit(Units.VELOCITY_MILLIMETRES_PER_SECOND),
            )
        else:
            raise TypeError("_move_relative_stream method requires a shaper to be an instance of "
                            "ZeroVibrationStreamGenerator class.")

        self.stream.disable()
        self.stream.setup_live(self.axis.axis_number)
        self.stream.cork()
        for segment in stream_segments:
            # Set acceleration making sure it is greater than zero by comparing 1 native accel unit
            if (
                self.axis.settings.convert_to_native_units(
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
                self.axis.settings.convert_to_native_units(
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
        current_position = self.axis.get_position(unit)
        self.move_relative(position - current_position, unit, wait_until_idle, acceleration, acceleration_unit)

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
        current_position = self.axis.get_position(Units.NATIVE)
        end_position = self.axis.settings.get("limit.max", Units.NATIVE)
        self.move_relative(end_position - current_position, Units.NATIVE, wait_until_idle, acceleration,
                           acceleration_unit)

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
        current_position = self.axis.get_position(Units.NATIVE)
        end_position = self.axis.settings.get("limit.min", Units.NATIVE)
        self.move_relative(end_position - current_position, Units.NATIVE, wait_until_idle, acceleration,
                           acceleration_unit)


# Example code for using the class.
if __name__ == "__main__":
    AXIS_INDEX = 1  # The Zaber axis index to test.
    RESONANT_FREQUENCY = 10  # Input shaping resonant frequency in Hz.
    DAMPING_RATIO = 0.1  # Input shaping damping ratio.

    with Connection.open_serial_port("COMx") as connection:
        # Get all the devices on the connection
        device_list = connection.detect_devices()
        print(f"Found {len(device_list)} devices.")

        if len(device_list) < 1:
            print("No devices, exiting.")
            sys.exit(0)

        device = device_list[0]  # Get the first device on the port
        axis = device.get_axis(
            AXIS_INDEX
        )  # Get the first axis from the device. This will become the ShapedAxis.
        shaped_axis = ShapedAxis(
            axis, RESONANT_FREQUENCY, DAMPING_RATIO, ShaperConfig(ShaperMode.DECEL)
        )  # Initialize the ShapedAxis class with the frequency and damping ratio

        if (
            not shaped_axis.axis.is_homed()
        ):  # The ShapedAxis has all the same functionality as the normal Axis class.
            shaped_axis.axis.home()

        print("Performing unshaped moves.")
        shaped_axis.axis.move_absolute(0, Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.2)
        shaped_axis.axis.move_relative(5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.2)
        shaped_axis.axis.move_relative(-5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(1)

        print("Shaping through changing deceleration.")

        print("Performing shaped moves.")
        shaped_axis.move_relative(5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.2)
        shaped_axis.move_relative(-5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(1)

        print("Performing shaped moves with speed limit.")
        shaped_axis.set_max_speed_limit(5, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        shaped_axis.move_relative(5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.2)
        shaped_axis.move_relative(-5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(1)

        print("Performing full travel shaped moves.")
        shaped_axis.reset_max_speed_limit()
        shaped_axis.move_max(True)
        time.sleep(0.2)
        shaped_axis.move_min(True)

        # Reset the deceleration to the original value in case the shaping algorithm changed it.
        # Deceleration is the only setting that may change.
        shaped_axis.reset_deceleration()

        print("Repeating shaped moves with ZV shaper using streams.")
        shaped_axis = ShapedAxis(
            axis,
            RESONANT_FREQUENCY,
            DAMPING_RATIO,
            ShaperConfig(ShaperMode.STREAM, shaper_type=ShaperType.ZV),
        )  # Re-initialize ShapedAxis class using streams to perform shaping and specify ZV shaper

        print("Performing shaped moves.")
        shaped_axis.move_relative(5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.2)
        shaped_axis.move_relative(-5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(1)

        print("Performing shaped moves with speed limit.")
        shaped_axis.set_max_speed_limit(5, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        shaped_axis.move_relative(5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.2)
        shaped_axis.move_relative(-5, Units.LENGTH_MILLIMETRES, True)
        time.sleep(1)

        print("Performing full travel shaped moves.")
        shaped_axis.reset_max_speed_limit()
        shaped_axis.move_max(True)
        time.sleep(0.2)
        shaped_axis.move_min(True)

        # Shaping with streams does not alter settings so no resetting is necessary

        print("Complete.")
