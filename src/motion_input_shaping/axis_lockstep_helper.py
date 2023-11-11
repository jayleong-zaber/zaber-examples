import numpy as np
from zaber_motion import Units, CommandFailedException, UnitsAndLiterals
from zaber_motion.ascii import Axis, Lockstep


def get_axis_lockstep_group_number(axis: Axis) -> int:
    """
    Get lockstep group number that axis is part of on its device.

    Returns 0 if axis is not part of a group

    :param axis: Zaber Axis is to check
    """
    try:
        num_lockstep_groups_possible = axis.device.settings.get("lockstep.numgroups")
        for group_num in range(1, int(num_lockstep_groups_possible) + 1):
            axis_nums = axis.device.get_lockstep(group_num).get_axis_numbers()
            if axis.axis_number in axis_nums:
                print(f"Axis {axis.axis_number} is part of Lockstep group {group_num}.")
                return group_num
    except CommandFailedException:
        # Unable to get lockstep.numgroups settings meaning device is not capable of lockstep.
        pass

    return 0


def get_lockstep_axes(lockstep: Lockstep) -> list[Axis]:
    lockstep_axes = []
    for axis_number in lockstep.get_axis_numbers():
        lockstep_axes.append(lockstep.device.get_axis(axis_number))
    return lockstep_axes


class AxisLockstepSettings:
    def __init__(self, zaber_axis: Axis | Lockstep) -> None:
        self._axis = zaber_axis

        if isinstance(self._axis, Lockstep):
            # Get all axes in the lockstep group
            self._lockstep_axes = get_lockstep_axes(self._axis)

    def get(self, setting: str, unit: UnitsAndLiterals = Units.NATIVE) -> float:
        """
        Get setting value from axis returning the minimum value for axes if Lockstep.

        This returns the minimum for Lockstep because it is assuming that lower setting values are
        more limiting.

        :param setting: The name of setting
        :param unit: The values will be returned in these units.
        :return: A list of setting values
        """
        if isinstance(self._axis, Lockstep):
            values = []
            for axis in self._lockstep_axes:
                values.append(axis.settings.get(setting, unit))
            return min(values)

        return self._axis.settings.get(setting, unit)

    def get_all(self, setting: str, unit: UnitsAndLiterals = Units.NATIVE) -> list[float] | float:
        """
        Get setting values from axis or all axes in the lockstep group.

        :param setting: The name of setting
        :param unit: The values will be returned in these units.
        :return: A list of setting values
        """
        if isinstance(self._axis, Lockstep):
            values = []
            for axis in self._lockstep_axes:
                values.append(axis.settings.get(setting, unit))
            return values

        return self._axis.settings.get(setting, unit)

    def set(
        self, setting: str, value: list[float] | float, unit: UnitsAndLiterals = Units.NATIVE
    ) -> None:
        """
        Set settings for all axes in the lockstep group.

        :param setting: The name of setting
        :param value: List of values to apply as setting for each axis or a single value to
        apply to all
        :param unit: The values will be returned in these units.
        """
        if isinstance(self._axis, Lockstep):
            if isinstance(value, list):
                if len(value) != len(self._lockstep_axes):
                    raise ValueError(
                        "Length of setting values does not match the number of axes. "
                        "The list must either be a single value or match the number of axes."
                    )
                for n, axis in enumerate(self._lockstep_axes):
                    axis.settings.set(setting, value[n], unit)
            else:
                for n, axis in enumerate(self._lockstep_axes):
                    axis.settings.set(setting, value, unit)
        else:
            if isinstance(value, list):
                raise TypeError("List of settings values provided for a single Axis.")
            self._axis.settings.set(setting, value, unit)

    def convert_to_native_units(self, setting: str, value: float, unit: UnitsAndLiterals) -> float:
        """Convert to native units using first axis if Lockstep"""
        if isinstance(self._axis, Lockstep):
            return self._lockstep_axes[0].settings.convert_to_native_units(setting, value, unit)

        return self._axis.settings.convert_to_native_units(setting, value, unit)

    def convert_from_native_units(
        self, setting: str, value: float, unit: UnitsAndLiterals
    ) -> float:
        """Convert from native units using first axis if Lockstep"""
        if isinstance(self._axis, Lockstep):
            return self._lockstep_axes[0].settings.convert_from_native_units(setting, value, unit)

        return self._axis.settings.convert_from_native_units(setting, value, unit)


class AxisLockstep:
    axis: Axis | Lockstep

    def __init__(self, zaber_axis: Axis | Lockstep) -> None:
        if isinstance(zaber_axis, Axis):
            # Sanity check if the passed axis has a higher number than the number of axes on the
            # device.
            if zaber_axis.axis_number > zaber_axis.device.axis_count or zaber_axis is None:
                raise TypeError("Invalid Axis class was used to initialized class.")

            # Check if the Axis is part of a lockstep group
            lockstep_group_number = get_axis_lockstep_group_number(zaber_axis)
            if lockstep_group_number > 0:
                # Use Lockstep instead
                print("Initializing as Lockstep group.")
                self.axis = zaber_axis.device.get_lockstep(lockstep_group_number)
            else:
                self.axis = zaber_axis

        elif isinstance(zaber_axis, Lockstep):
            # Sanity check if the passed lockstep group number exceeds than the number of
            # lockstep groups on the device.
            if (
                zaber_axis.lockstep_group_id > zaber_axis.device.settings.get("lockstep.numgroups")
                or zaber_axis is None
            ):
                raise TypeError("Invalid Lockstep class was used to initialized class.")

            self.axis = zaber_axis
        else:
            raise TypeError("Invalid class instance was used to initialized class.")

        self.settings = AxisLockstepSettings(self.axis)

        if isinstance(self.axis, Lockstep):
            self._lockstep_axes = get_lockstep_axes(self.axis)

    def is_homed(self) -> bool:
        """Check if all axes in lockstep group are homed."""
        if isinstance(self.axis, Lockstep):
            for axis in get_lockstep_axes(self.axis):
                if not axis.is_homed():
                    return False
            return True

        return self.axis.is_homed()

    def get_position(self, unit: Units = Units.NATIVE) -> float:
        """
        Get positions from Axis or Lockstep.

        :param unit: The positions will be returned in these units.
        """
        return self.axis.get_position(unit)

    def get_all_position(self, unit: Units = Units.NATIVE) -> float | list[float]:
        """
        Get positions from all axes in the Lockstep group or position of the Axis.

        :param unit: The positions will be returned in these units.
        """
        if isinstance(self.axis, Lockstep):
            positions = []
            for axis in self._lockstep_axes:
                positions.append(axis.get_position(unit))
            return positions

        return self.axis.get_position(unit)

    def get_max_relative_move(self) -> float:
        current_position = self.get_all_position(Units.NATIVE)
        end_position = self.settings.get_all("limit.max", Units.NATIVE)

        # Use numpy subtract in case of Lockstep and positions being a list
        # Move will be positive so find min relative move
        return float(np.min(np.subtract(end_position, current_position)))

    def get_min_relative_move(self) -> float:
        current_position = self.get_all_position(Units.NATIVE)
        end_position = self.settings.get_all("limit.min", Units.NATIVE)

        # Use numpy subtract in case of Lockstep and positions being a list
        # Move will be negative so find max relative move
        return float(np.max(np.subtract(end_position, current_position)))
