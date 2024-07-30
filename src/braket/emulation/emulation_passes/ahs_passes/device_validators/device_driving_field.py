from typing import Dict

from pydantic.v1.class_validators import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.driving_field import (
    DrivingFieldValidator,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.field_validator_util import (
    validate_max_absolute_slope,
    validate_time_precision,
    validate_time_separation,
    validate_value_precision,
    validate_value_range_with_warning,
)
from braket.analog_hamiltonian_simulator.rydberg.validators.physical_field import PhysicalField
from braket.emulation.emulation_passes.ahs_passes.device_capabilities_constants import (
    DeviceCapabilitiesConstants,
)


class DeviceDrivingFieldValidator(DrivingFieldValidator):
    capabilities: DeviceCapabilitiesConstants

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_start_and_end_values(cls, values: Dict) -> Dict:
        """
        Checks that the amplitudes in the driving field time series begin and end at 0.0.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If either the starting or ending amplitude are nonzero.
        """
        amplitude = values["amplitude"]
        time_series = amplitude["time_series"]
        time_series_values = time_series["values"]
        if time_series_values:
            start_value, end_value = time_series_values[0], time_series_values[-1]
            if start_value != 0 or end_value != 0:
                raise ValueError(
                    f"The values of the Rabi frequency at the first and last time points are "
                    f"{start_value}, {end_value}; they both must be nonzero."
                )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_time_precision_is_correct(cls, values: Dict) -> Dict:
        """
        Checks that the precision of the amplitude times points are within the GLOBAL_TIME_PRECISION
        of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the amplitude times points in the driving field have precision
            greater than that supported in the device capabilities.
        """
        amplitude = values["amplitude"]
        capabilities = values["capabilities"]
        amplitude_obj = PhysicalField.parse_obj(amplitude)
        validate_time_precision(
            amplitude_obj.time_series.times, capabilities.GLOBAL_TIME_PRECISION, "amplitude"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_timepoint_not_too_close(cls, values: Dict) -> Dict:
        """
        Checks that data points in the time series are separated in time by at least the
        GLOBAL_MIN_TIME_SEPARATION of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the data points in the driving field time series do not differ
            in time by at least the GLOBAL_MIN_TIME_SEPARATION of the device capabilities.
        """
        amplitude = values["amplitude"]
        capabilities = values["capabilities"]
        validate_time_separation(
            amplitude["time_series"]["times"], capabilities.GLOBAL_MIN_TIME_SEPARATION, "amplitude"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_value_precision_is_correct(cls, values: Dict) -> Dict:
        """
        Checks that the precision of the amplitude values are within the GLOBAL_VALUE_PRECISION
        of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the amplitude values in the driving field have precision
            greater than that supported in the device capabilities.
        """
        amplitude = values["amplitude"]
        capabilities = values["capabilities"]
        amplitude_obj = PhysicalField.parse_obj(amplitude)
        validate_value_precision(
            amplitude_obj.time_series.values,
            capabilities.GLOBAL_AMPLITUDE_VALUE_PRECISION,
            "amplitude",
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def amplitude_slopes_not_too_steep(cls, values: Dict) -> Dict:
        """
        Checks that the max absolute slope in the amplitude time series does not exceed the
        GLOBAL_AMPLITUDE_SLOPE_MAX of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If the slope between any points in the amplitude time series exceeds the
            GLOBAL_AMPLITUDE_SLOPE_MAX of the device capabilities.
        """
        amplitude = values["amplitude"]
        capabilities = values["capabilities"]
        amplitude_times = amplitude["time_series"]["times"]
        amplitude_values = amplitude["time_series"]["values"]
        if amplitude_times and amplitude_values:
            validate_max_absolute_slope(
                amplitude_times,
                amplitude_values,
                capabilities.GLOBAL_AMPLITUDE_SLOPE_MAX,
                "amplitude",
            )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def phase_time_precision_is_correct(cls, values: Dict) -> Dict:
        """
        Checks that the precision of the phase times points are within the GLOBAL_TIME_PRECISION
        of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the phase times points in the driving field have precision
            greater than that supported in the device capabilities.
        """
        phase = values["phase"]
        capabilities = values["capabilities"]
        phase_obj = PhysicalField.parse_obj(phase)
        validate_time_precision(
            phase_obj.time_series.times, capabilities.GLOBAL_TIME_PRECISION, "phase"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def phase_timepoint_not_too_close(cls, values: Dict) -> Dict:
        """
        Checks that the phase time points of the driving field differ by at least the
        GLOBAL_MIN_SEPARATION of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any pair of phase time points do not differ by more than the
            GLOBAL_MIN_SEPARATION of the device capabilities.
        """
        phase = values["phase"]
        capabilities = values["capabilities"]
        validate_time_separation(
            phase["time_series"]["times"], capabilities.GLOBAL_MIN_TIME_SEPARATION, "phase"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def phase_values_start_with_0(cls, values: Dict) -> Dict:
        """
        Checks that the phase values of the driving field start with 0.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If the phase values of the driving field do not start with 0.
        """
        phase = values["phase"]
        phase_values = phase["time_series"]["values"]
        if phase_values:
            if phase_values[0] != 0:
                raise ValueError(
                    f"The first value of of driving field phase is {phase_values[0]}; it must be 0."
                )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def phase_values_within_range(cls, values: Dict) -> Dict:
        """
        Checks that the phase values of the driving field are in the range
        [GLOBAL_PHASE_VALUE_MIN, GLOBAL_PHASE_VALUE_MAX] of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the phase values are outside the range
            [GLOBAL_PHASE_VALUE_MIN, GLOBAL_PHASE_VALUE_MAX] of the device capabilities.
        """
        phase = values["phase"]
        capabilities = values["capabilities"]
        validate_value_range_with_warning(
            phase["time_series"]["values"],
            capabilities.GLOBAL_PHASE_VALUE_MIN,
            capabilities.GLOBAL_PHASE_VALUE_MAX,
            "phase",
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def phase_value_precision_is_correct(cls, values: Dict) -> Dict:
        """
        Checks that the precision of the phase values are within the GLOBAL_VALUE_PRECISION
        of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the phase values in the driving field have precision
            greater than that supported in the device capabilities.
        """
        phase = values["phase"]
        capabilities = values["capabilities"]
        phase_obj = PhysicalField.parse_obj(phase)
        validate_value_precision(
            phase_obj.time_series.values, capabilities.GLOBAL_PHASE_VALUE_PRECISION, "phase"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def detuning_time_precision_is_correct(cls, values: Dict) -> Dict:
        """
        Checks that the precision of the detuning times points are within the GLOBAL_TIME_PRECISION
        of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any of the detuning times points in the driving field have precision
            greater than that supported in the device capabilities.
        """
        detuning = values["detuning"]
        capabilities = values["capabilities"]
        detuning_obj = PhysicalField.parse_obj(detuning)
        validate_time_precision(
            detuning_obj.time_series.times, capabilities.GLOBAL_TIME_PRECISION, "detuning"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def detuning_timepoint_not_too_close(cls, values: Dict) -> Dict:
        """
        Checks that the detuning time points of the driving field differ by at least the
        GLOBAL_MIN_SEPARATION of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If any pair of detuning time points do not differ by more than the
            GLOBAL_MIN_SEPARATION of the device capabilities.
        """
        detuning = values["detuning"]
        capabilities = values["capabilities"]
        validate_time_separation(
            detuning["time_series"]["times"], capabilities.GLOBAL_MIN_TIME_SEPARATION, "detuning"
        )
        return values

    @root_validator(pre=True, skip_on_failure=True)
    def detuning_slopes_not_too_steep(cls, values: Dict) -> Dict:
        """
        Checks that the max absolute slope in the detuning time series does not exceed the
        GLOBAL_AMPLITUDE_SLOPE_MAX of the device capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the driving field data
                to validate.
        Returns:
            Dict: Unmodified driving field data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If the slope between any points in the detuning time series exceeds the
            GLOBAL_AMPLITUDE_SLOPE_MAX of the device capabilities.
        """
        detuning = values["detuning"]
        capabilities = values["capabilities"]
        detuning_times = detuning["time_series"]["times"]
        detuning_values = detuning["time_series"]["values"]
        if detuning_times and detuning_values:
            validate_max_absolute_slope(
                detuning_times,
                detuning_values,
                capabilities.GLOBAL_DETUNING_SLOPE_MAX,
                "detuning",
            )
        return values
