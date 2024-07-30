from typing import Union

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.emulation.emulation_passes import ValidationPass
from braket.emulation.emulation_passes.ahs_passes.device_capabilities_constants import (
    DeviceCapabilitiesConstants,
)
from braket.emulation.emulation_passes.ahs_passes.device_ir_validator import validate_program
from braket.ir.ahs import Program as AHSProgram


class AhsValidator(ValidationPass):
    def __init__(self, device_capabilities_constants: DeviceCapabilitiesConstants):
        self._capabilities = device_capabilities_constants

    def validate(self, program: Union[AnalogHamiltonianSimulation, AHSProgram]) -> None:
        """
        Validates whether or not a AHS program is runnable on the AHS device properties defined
        in device_capabilities_constants.

        Args:
            program (Union[AnalogHamiltonianSimulation, AHSProgram]): The input AHS program
                to validate.
        """
        if isinstance(program, AnalogHamiltonianSimulation):
            program = program.to_ir()
        validate_program(program, self._capabilities)

    def __eq__(self, other: ValidationPass) -> bool:
        return isinstance(other, AhsValidator) and other._capabilities == self._capabilities
