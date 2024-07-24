from typing import Union

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.circuits.circuit import Circuit
from braket.emulators.emulator_passes import EmulatorCriterion
from braket.emulators.emulator_passes.ahs_passes.device_capabilities_constants import (
    DeviceCapabilitiesConstants,
)
from braket.emulators.emulator_passes.ahs_passes.device_validators import validate_program
from braket.ir.ahs import Program as AHSProgram


class AhsCriterion(EmulatorCriterion):
    def __init__(self, device_capabilities_constants: DeviceCapabilitiesConstants):
        self._capabilities = device_capabilities_constants

    def validate(self, program: Union[AnalogHamiltonianSimulation, AHSProgram]) -> None:
        if isinstance(program, AnalogHamiltonianSimulation):
            program = program.to_ir()
        validate_program(program, self._capabilities)

    def __eq__(self, other: EmulatorCriterion) -> bool:
        return isinstance(other, AhsCriterion) and other._capabilities == self._capabilities
