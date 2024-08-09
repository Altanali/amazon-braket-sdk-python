import math

from braket.circuits import Circuit, FreeParameter, gates
from braket.circuits.compiler_directives import EndVerbatimBox, StartVerbatimBox
from braket.emulation.emulation_passes import ValidationPass


class RigettiRxArgsValidator(ValidationPass[Circuit]):
    _ALLOWED_ANGLES = [math.pi, -math.pi, math.pi / 2, -math.pi / 2]

    def validate(self, program: Circuit) -> None:
        """
        Validates that the only angles used in a verbatim Rx gate are -pi, pi, -pi/2, or pi/2.

        Args:
            program (Circuit): The braket circuit to validate.

        Raises:
            ValueError: If an Rx gate used in a verbatim subcircuit uses an unallowed angle.
        """
        idx = 0
        while idx < len(program.instructions):
            instruction = program.instructions[idx]
            if isinstance(instruction.operator, StartVerbatimBox):
                idx += 1
                while idx < len(program.instructions) and not isinstance(
                    program.instructions[idx].operator, EndVerbatimBox
                ):
                    instruction = program.instructions[idx]
                    if isinstance(instruction.operator, gates.Rx):
                        angle = instruction.operator.angle
                        if not isinstance(angle, FreeParameter):
                            if angle not in self._ALLOWED_ANGLES:
                                raise ValueError(
                                    f"Invalid RX angle '{angle}' with verbatim usage on Rigetti"
                                    "device. Valid angles are (-π, -π/2, π/2, π).",
                                )
                    idx += 1
            idx += 1

    def __eq__(self, other):
        return isinstance(other, RigettiRxArgsValidator)
