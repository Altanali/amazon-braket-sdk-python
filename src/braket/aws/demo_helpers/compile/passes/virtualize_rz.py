from collections import defaultdict
from pytket.passes import CustomPass
from pytket.circuit import Circuit, OpType
from braket.aws.demo_helpers.compile.utils import (
_initialize_circ_copy
)


ALLOWED_GATES = {OpType.CZ, OpType.Rz, OpType.PhasedX, OpType.Barrier, OpType.Measure}

def get_virtualize_rz_pass():
    return CustomPass(_virtualize_rz)

def _virtualize_rz(circ: Circuit) -> Circuit:
    """Return a new circuit which enacts the same operations as the input circuit, up
    to final Rz rotations.

    Rz gates are commuted to the end of the circuit, where they are dropped.
    Circuit-final Rzs do not affect measurement outcomes.

    Warning: The final statevector will not be equivalent, but will yield equivalent
    measurements.

    Args:
        circ (Circuit): A circuit with gates in the ALLOWED_GATES set.

    Returns:
        Circuit: A circuit where every single qubit rotation is a PRx (PhasedX) gate.

    Raises:
        ValueError: If an unsupported operation is encountered. Rebase your circuit
            onto {CZ, Rz, PhasedX} before running this pass.
    """
    new_circ = _initialize_circ_copy(circ)
    # This dictionary tracks the ongoing phase corrections to apply to PRx gates,
    # by commuting Rz gates past them
    phase_per_qubit = defaultdict(int)

    for command in circ.get_commands():
        if command.op.type == OpType.Rz:
            prev_phase = phase_per_qubit[command.args[0]]
            new_phase = prev_phase + command.op.params[0]
            phase_per_qubit[command.args[0]] = new_phase

        elif command.op.type == OpType.PhasedX:
            qubits = command.args
            alpha, beta = command.op.params
            beta = beta + phase_per_qubit[qubits[0]]  # PRx takes one qubit only
            new_circ.add_gate(command.op.type, [alpha, beta], qubits)

        elif command.op.type == OpType.Barrier:
            new_circ.add_barrier(command.args, data=command.op.data)
        else:
            if command.op.type not in ALLOWED_GATES:
                raise ValueError(
                    f"Unsupported operation ({command.op.type}) encountered while virtualizing Rz gates."
                )
            new_circ.add_gate(command.op, command.args)

    return new_circ




