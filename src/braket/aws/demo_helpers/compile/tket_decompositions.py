from pytket.circuit import Circuit
import braket.emulation.pytket_translator.custom_gatedefs as custom_gatedefs

def cx_to_ms() -> Circuit:
    """CX to MS decomposition, using Ry and Rx for single qubit corrections."""
    circ = Circuit(2)
    circ.Ry(0.5, 0)
    circ.add_custom_gate(custom_gatedefs.ms(), [], [0, 1])
    circ.Ry(3.5, 0)
    circ.Rx(3.5, 1)
    circ.Rx(3.5, 0)
    circ.Ry(3.5, 0)
    circ.Rx(0.5, 0)
    return circ


def tk1_to_zxz(a, b, c) -> Circuit:
    """The classic Rz, Rx, Rz decomposition for TK1, which essentially defines the TK1 gate."""
    circ = Circuit(1)
    circ.Rz(c, 0).Rx(b, 0).Rz(a, 0)
    return circ


def tk1_to_gpi(a, b, c) -> Circuit:
    """Decomposition from TK1 to GPi and GPi2 gates."""
    circ = Circuit(1)
    circ.add_custom_gate(custom_gatedefs.gpi2(), [1.5 - c], [0])
    circ.add_custom_gate(custom_gatedefs.gpi(), [0.5 * (a - b - c + 1.0)], [0])
    circ.add_custom_gate(custom_gatedefs.gpi2(), [a - 0.5], [0])
    return circ