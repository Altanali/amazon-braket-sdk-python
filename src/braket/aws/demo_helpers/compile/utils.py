from pytket.circuit import Circuit


def _initialize_circ_copy(circ: Circuit) -> Circuit:
    """Return an empty circuit with the basic attributes of the input circuit:
    same qubits, bits, and phase.
    
    Args: 
        circ (Circuit): Pytket circuit whose qubits, bits, and phase to use
        for initialization. 
        
    Returns: 
        Circuit: An empty Pytket circuit with the same qubits, bits, and phase
        as the input circuit. 
    """
    new_circ = Circuit()
    for qubit in circ.qubits:
        new_circ.add_qubit(qubit, reject_dups=False)
    for bit in circ.bits:
        new_circ.add_bit(bit, reject_dups=False)
    new_circ.add_phase(circ.phase)
    return new_circ
