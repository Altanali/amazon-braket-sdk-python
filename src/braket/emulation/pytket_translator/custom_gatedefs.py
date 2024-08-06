from pytket.circuit import Circuit, CustomGateDef
from sympy import Symbol

def ms():
    ms_circ = Circuit(2)
    ms_circ.XXPhase(1.0 / 2, 0, 1)
    return CustomGateDef("ms(0, 0)", ms_circ, [])


def gpi():
    phi = Symbol("phi")
    circ = Circuit(1)
    circ.X(0)
    circ.Rz(2 * phi, 0)
    return CustomGateDef("gpi", circ, [phi])


def gpi2():
    phi = Symbol("phi")
    circ = Circuit(1)
    circ.Rz(-phi, 0)
    circ.Rx(0.5, 0)
    circ.Rz(phi, 0)
    return CustomGateDef("gpi2", circ, [phi])