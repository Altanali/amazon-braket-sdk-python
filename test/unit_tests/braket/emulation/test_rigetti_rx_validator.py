import numpy as np
import pytest

from braket.circuits import Circuit
from braket.emulation.emulation_passes.gate_device_passes import RigettiRxArgsValidator


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(),
        Circuit().rx(0, np.pi / 2),
        Circuit().rx(0, np.pi / 7),
        Circuit()
        .add_verbatim_box(Circuit().rx(2, np.pi).rx(0, -np.pi).rx(3, np.pi / 2).rx(1, -np.pi / 2))
        .ry(4, np.pi / 9),
    ],
)
def test_valid_circuits(circuit):
    try:
        RigettiRxArgsValidator().validate(circuit)
    except ValueError as e:
        pytest.fail("Failed Valid Rigetti RX Args Validation: " + repr(e))


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit().add_verbatim_box(Circuit().rx(0, np.pi / 2).rx(2, np.pi / 4)),
        Circuit().ry(0, np.pi / 8).add_verbatim_box(Circuit().rx(4, 0)),
        Circuit()
        .i(range(5))
        .add_verbatim_box(Circuit().rx(range(5), -np.pi / 2))
        .add_verbatim_box(Circuit().rx(0, np.pi / 2 + 1e-4)),
    ],
)
def test_invalid_circuits(circuit):
    with pytest.raises(ValueError):
        RigettiRxArgsValidator().validate(circuit)
