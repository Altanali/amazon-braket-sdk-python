from decimal import Decimal

import pytest

from braket.emulators.emulator_passes.ahs_passes.device_capabilities_constants import (
    DeviceCapabilitiesConstants,
)
from braket.ir.ahs.program_v1 import Program


@pytest.fixture
def non_local_capabilities_constants():
    capabilities_dict = {
        "BOUNDING_BOX_SIZE_X": Decimal("0.000075"),
        "BOUNDING_BOX_SIZE_Y": Decimal("0.000076"),
        "DIMENSIONS": 2,
        "GLOBAL_AMPLITUDE_SLOPE_MAX": Decimal("250000000000000"),
        "GLOBAL_AMPLITUDE_VALUE_MAX": Decimal("15800000.0"),
        "GLOBAL_AMPLITUDE_VALUE_MIN": Decimal("0.0"),
        "GLOBAL_AMPLITUDE_VALUE_PRECISION": Decimal("400.0"),
        "GLOBAL_DETUNING_SLOPE_MAX": Decimal("250000000000000"),
        "GLOBAL_DETUNING_VALUE_MAX": Decimal("125000000.0"),
        "GLOBAL_DETUNING_VALUE_MIN": Decimal("-125000000.0"),
        "GLOBAL_DETUNING_VALUE_PRECISION": Decimal("0.2"),
        "GLOBAL_MIN_TIME_SEPARATION": Decimal("1E-8"),
        "GLOBAL_PHASE_VALUE_MAX": Decimal("99.0"),
        "GLOBAL_PHASE_VALUE_MIN": Decimal("-99.0"),
        "GLOBAL_PHASE_VALUE_PRECISION": Decimal("5E-7"),
        "GLOBAL_TIME_PRECISION": Decimal("1E-9"),
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX": None,
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN": None,
        "LOCAL_MAGNITUDE_SLOPE_MAX": None,
        "LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES": None,
        "LOCAL_MIN_TIME_SEPARATION": None,
        "LOCAL_RYDBERG_CAPABILITIES": False,
        "LOCAL_TIME_PRECISION": None,
        "MAGNITUDE_PATTERN_VALUE_MAX": None,
        "MAGNITUDE_PATTERN_VALUE_MIN": None,
        "MAX_FILLED_SITES": 4,
        "MAX_NET_DETUNING": None,
        "MAX_SITES": 8,
        "MAX_TIME": Decimal("0.000004"),
        "MIN_DISTANCE": Decimal("0.000004"),
        "MIN_ROW_DISTANCE": Decimal("0.000004"),
        "SITE_PRECISION": Decimal("1E-7"),
    }
    return DeviceCapabilitiesConstants(**capabilities_dict)


@pytest.fixture
def capabilities_with_local_rydberg(non_local_capabilities_constants):
    local_rydberg_constants = {
        "LOCAL_RYDBERG_CAPABILITIES": True,
        "LOCAL_TIME_PRECISION": Decimal("1e-9"),
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX": Decimal("125000000.0"),
        "LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN": Decimal("0.0"),
        "LOCAL_MAGNITUDE_SLOPE_MAX": Decimal("1256600000000000.0"),
        "LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES": Decimal("5e-06"),
        "LOCAL_MIN_TIME_SEPARATION": Decimal("1E-8"),
        "LOCAL_MAX_NONZERO_PATTERN_VALUES": 200,
        "MAGNITUDE_PATTERN_VALUE_MAX": Decimal("1.0"),
        "MAGNITUDE_PATTERN_VALUE_MIN": Decimal("0.0"),
    }
    capabilities_dict = non_local_capabilities_constants.dict()
    capabilities_dict.update(local_rydberg_constants)
    return DeviceCapabilitiesConstants(**capabilities_dict)


@pytest.fixture
def program_data():
    data = {
        "setup": {
            "ahs_register": {
                "sites": [[0, 0], [0, 4e-6], [5e-6, 0], [5e-6, 4e-6]],
                "filling": [1, 0, 1, 0],
            }
        },
        "hamiltonian": {
            "drivingFields": [
                {
                    "amplitude": {
                        "pattern": "uniform",
                        "time_series": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [0, 12566400.0, 12566400.0, 0],
                        },
                    },
                    "phase": {
                        "pattern": "uniform",
                        "time_series": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [0, 0, -16.0832, -16.0832],
                        },
                    },
                    "detuning": {
                        "pattern": "uniform",
                        "time_series": {
                            "times": [0, 1e-07, 3.9e-06, 4e-06],
                            "values": [-125000000, -125000000, 125000000, 125000000],
                        },
                    },
                }
            ],
            "localDetuning": [
                {
                    "magnitude": {
                        "time_series": {"times": [0, 4e-6], "values": [0, 0]},
                        "pattern": [0.0, 1.0, 0.5, 0.0],
                    }
                }
            ],
        },
    }
    return Program.parse_obj(data)
