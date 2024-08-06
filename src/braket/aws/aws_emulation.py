from collections.abc import Iterable
from functools import singledispatch
from typing import Union

from networkx import DiGraph

from braket.device_schema import DeviceActionType, DeviceCapabilities
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.iqm import IqmDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.rigetti import RigettiDeviceCapabilities
from braket.emulation.emulation_passes.ahs_passes import AhsValidator, AhsNoise, AhsNoiseData
from braket.emulation.emulation_passes.ahs_passes.device_capabilities_constants import (
    DeviceCapabilitiesConstants,
)
from braket.passes import NoOpPass
from braket.emulation.emulation_passes.gate_device_passes import (
    ConnectivityValidator,
    GateConnectivityValidator,
    GateValidator,
    QubitCountValidator,
    LexiRoutingPass, 
    TketCompilerPass
)

from pytket.transform import Transform
from pytket.passes import (
    RemoveRedundancies, 
    SimplifyInitial, 
    auto_rebase_pass,
    SquashRzPhasedX, 
    auto_squash_pass, 
    RebaseCustom, 
    SquashTK1
)
from pytket.circuit import OpType
from pytket.architecture import Architecture
from braket.aws.demo_helpers.compile.passes import get_virtualize_rz_pass
from braket.aws.demo_helpers.compile import tket_decompositions


def qubit_count_validator(properties: DeviceCapabilities) -> QubitCountValidator:
    """
    Create a QubitCountValidator pass which checks that the number of qubits used in a program does
    not exceed the number of qubits allowed by a QPU, as defined in the device properties.

    Args:
        properties (DeviceCapabilities): QPU Device Capabilities object with a
            QHP-specific schema.

    Returns:
        QubitCountValidator: An emulator pass that checks that the number of qubits used in a
        program does not exceed that of the max qubit count on the device.
    """
    qubit_count = properties.paradigm.qubitCount
    return QubitCountValidator(qubit_count)


def gate_validator(properties: DeviceCapabilities) -> GateValidator:
    """
    Create a GateValidator pass which defines what supported and native gates are allowed in a
    program based on the provided device properties.

    Args:
        properties (DeviceCapabilities): QPU Device Capabilities object with a
            QHP-specific schema.

    Returns:
        GateValidator: An emulator pass that checks that a circuit only uses supported gates and
        verbatim circuits only use native gates.
    """

    supported_gates = properties.action[DeviceActionType.OPENQASM].supportedOperations
    native_gates = properties.paradigm.nativeGateSet

    return GateValidator(supported_gates=supported_gates, native_gates=native_gates)


def connectivity_validator(
    properties: DeviceCapabilities, connectivity_graph: DiGraph
) -> ConnectivityValidator:
    """
    Creates a ConnectivityValidator pass which validates that two-qubit gates are applied to
    connected qubits based on this device's connectivity graph.

    Args:
        properties (DeviceCapabilities): QPU Device Capabilities object with a
            QHP-specific schema.

        connectivity_graph (DiGraph): Connectivity graph for this device.

    Returns:
        ConnectivityValidator: An emulator pass that checks that a circuit only applies two-qubit
        gates to connected qubits on the device.
    """

    return _connectivity_validator(properties, connectivity_graph)


@singledispatch
def _connectivity_validator(
    properties: DeviceCapabilities, connectivity_graph: DiGraph
) -> ConnectivityValidator:

    connectivity_validator = ConnectivityValidator(connectivity_graph)
    return connectivity_validator


@_connectivity_validator.register(IqmDeviceCapabilities)
def _(properties: IqmDeviceCapabilities, connectivity_graph: DiGraph) -> ConnectivityValidator:
    """
    IQM qubit connectivity is undirected but the directed graph that represents qubit connectivity
    does not include back-edges. Thus, we must explicitly introduce back edges before creating
    the ConnectivityValidator for an IQM device.
    """
    connectivity_graph = connectivity_graph.copy()
    for edge in connectivity_graph.edges:
        connectivity_graph.add_edge(edge[1], edge[0])
    return ConnectivityValidator(connectivity_graph)


def gate_connectivity_validator(
    properties: DeviceCapabilities, connectivity_graph: DiGraph
) -> GateConnectivityValidator:
    return _gate_connectivity_validator(properties, connectivity_graph)


@singledispatch
def _gate_connectivity_validator(
    properties: DeviceCapabilities, connectivity_graph: DiGraph
) -> GateConnectivityValidator:
    raise NotImplementedError


@_gate_connectivity_validator.register(IqmDeviceCapabilities)
@_gate_connectivity_validator.register(RigettiDeviceCapabilities)
def _(
    properties: RigettiDeviceCapabilities, connectivity_graph: DiGraph
) -> GateConnectivityValidator:
    """
    Both IQM and Rigetti have undirected connectivity graphs; Rigetti device capabilities
    provide back edges, but the calibration data only provides edges in one direction.
    Additionally, IQM does not provide back edges in its connectivity_graph (nor is this
    resolved manually by AwsDevice at the moment).
    """
    gate_connectivity_graph = connectivity_graph.copy()
    edge_properties = properties.standardized.twoQubitProperties
    for u, v in gate_connectivity_graph.edges:
        edge_key = "-".join([str(qubit) for qubit in (u, v)])
        edge_property = edge_properties.get(edge_key)

        # Check that the QHP provided calibration data for this edge.
        if not edge_property:
            gate_connectivity_graph[u][v]["supported_gates"] = set()
            continue
        edge_supported_gates = _get_qpu_gate_translations(
            properties, [property.gateName for property in edge_property.twoQubitGateFidelity]
        )
        gate_connectivity_graph[u][v]["supported_gates"] = set(edge_supported_gates)

    # Add the reversed edge to ensure gates can be applied
    # in both directions for a given qubit pair.
    for u, v in gate_connectivity_graph.edges:
        if (v, u) not in gate_connectivity_graph.edges or gate_connectivity_graph[v][u].get(
            "supported_gates"
        ) in [None, set()]:
            gate_connectivity_graph.add_edge(
                v, u, supported_gates=set(gate_connectivity_graph[u][v]["supported_gates"])
            )

    return GateConnectivityValidator(gate_connectivity_graph)


@_gate_connectivity_validator.register(IonqDeviceCapabilities)
def _(properties: IonqDeviceCapabilities, connectivity_graph: DiGraph) -> GateConnectivityValidator:
    """
    Qubits in IonQ's trapped ion devices are all fully connected with identical
    gate-pair capabilities. IonQ does not expliclty provide a set of edges for
    gate connectivity between qubit pairs in their trapped ion QPUs.
    We extrapolate gate connectivity across all possible qubit edge pairs.
    """
    gate_connectivity_graph = connectivity_graph.copy()
    native_gates = _get_qpu_gate_translations(properties, properties.paradigm.nativeGateSet)

    for edge in gate_connectivity_graph.edges:
        gate_connectivity_graph[edge[0]][edge[1]]["supported_gates"] = set(native_gates)

    return GateConnectivityValidator(gate_connectivity_graph)


def _get_qpu_gate_translations(
    properties: DeviceCapabilities, gate_name: Union[str, Iterable[str]]
) -> Union[str, list[str]]:
    """Returns the translated gate name(s) for a given QPU device capabilities schema type
        and gate name(s).

    Args:
        properties (DeviceCapabilities): Device capabilities object based on a
            device-specific schema.
        gate_name (Union[str, Iterable[str]]): The name(s) of the gate(s). If gate_name is a list
            of string gate names, this function attempts to retrieve translations of all the gate
            names.

    Returns:
        Union[str, list[str]]: The translated gate name(s)
    """
    if isinstance(gate_name, str):
        return _get_qpu_gate_translation(properties, gate_name)
    else:
        return [_get_qpu_gate_translation(properties, name) for name in gate_name]


@singledispatch
def _get_qpu_gate_translation(properties: DeviceCapabilities, gate_name: str) -> str:
    """Returns the translated gate name for a given QPU ARN and gate name.

    Args:
        properties (DeviceCapabilities): QPU Device Capabilities object with a
            QHP-specific schema.
        gate_name (str): The name of the gate

    Returns:
        str: The translated gate name
    """
    return gate_name


@_get_qpu_gate_translation.register(RigettiDeviceCapabilities)
def _(properties: RigettiDeviceCapabilities, gate_name: str) -> str:
    translations = {"CPHASE": "CPhaseShift"}
    return translations.get(gate_name, gate_name)


@_get_qpu_gate_translation.register(IonqDeviceCapabilities)
def _(properties: IonqDeviceCapabilities, gate_name: str) -> str:
    translations = {"GPI": "GPi", "GPI2": "GPi2"}
    return translations.get(gate_name, gate_name)


def ahs_criterion(properties: DeviceCapabilities) -> AhsValidator:
    """
    Creates an AHS program validation pass using the input QPU device capabilities.

    Args:
        properties (DeviceCapabilities): Device capabilities of a QPU device that supports
            Analog Hamiltonian Simulation Programs.

    Returns:
        AhsValidator: An emulator pass that validates whether or not an AHS program is able to run
        on the target device.
    """
    device_capabilities_constants = _get_ahs_device_capabilities(properties)
    return AhsValidator(device_capabilities_constants)


@singledispatch
def _get_ahs_device_capabilities(properties: DeviceCapabilities) -> DeviceCapabilitiesConstants:
    raise NotImplementedError(
        "AHS Device Capabilities Constants cannot be created"
        f"using capabilities type ({type(properties)})"
    )


@_get_ahs_device_capabilities.register(QueraDeviceCapabilities)
def _(properties: QueraDeviceCapabilities) -> DeviceCapabilitiesConstants:
    properties = properties.dict()
    capabilities = dict()
    lattice = properties["paradigm"]["lattice"]
    capabilities["MAX_SITES"] = lattice["geometry"]["numberSitesMax"]
    capabilities["MIN_DISTANCE"] = lattice["geometry"]["spacingRadialMin"]
    capabilities["MIN_ROW_DISTANCE"] = lattice["geometry"]["spacingVerticalMin"]
    capabilities["SITE_PRECISION"] = lattice["geometry"]["positionResolution"]
    capabilities["BOUNDING_BOX_SIZE_X"] = lattice["area"]["width"]
    capabilities["BOUNDING_BOX_SIZE_Y"] = lattice["area"]["height"]
    capabilities["MAX_FILLED_SITES"] = properties["paradigm"]["qubitCount"]

    rydberg = properties["paradigm"]["rydberg"]
    rydberg_global = rydberg["rydbergGlobal"]
    capabilities["MIN_TIME"] = rydberg_global["timeMin"]
    capabilities["MAX_TIME"] = rydberg_global["timeMax"]
    capabilities["GLOBAL_TIME_PRECISION"] = rydberg_global["timeResolution"]
    capabilities["GLOBAL_MIN_TIME_SEPARATION"] = rydberg_global["timeDeltaMin"]

    (
        capabilities["GLOBAL_AMPLITUDE_VALUE_MIN"],
        capabilities["GLOBAL_AMPLITUDE_VALUE_MAX"],
    ) = rydberg_global["rabiFrequencyRange"]
    capabilities["GLOBAL_AMPLITUDE_VALUE_PRECISION"] = rydberg_global["rabiFrequencyResolution"]
    capabilities["GLOBAL_AMPLITUDE_SLOPE_MAX"] = rydberg_global["rabiFrequencySlewRateMax"]

    (
        capabilities["GLOBAL_PHASE_VALUE_MIN"],
        capabilities["GLOBAL_PHASE_VALUE_MAX"],
    ) = rydberg_global["phaseRange"]
    capabilities["GLOBAL_PHASE_VALUE_PRECISION"] = rydberg_global["phaseResolution"]

    (
        capabilities["GLOBAL_DETUNING_VALUE_MIN"],
        capabilities["GLOBAL_DETUNING_VALUE_MAX"],
    ) = rydberg_global["detuningRange"]
    capabilities["GLOBAL_DETUNING_VALUE_PRECISION"] = rydberg_global["detuningResolution"]
    capabilities["GLOBAL_DETUNING_SLOPE_MAX"] = rydberg_global["detuningSlewRateMax"]

    rydberg_local = rydberg.get("rydbergLocal")
    if rydberg_local:
        capabilities["LOCAL_RYDBERG_CAPABILITIES"] = True
        capabilities["LOCAL_MIN_DISTANCE_BETWEEN_SHIFTED_SITES"] = rydberg_local["spacingRadialMin"]
        capabilities["LOCAL_TIME_PRECISION"] = rydberg_local["timeResolution"]
        capabilities["LOCAL_MIN_TIME_SEPARATION"] = rydberg_local["timeDeltaMin"]
        (
            capabilities["LOCAL_MAGNITUDE_SEQUENCE_VALUE_MIN"],
            capabilities["LOCAL_MAGNITUDE_SEQUENCE_VALUE_MAX"],
        ) = rydberg_local["detuningRange"]
        capabilities["LOCAL_MAGNITUDE_SLOPE_MAX"] = rydberg_local["detuningSlewRateMax"]
        capabilities["LOCAL_MAX_NONZERO_PATTERN_VALUES"] = rydberg_local[
            "numberLocalDetuningSitesMax"
        ]
        (
            capabilities["MAGNITUDE_PATTERN_VALUE_MIN"],
            capabilities["MAGNITUDE_PATTERN_VALUE_MAX"],
        ) = rydberg_local["siteCoefficientRange"]

    return DeviceCapabilitiesConstants.parse_obj(capabilities)



def ahs_noise_model(properties: DeviceCapabilities) -> AhsNoise: 
    return _ahs_noise_model(properties)

@singledispatch
def _ahs_noise_model(properties: DeviceCapabilities) -> AhsNoise:
    raise NotImplementedError("An AHS noise model cannot be created from device capabilities of "
                              f"type {type(properties)}.")
    
@_ahs_noise_model.register(QueraDeviceCapabilities)
def _(properties: QueraDeviceCapabilities): 

    capabilities = properties.paradigm
    performance = capabilities.performance
    noise_data = AhsNoiseData(
        site_position_error= float(performance.lattice.sitePositionError),
        filling_error = float(performance.lattice.vacancyErrorTypical),
        vacancy_error = float(performance.lattice.vacancyErrorTypical), 
        ground_prep_error = float(performance.rydberg.rydbergGlobal.groundPrepError),
        rabi_amplitude_ramp_correction = performance.rydberg.rydbergGlobal.rabiAmplitudeRampCorrection,
        rabi_frequency_error_rel = float(performance.rydberg.rydbergGlobal.rabiFrequencyGlobalErrorRel),
        detuning_error = float(performance.rydberg.rydbergGlobal.detuningError),
        detuning_inhomogeneity = float(performance.rydberg.rydbergGlobal.detuningInhomogeneity), 
        atom_detection_error_false_positive = float(performance.lattice.atomDetectionErrorFalsePositiveTypical), 
        atom_detection_error_false_negative = float(performance.lattice.atomDetectionErrorFalseNegativeTypical), 
        rabi_amplitude_max = float(capabilities.rydberg.rydbergGlobal.rabiFrequencyRange[-1])
    )
    return AhsNoise(noise_data)



def lexi_mapping_routing_pass(
    properties: DeviceCapabilities, connectivity_graph: DiGraph
) -> LexiRoutingPass:
    return _lexi_mapping_routing_pass(properties, connectivity_graph)

@singledispatch
def _lexi_mapping_routing_pass(
    properties: DeviceCapabilities, connectivity_graph: DiGraph
) -> LexiRoutingPass:
    raise NotImplementedError


@_lexi_mapping_routing_pass.register(RigettiDeviceCapabilities)
@_lexi_mapping_routing_pass.register(IonqDeviceCapabilities)
def _(
    properties: Union[RigettiDeviceCapabilities, IonqDeviceCapabilities],
    connectivity_graph: DiGraph,
) -> LexiRoutingPass:
    return LexiRoutingPass(connectivity_graph)


@_lexi_mapping_routing_pass.register(IqmDeviceCapabilities)
def _(properties: IqmDeviceCapabilities, connectivity_graph: DiGraph) -> LexiRoutingPass:
    """
    IQM provides only forward edges for their *undirected* gate connectivity graph, so back-edges must
    be introduced when creating the GateConnectivityCriterion object for an IQM QPU.
    """

    connectivity_graph = connectivity_graph.copy()
    for edge in connectivity_graph.edges:
        connectivity_graph.add_edge(edge[1], edge[0])
    return LexiRoutingPass(connectivity_graph)


# Nativization set up
@singledispatch
def create_nativization_pass(properties: DeviceCapabilities, connectivity_graph: DiGraph) -> TketCompilerPass:
    return NoOpPass()

@create_nativization_pass.register(IqmDeviceCapabilities)
def _(properties: IqmDeviceCapabilities, connectivity_graph: DiGraph) -> TketCompilerPass:
    connectivity_graph = connectivity_graph.copy()
    for edge in connectivity_graph.edges:
        connectivity_graph.add_edge(edge[1], edge[0])
    
    iqm_architecture = Architecture(list(connectivity_graph.edges))
    
    tket_passes = [
        Transform.DecomposeBRIDGE(),
        Transform.DecomposeSWAPtoCX(iqm_architecture),
        Transform.OptimisePostRouting(),
        RemoveRedundancies(),
        SimplifyInitial(allow_classical=True, create_all_qubits=True),
        auto_rebase_pass(gateset={OpType.CZ, OpType.PhasedX, OpType.Rz}), 
        SquashRzPhasedX(),
        get_virtualize_rz_pass()
    ]
    return TketCompilerPass(tket_passes)



# @create_nativization_pass.register(IonqDeviceCapabilities)
# def _(properties: IonqDeviceCapabilities, connectivity_graph: DiGraph) -> TketCompilerPass:
    
#     ionq_architecture = Architecture(list(connectivity_graph.edges))
#     tket_passes = [
#         Transform.DecomposeBRIDGE(), 
#         Transform.DecomposeSWAPtoCX(ionq_architecture), 
#         Transform.OptimisePostRouting(),
#         RebaseCustom(
#             {OpType.Rz, OpType.Rx}, tket_decompositions.cx_to_ms(), tket_decompositions.tk1_to_zxz
#         ), 
#         SimplifyInitial(allow_classical=True, create_all_qubits=True),
#         SquashTK1(), 
#         RebaseCustom(
#             {OpType.Rz, OpType.CustomGate},
#             cx_replacement=tket_decompositions.cx_to_ms(),
#             tk1_replacement=tket_decompositions.tk1_to_gpi,
#         )
#     ]
#     return TketCompilerPass(tket_passes)