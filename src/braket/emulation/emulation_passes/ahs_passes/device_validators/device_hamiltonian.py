from typing import Dict

from pydantic.v1.class_validators import root_validator

from braket.analog_hamiltonian_simulator.rydberg.validators.hamiltonian import HamiltonianValidator


class DeviceHamiltonianValidator(HamiltonianValidator):
    LOCAL_RYDBERG_CAPABILITIES: bool = False

    @root_validator(pre=True, skip_on_failure=True)
    def max_zero_local_detuning(cls, values: Dict) -> Dict:
        """
        Checks if local detuning is supported in this Hamiltonian definition based on the device
        capabilities.

        Args:
            values (Dict): Contains DeviceCapabilitiesConstants and the Hamiltonian data
                to validate.
        Returns:
            Dict: Unmodified Hamiltonian data and DeviceCapabilitiesConstants

        Raises:
            ValueError: If local detuning is non empty in the Hamiltonian definition, but local
            rydberg capabilities are not provded or supported in the device capabilities.
        """
        LOCAL_RYDBERG_CAPABILITIES = values["LOCAL_RYDBERG_CAPABILITIES"]
        local_detuning = values.get("localDetuning", [])
        if not LOCAL_RYDBERG_CAPABILITIES and len(local_detuning):
            raise ValueError(
                f"Local detuning cannot be specified; \
{len(local_detuning)} are given. Specifying local \
detuning is an experimental capability, use Braket Direct to request access."
            )
        return values
