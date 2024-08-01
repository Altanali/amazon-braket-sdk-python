from braket.emulation.emulation_passes import EmulationPass
from braket.ahs import AnalogHamiltonianSimulation
from braket.timings.time_series import TimeSeries
from braket.ahs.pattern import Pattern
from braket.ahs.field import Field
from braket.ahs.atom_arrangement import AtomArrangement, SiteType
from braket.ahs.driving_field import DrivingField
from braket.ahs.local_detuning import LocalDetuning
from braket.ir.ahs import Program as AHSProgram
from typing import Union, List, TypeVar, Tuple
from dataclasses import dataclass
from functools import singledispatchmethod
import scipy
import numpy as np
from decimal import Decimal

@dataclass
class AhsNoiseData:
    site_position_error: float
    filling_error: float
    vacancy_error: float
    ground_prep_error: float
    rabi_amplitude_ramp_correction: List[float]
    rabi_frequency_error_rel: float
    rabi_amplitude_max: float
    detuning_error: float
    detuning_inhomogeneity: float
    atom_detection_error_false_positive: float
    atom_detection_error_false_negative: float
    
    
AhsProgramType = TypeVar('AhsProgramType', bound = AHSProgram | AnalogHamiltonianSimulation)


class AhsNoise(EmulationPass[AhsProgramType]):
    def __init__(self, ahs_noise_data: AhsNoiseData):
        self._noise_data = ahs_noise_data
    
            
    def run(self, program: AhsProgramType, steps: int = 100) -> AhsProgramType:
        return self._apply_noise_model(program, steps)
    
    @singledispatchmethod
    def _apply_noise_model(self, program: AhsProgramType, steps: int) -> AhsProgramType:
        raise NotImplementedError
    
    @_apply_noise_model.register(AnalogHamiltonianSimulation)
    def _(self, program: AnalogHamiltonianSimulation, steps: int) -> AnalogHamiltonianSimulation:
        sites, fillings, preseq = self._apply_lattice_initialization_errors(program)
        drive, local_detuning = self._apply_rydberg_noise(program, steps)
        
        register = AtomArrangement()
        for (site, filling) in zip(sites, fillings):
            if filling == 1:
                register.add(site)
            else:
                register.add(site, site_type=SiteType.VACANT)
                
                
        return AnalogHamiltonianSimulation(
            register=register, 
            hamiltonian=drive+local_detuning
        )
    
    @_apply_noise_model.register(AHSProgram)
    def _(self, program: AHSProgram) -> AHSProgram:
        raise NotImplementedError
        
    def _apply_lattice_initialization_errors(self, program: AnalogHamiltonianSimulation) \
        -> Tuple[List[List[float]], List[int], List[int]]:
        #Default to using the typical position error for now instead of the worst case error.
        sites = [[float(x), float(y)] for (x, y) in zip(program.register.coordinate_list(0), program.register.coordinate_list(1))]
        filling = program.to_ir().setup.ahs_register.filling
        
        erroneous_sites = self._apply_site_position_error(sites)
        erroneous_filling = self._apply_binomial_noise(filling, \
                    self._noise_data.filling_error, self._noise_data.vacancy_error)
        
        pre_seq = self._apply_binomial_noise(erroneous_filling, \
                self._noise_data.atom_detection_error_false_negative, \
                self._noise_data.atom_detection_error_false_positive)

        erroneous_filling = self._apply_binomial_noise(erroneous_filling, \
                0, self._noise_data.ground_prep_error)
        
        return erroneous_sites, erroneous_filling, pre_seq
    
    
    def _apply_site_position_error(self, 
                                   sites: List[List[float]]) -> List[List[float]]: 
        erroneous_sites = []
        for site in sites:
            erroneous_sites.append(
                site + self._noise_data.site_position_error * np.random.normal(size=2)
            )
            
        return erroneous_sites
    
    def _apply_binomial_noise(self, 
                              arr: List[int],
                              binomial_probability_p01: float,
                              binomial_probability_p10: float) -> List[int]: 
        noisy_arr = []
        for val in arr:
            if val == 1:
                # Apply the error of switching 1 as 0
                noisy_arr.append(1 - np.random.binomial(1, binomial_probability_p10))
            else:
                # Apply the error of switching 0 as 1
                noisy_arr.append(np.random.binomial(1, binomial_probability_p01))
            
        return noisy_arr
    
    
    def _apply_rydberg_noise(self, program: AnalogHamiltonianSimulation, steps: int) \
        -> Tuple[TimeSeries, LocalDetuning]:
            noisy_detuning, local_detuning = self._apply_detuning_errors(
                program.hamiltonian.detuning,
                program.to_ir().setup.ahs_register.filling,
                steps
            )
            
            noisy_amplitude = self._apply_amplitude_errors(
                program.hamiltonian.amplitude,
                steps, 
            )
            noisy_drive = DrivingField(amplitude = noisy_amplitude,
                                       detuning = noisy_detuning,
                                       phase = program.hamiltonian.phase)
            
            return noisy_drive, local_detuning
        
    
    def _apply_detuning_errors(self, 
                               detuning: TimeSeries,
                               fillings: List[int], 
                               steps: int
        ) -> Tuple[TimeSeries, LocalDetuning]: 
        
        detuning_times = np.array(detuning.time_series.times(), dtype='float64')
        detuning_values = np.array(detuning.time_series.values(), dtype='float64')
        
        noisy_detuning_times = np.linspace(0, detuning_times[-1], steps)
        noisy_detuning_values = np.interp(noisy_detuning_times, detuning_times, detuning_values)

        # Apply the detuning error
        noisy_detuning_values += \
            self._noise_data.detuning_error * np.random.normal(size=len(noisy_detuning_values))
        
        noisy_detuning = TimeSeries.from_lists(noisy_detuning_times, noisy_detuning_values)
        
        # Apply detuning inhomogeneity        
        h = Pattern([np.random.rand() for _ in fillings])
        detuning_local = TimeSeries.from_lists(noisy_detuning_times, 
                                            self._noise_data.detuning_inhomogeneity * np.ones(
                                                len(noisy_detuning_times)
                                            )
                                            )

        # Assemble the local detuning
        local_detuning = LocalDetuning(
            magnitude=Field(
                time_series=detuning_local,
                pattern=h
            )
        )
        
        return noisy_detuning, local_detuning  
            
        
        
    def _apply_amplitude_errors(self, amplitude: TimeSeries, steps: int) \
        -> TimeSeries:
        amplitude_times = np.array(amplitude.time_series.times(), dtype='float64')
        amplitude_values = np.array(amplitude.time_series.values(), dtype='float64')
        
        # Rewrite the rabi_ramp_correction as a function of slopes
        rabi_ramp_correction_slopes = [self._noise_data.rabi_amplitude_max / float(corr.rampTime)
            for corr in self._noise_data.rabi_amplitude_ramp_correction
        ]
        rabi_ramp_correction_fracs = [float(corr.rabiCorrection)
            for corr in self._noise_data.rabi_amplitude_ramp_correction
        ]    
        rabi_ramp_correction_slopes = rabi_ramp_correction_slopes[::-1]
        rabi_ramp_correction_fracs = rabi_ramp_correction_fracs[::-1]
        
        # Helper function to find the correction factor for a given slope
        get_frac = scipy.interpolate.interp1d(rabi_ramp_correction_slopes, 
                                    rabi_ramp_correction_fracs, 
                                    bounds_error=False, 
                                    fill_value="extrapolate"
                                    )
            
        noisy_amplitude_times = np.linspace(0, amplitude_times[-1], steps)
        noisy_amplitude_values = []
        
        # First apply the rabi ramp correction
        for ind in range(len(amplitude_times)):
            if ind == 0:
                continue
                
            # First determine the correction factor from the slope
            t1, t2 = amplitude_times[ind-1], amplitude_times[ind]
            v1, v2 = amplitude_values[ind-1], amplitude_values[ind]
            slope = (v2 - v1) / (t2 - t1)
            if np.abs(slope) > 0:
                frac = get_frac(np.abs(slope)) * np.sign(slope)        
            else:
                frac = 1.0
            
            # Next, determine the coefficients for the quadratic correction
            if frac >= 1.0:
                a, b, c = 0, 0, v2
            else:
                # Determine the coefficients for the quadratic correction
                # of the form f(t) = a*t^2 + b * t + c 
                # such that f(t1) = v1 and f(t2) = v2 and 
                # a/3*(t2^3-t1^3) + b/2*(t2^2-t1^2) + c(t2-t1) = frac * (t2-t1) * (v2-v1)/2
                
                a = 3 * (v1 + frac * v1 + v2 - frac * v2)/(t1 - t2)**2
                c = (t2 * v1 * ((2 + 3 * frac) * t1 + t2) + t1 * v2 * (t1 + (2 - 3 * frac) * t2))/(t1 - t2)**2
                b = (v2 - c - a * t2**2) / t2    
            
            # Finally, put values into noisy_amplitude_values
            for t in noisy_amplitude_times:
                if t1 <= t and t <= t2:
                    noisy_amplitude_values.append(a * t**2 + b * t + c)

                    
        # Next apply amplitude error
        rabi_errors = 1 + self._noise_data.rabi_frequency_error_rel * np.random.normal(size=len(noisy_amplitude_values))
        noisy_amplitude_values = np.multiply(noisy_amplitude_values, rabi_errors)
        noisy_amplitude_values = [max(0, value) for value in noisy_amplitude_values] # amplitude has to be non-negative
                    
        noisy_amplitude = TimeSeries.from_lists(noisy_amplitude_times, noisy_amplitude_values)
        
        return noisy_amplitude
            