from braket.passes.base_pass import BasePass
from collections.abc import Iterable
from typing import TypeVar, Union
from pytket.passes import SequencePass
from pytket.transform import Transform
from pytket.circuit import Circuit as PytketCircuit
from braket.circuits import Circuit
from braket.ir.openqasm import Program as OpenQasmProgram
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.emulation.pytket_translator import (
            PytketProgramContext, 
            tket_to_qasm3, 
            PytketProgramTranslation
)

from functools import singledispatchmethod

TketPass = Union[BasePass, SequencePass, Transform]
BraketQasmType = TypeVar("BraketQasmType", bound=Union[Circuit, OpenQasmProgram])
class TketCompilerPass(BasePass[TketPass]):
    
    def __init__(self, tket_passes: Union[TketPass, Iterable[TketPass]]):
        """
        A TketCompilerPass consists of a list of Tket passes or transforms that 
        can be applied in place to a PytketCircuit object. If a TketCompilerPass is called 
        and supplied with a Braket Circuit or an OpenQASM program as the 
        task specification, the task specification is translated to a PytketCircuit, 
        compiled, and translated back to the original type. 
        """
        self._tket_passes = [tket_passes] if not isinstance(tket_passes, Iterable) \
                            else tket_passes
     
    
    @singledispatchmethod                       
    def run(self, task_specification) -> None: 
        raise ValueError(f"Circuit type {type(task_specification)} not supported for\
            compilation with Tket Passes")
        
        
    @run.register
    def _(self, task_specification: Circuit) -> Circuit: 
        """
        Translates a Braket Circuit to OpenQASM and subsequently to a PytketCircuit.
        Then applies all Tket Passes to the translation result and converts the 
        compiled PytketCircuit back to a Braket Circuit before returning. 
        
        Args:
            task_specification (Circuit): The Braket Circuit to apply the TketPasses 
            to. 
            
        Returns: 
            Circuit: A compiled Braket circuit. 
        """
        open_qasm_program = self.run(task_specification.to_ir("OPENQASM"))
        return Circuit.from_ir(open_qasm_program)

        
        
    @run.register
    def _(self, task_specification: OpenQasmProgram) -> OpenQasmProgram:
        """
        Translates the OpenQASM program to a PytketCircuit before running all Tket Passes. 
        Converts the compiled PytketCircuit back to an OpenQasmProgram before returning.
        
        Args: 
            task_specification (OpenQasmProgram): The program to apply the TketPasses
            to. 
            
        Returns: 
            OpenQasmProgram: A compiled OpenQasmProgram.
        """
        pytket_translation: PytketProgramTranslation = \
            Interpreter(PytketProgramContext()).build_circuit(task_specification.source)
        
        pytket_circuit = pytket_translation.circuit
        self.run(pytket_circuit)
        openqasm_result = tket_to_qasm3(pytket_circuit)
        return OpenQasmProgram(source=openqasm_result)
        
    @run.register
    def _(self, task_specification: PytketCircuit) -> PytketCircuit:
        """
        Applies a list of Tket sequence or transformation passes to the input
        PytketCircuit object. 
        
        Args: 
            task_specification (PytketCircuit): The circuit to apply the TketPasses
            to. 
            
        Returns: 
            PytketCircuit: A compiled PytketCircuit. 
        """
        for tket_pass in self._tket_passes:
            tket_pass.apply(task_specification)
        return task_specification
    
    