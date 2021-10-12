import numpy as np
from qiskit import *
import random


class QuantumCircuitError(Exception):
    pass


def MixedStateGenerationCircuit(target_unitary: QuantumCircuit, ref=None):
    # Get state scale(the number of qubit)
    scale = len(target_unitary.qubits)
    # Input validity check / reference list prepare
    if ref is None:
        ref = 2 ** scale
    if type(ref) == int:
        if ref > 2 ** scale:
            raise QuantumCircuitError('Error in MixedStateGenerationCircuit! Rank should be 2**scale at most!')
        ref = [1 / np.sqrt(ref) for i in range(ref)]
    elif type(ref) == list:
        if len(ref) > 2 ** scale:
            raise QuantumCircuitError('Error in MixedStateGenerationCircuit! Rank should be 2**scale at most!')
        ref = np.array(ref)
        ref = ref / np.sqrt(sum(abs(ref) ** 2))
    else:
        raise QuantumCircuitError('Error in MixedStateGenerationCircuit! Invalid ref input!')
    initvec = np.zeros(2 ** scale)
    for i in range(len(ref)):
        initvec[i] = ref[i]
    msg_circ = QuantumCircuit(scale * 2)
    msg_circ.initialize(initvec, [i for i in range(scale)])
    msg_circ.barrier([i for i in range(scale)])
    for i in range(scale):
        msg_circ.cx(i, scale + i)
    msg_circ.barrier([i for i in range(scale)])
    msg_circ.compose(target_unitary, [i for i in range(scale)], inplace=True)
    return msg_circ


def Paulistring2Index(paulistring: str):
    paulistring = paulistring[::-1]
    index = 0
    for i in range(len(paulistring)):
        if paulistring[i] == 'X':
            index += 1 * 4 ** i
        elif paulistring[i] == 'Y':
            index += 2 * 4 ** i
        elif paulistring[i] == 'Z':
            index += 3 * 4 ** i
        elif paulistring[i] != 'I':
            raise QuantumCircuitError("Error in Paulistring2Index! Incorrect Pauli string input.")
    return int(index)


def SingleMeasurementBasisGenerate(ensemble: list, weight: list = None):
    if weight is None:
        observable_id = random.choice([i for i in range(len(ensemble))])
    else:
        if len(ensemble) != len(weight):
            raise QuantumCircuitError(
                'Error in SingleMeasurementBasisGenerate. Different length of ensembles '
                'and weights ')
        observable_id = random.choices([i for i in range(len(ensemble))], weight)[0]
    m_basis = ensemble[observable_id]
    for i in range(len(m_basis)):
        if m_basis[i] == 'N':
            m_basis[i] = random.choice(['X', 'Y', 'Z'])
    return m_basis, observable_id
