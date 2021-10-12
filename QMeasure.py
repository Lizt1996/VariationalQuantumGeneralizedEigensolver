import numpy as np
from qiskit import *


class QuantumMeasurementError(Exception):
    pass


qasm_backend = BasicAer.get_backend('qasm_simulator')
state_backend = BasicAer.get_backend('statevector_simulator')


def HadamardTest(P: QuantumCircuit, U: QuantumCircuit, U_active_qubits: list, shots: int = 10000):
    """
    :param shots: the number of measurement
    :param U: Testing state generation circuit
    :param U_active_qubits:
    :param P: Pauli matrix
    :return: Re(<x|P|x>)
    """
    p_size = len(P.qubits)
    u_size = len(U.qubits)
    statescale = len(U_active_qubits)
    # Input validity check
    if statescale != p_size:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Length of active qubit lists are not equal.")
    if not statescale:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Empty list input.")
    if statescale > p_size or statescale > u_size:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Length of active qubit lists should less than qubits.")
    if max(U_active_qubits) >= u_size or min(U_active_qubits) < 0:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Value in active qubit lists out of range.")
    # Quantum circuit compose
    test_circuit = QuantumCircuit(u_size + 1, 1)
    test_circuit.compose(U, [i for i in range(u_size)], inplace=True)
    test_circuit.barrier([i for i in range(u_size)])
    test_circuit.h(u_size)
    test_circuit.compose(P.control(1), [u_size] + [i for i in range(p_size)], inplace=True)
    test_circuit.h(u_size)
    test_circuit.measure(u_size, 0)
    job_sim = execute(test_circuit, qasm_backend, shots=shots)
    result_sim = job_sim.result()
    res = result_sim.get_counts(test_circuit)
    if res.get('0') is None:
        return -1
    return 2 * (res['0'] / shots) - 1


def DestructiveSwapTest(P: QuantumCircuit, P_active_qubits: list, U: QuantumCircuit, U_active_qubits: list,
                        shots: int = 10000):
    p_size = len(P.qubits)
    u_size = len(U.qubits)
    statescale = len(P_active_qubits)
    # Input validity check
    if statescale != len(U_active_qubits):
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Length of active qubit lists are not equal.")
    if not statescale:
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Empty list input.")
    if statescale > p_size or statescale > u_size:
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Length of active qubit lists should less than qubits.")
    if max(P_active_qubits) >= p_size or min(P_active_qubits) < 0 or max(U_active_qubits) >= u_size or min(
            U_active_qubits) < 0:
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Value in active qubit lists out of range.")
    # Quantum circuit compose
    test_circuit = QuantumCircuit(p_size + u_size, statescale * 2)
    test_circuit.compose(P, qubits=[i for i in range(p_size)], inplace=True)
    test_circuit.compose(U, [i + p_size for i in range(u_size)], inplace=True)
    for i in range(statescale):
        test_circuit.cnot(P_active_qubits[i], p_size + U_active_qubits[i])
        test_circuit.h(P_active_qubits[i])
        test_circuit.measure(test_circuit.qubits[P_active_qubits[i]], test_circuit.clbits[i])
        test_circuit.measure(test_circuit.qubits[p_size + U_active_qubits[i]], test_circuit.clbits[statescale + i])
    backend_sim = BasicAer.get_backend('qasm_simulator')
    res = 0
    job_sim = execute(test_circuit, backend_sim, shots=shots)
    result_sim = job_sim.result()
    result_dict = result_sim.get_counts(test_circuit)
    while result_dict != {}:
        piece = result_dict.popitem()
        cr = 0
        for i in range(statescale):
            if piece[0][i] == '1' and piece[0][statescale + i] == '1':
                cr += 1
        if cr % 2 == 0:
            res += piece[1]
    return 2 * (res / shots) - 1


def SwapTest_Analytical(P: QuantumCircuit, P_active_qubits: list, U: QuantumCircuit, U_active_qubits: list):
    p_size = len(P.qubits)
    u_size = len(U.qubits)
    statescale = len(P_active_qubits)
    # Input validity check
    if statescale != len(U_active_qubits):
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Length of active qubit lists are not equal.")
    if not statescale:
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Empty list input.")
    if statescale > p_size or statescale > u_size:
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Length of active qubit lists should less than qubits.")
    if max(P_active_qubits) >= p_size or min(P_active_qubits) < 0 or max(U_active_qubits) >= u_size or min(
            U_active_qubits) < 0:
        raise QuantumMeasurementError(
            "Error in DestructiveSwapTest! Value in active qubit lists out of range.")
    # Quantum circuit compose
    test_circuit = QuantumCircuit(p_size + u_size + 1)
    test_circuit.compose(P, qubits=[i for i in range(p_size)], inplace=True)
    test_circuit.compose(U, [i + p_size for i in range(u_size)], inplace=True)
    test_circuit.barrier()
    test_circuit.h(p_size + u_size)
    for i in range(statescale):
        test_circuit.cswap(p_size + u_size, P_active_qubits[i], p_size + U_active_qubits[i])
    test_circuit.h(p_size + u_size)
    job = execute(test_circuit, state_backend)
    result = job.result()
    statevec = result.get_statevector(test_circuit, decimals=3)
    res = 0
    for i in range(int(len(statevec) / 2)):
        res += abs(statevec[i]) ** 2
    return 2 * np.real(res) - 1


def HadamardTest_Analytical(P: QuantumCircuit, U: QuantumCircuit, U_active_qubits: list, shots: int = 10000):
    p_size = len(P.qubits)
    u_size = len(U.qubits)
    statescale = len(U_active_qubits)
    # Input validity check
    if statescale != p_size:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Length of active qubit lists are not equal.")
    if not statescale:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Empty list input.")
    if statescale > p_size or statescale > u_size:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Length of active qubit lists should less than qubits.")
    if max(U_active_qubits) >= u_size or min(U_active_qubits) < 0:
        raise QuantumMeasurementError(
            "Error in HadamardTest! Value in active qubit lists out of range.")
    # Quantum circuit compose
    test_circuit = QuantumCircuit(u_size + 1)
    test_circuit.compose(U, [i for i in range(u_size)], inplace=True)
    test_circuit.barrier([i for i in range(u_size)])
    test_circuit.h(u_size)
    test_circuit.compose(P.control(1), [u_size] + [i for i in range(p_size)], inplace=True)
    test_circuit.h(u_size)
    job = execute(test_circuit, state_backend)
    result = job.result()
    statevec = result.get_statevector(test_circuit, decimals=3)
    res = 0
    for i in range(int(len(statevec) / 2)):
        res += abs(statevec[i]) ** 2
    return 2 * np.real(res) - 1


def ExpectationAnalytical(P: QuantumCircuit, U: QuantumCircuit = None, shots=None):
    """
    :param U:   |x> = U|0>
    :param P:   Pauli matrix
    :return:    <x|P|x>
    """
    num = P.qubits.__len__()
    if U is not None:
        if P.qubits.__len__() != U.qubits.__len__():
            raise QuantumMeasurementError(
                "Error in Hadamard Test. The number of qubit between P and U are not "
                "equal.")
    if U is not None:
        qc = U + P + U.inverse()
    else:
        qc = P
    # qc.draw('mpl', 1, 'C:\SecretStorage\Project\Quantum\Quantum\VQA_LS\s.png')
    backend = BasicAer.get_backend('statevector_simulator')
    # print(qc.draw('text'))
    job = execute(qc, backend)
    result = job.result()
    statevec = result.get_statevector(qc, decimals=3)
    return statevec[0]


def PauliMeasurement(basis: list, rho_circuit: QuantumCircuit = None):
    statescale = len(basis)
    if rho_circuit is None:
        rho_circuit = QuantumCircuit(statescale)
    append_circuit = QuantumCircuit(statescale)
    for i in range(len(basis)):
        if basis[i] == 'X':
            append_circuit.ry(-np.pi / 2, i)
        elif basis[i] == 'Y':
            append_circuit.rx(np.pi / 2, i)
        elif basis[i] != 'Z':
            raise QuantumMeasurementError('Error in SubspaceEigSolver! Invalid basis input')
    full_circuit = rho_circuit + append_circuit
    full_circuit.barrier(range(statescale))
    full_circuit.measure_all()
    job_sim = execute(full_circuit, qasm_backend, shots=1)
    result_sim = job_sim.result()
    measure_result = result_sim.get_counts(full_circuit).popitem()[0][::-1]
    res = []
    for i in measure_result:
        res.append((-1) ** int(i))
    return res
