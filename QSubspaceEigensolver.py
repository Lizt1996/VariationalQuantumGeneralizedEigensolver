from qiskit import *
from QMeasure import HadamardTest, HadamardTest_Analytical, state_backend
from QAnsatz import Ansatze
from QCircuit import MixedStateGenerationCircuit
from QHamiltonian import Hamiltonian_in_Pauli_String
import random
import numpy as np


class SubspaceEigSolverError(Exception):
    pass


class SubspaceEigSolver:
    """
    Hamiltonian is in the format of the sum of weighted Pauli string
    L(x) = Tr(ha)
    a = sum_{i=0}^{m-1} U(x)|i><i|U^{-1}(x)
    h is the observable state which may be a mixed state
    """

    def __init__(self, Hamiltonian: Hamiltonian_in_Pauli_String, ansatze: Ansatze, weight_list: list):
        """
        Class of state subspace eigen solver.
        :param Hamiltonian: H in Hx = 入x
        :param ansatze: The quantum circuit network with respect to parameters.
        :param weight_list: The guidance of subspace searching. c.f. Von Neumann theorem.
        """
        self.state_scale = Hamiltonian.qubits
        if len(weight_list) > 2 ** self.state_scale:
            raise SubspaceEigSolverError('Error in StateSubspaceEigSolver! Incorrect weight list size')
        self.ansatze = ansatze
        self.weight_list = weight_list
        self.Hamiltonian = Hamiltonian

    def AnsatzStateGenerationCircuit(self, partial_flag: bool = False, pid: int = 0, pn=None):
        return MixedStateGenerationCircuit(self.ansatze.circuit(partial_flag, pid, pn),
                                           list(np.sqrt(abs(np.array(self.weight_list)))))

    def LossFunctionAnalytical(self):
        return self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                       test_circuit=self.AnsatzStateGenerationCircuit(),
                                                       active_qubits=[i for i in range(self.state_scale)])

    def PartialDerivativeAnalytical(self, pid):
        """
        Partial derivative of parameter pid. (c.f. K. Mitarai, Quantum circuit learning)
        :param pid: parameter identifier.
        :return: Partial derivative = 1/2*ppd+1/2*npd.
        """
        ppd = self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                      test_circuit=self.AnsatzStateGenerationCircuit(partial_flag=True,
                                                                                                     pid=pid,
                                                                                                     pn='+'),
                                                      active_qubits=[i for i in range(self.state_scale)])
        npd = self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                      test_circuit=self.AnsatzStateGenerationCircuit(partial_flag=True,
                                                                                                     pid=pid,
                                                                                                     pn='-'),
                                                      active_qubits=[i for i in range(self.state_scale)])
        return np.real(1 / 2 * ppd - 1 / 2 * npd)

    def GetJacobianAnalytical(self, par: list):
        if len(par) != self.ansatze.getParameterLength():
            raise SubspaceEigSolverError(
                'Error in SubspaceEigSolver GetJacobian! Incorrect parameter length')
        self.setParameter(par)
        jac = [0 for i in range(len(par))]
        for i in range(len(par)):
            jac[i] = self.PartialDerivativeAnalytical(i)
        return np.array(jac)

    def LossFunction(self, shots: int = 10000):
        return self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                       test_circuit=self.AnsatzStateGenerationCircuit(),
                                                       active_qubits=[i for i in range(self.state_scale)],
                                                       shots=shots)

    def PartialDerivative(self, pid, shots: int = 10000):
        """
        Partial derivative of parameter pid. (c.f. K. Mitarai, Quantum circuit learning)
        :param pid: Parameter identifier.
        :param shots: How many times the DSWAPT measure the density matrix product trace.
        :return: Partial derivative = 1/2*ppd+1/2*npd.
        """
        ppd = self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                      test_circuit=self.AnsatzStateGenerationCircuit(partial_flag=True,
                                                                                                     pid=pid,
                                                                                                     pn='+'),
                                                      active_qubits=[i for i in range(self.state_scale)],
                                                      shots=shots)
        npd = self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                      test_circuit=self.AnsatzStateGenerationCircuit(partial_flag=True,
                                                                                                     pid=pid,
                                                                                                     pn='-'),
                                                      active_qubits=[i for i in range(self.state_scale)],
                                                      shots=shots)
        return np.real(1 / 2 * ppd - 1 / 2 * npd)

    def GetJacobian(self, par: list, shots: int = 10000):
        if len(par) != self.ansatze.getParameterLength():
            raise SubspaceEigSolverError(
                'Error in SubspaceEigSolver GetJacobian! Incorrect parameter length')
        self.setParameter(par)
        jac = [0 for i in range(len(par))]
        for i in range(len(par)):
            jac[i] = self.PartialDerivative(i, shots)
        return np.array(jac)

    def EigTrace(self, getEigenstate: bool = False, getLossFunction: bool = False):
        eigval = []
        eigvec = []
        lossfun = 0
        initvec = np.zeros(2 ** self.state_scale)
        for i in range(len(self.weight_list)):
            initvec[i] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            initvec[i] = 0
            check_circuit.compose(self.ansatze.circuit(), [i for i in range(self.state_scale)], inplace=True)
            eigval.append(self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                                  test_circuit=check_circuit,
                                                                  active_qubits=[i for i in range(self.state_scale)]))
            if getEigenstate:
                job = execute(check_circuit, state_backend)
                result = job.result()
                eigvec.append(result.get_statevector(check_circuit, decimals=3))
        if getLossFunction:
            lossfun = self.LossFunctionAnalytical()
        return {'eigval': eigval, 'eigvec': eigvec, 'lossfun': lossfun}

    def setParameter(self, new_parameter: list):
        self.ansatze.setParameter(new_parameter)

    def getLossFunction(self, parameter: np.array):
        p = [parameter[i] for i in range(self.ansatze.getParameter().__len__())]
        self.setParameter(parameter)
        return self.LossFunction()

    def getLossFunctionAnalytical(self, parameter: np.array):
        p = [parameter[i] for i in range(self.ansatze.getParameter().__len__())]
        self.setParameter(parameter)
        return self.LossFunctionAnalytical()

    def getParameter(self):
        return self.ansatze.getParameter()

    def getEigenData(self, par, vector_required: bool = True, lossfun_required: bool = True):
        self.setParameter(par)
        return self.EigTrace(vector_required, lossfun_required)

    def showStateVector(self, parameter: list):
        self.setParameter(parameter)
        backend = BasicAer.get_backend('statevector_simulator')
        qc = self.ansatze.circuit()
        print(qc.draw('text'))
        job = execute(qc, backend)
        result = job.result()
        return result.get_statevector(qc, decimals=3)


class SubspaceEigSolver_ClassicalEfficientSimulator:
    """
    Hamiltonian is in the format of the sum of weighted Pauli string
    L(x) = Tr(ha)
    a = sum_{i=0}^{m-1} U(x)|i><i|U^{-1}(x)
    h is the observable state which may be a mixed state
    """

    def __init__(self, Hamiltonian: Hamiltonian_in_Pauli_String, ansatze: Ansatze, weight_list: list):
        """
        Class of state subspace eigen solver.
        :param Hamiltonian: H in Hx = 入x
        :param ansatze: The quantum circuit network with respect to parameters.
        :param weight_list: The guidance of subspace searching. c.f. Von Neumann theorem.
        """
        self.state_scale = Hamiltonian.qubits
        if len(weight_list) > 2 ** self.state_scale:
            raise SubspaceEigSolverError('Error in StateSubspaceEigSolver! Incorrect weight list size')
        self.ansatze = ansatze
        self.weight_list = weight_list
        self.Hamiltonian = Hamiltonian

    def LossFunctionAnalytical(self):
        res = 0
        for j in range(len(self.weight_list)):
            initvec = np.zeros(2 ** self.state_scale)
            initvec[j] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            check_circuit.compose(self.ansatze.circuit(), [i for i in range(self.state_scale)], inplace=True)
            job = execute(check_circuit, state_backend)
            result = job.result()
            state = result.get_statevector(check_circuit, decimals=3)
            res += self.weight_list[j] * np.dot(np.dot(state.conj(), self.Hamiltonian.hamiltonian_mat), state)
        return np.real(res)

    def PartialDerivativeAnalytical(self, pid):
        """
        Partial derivative of parameter pid. (c.f. K. Mitarai, Quantum circuit learning)
        :param pid: parameter identifier.
        :return: Partial derivative = 1/2*ppd-1/2*npd.
        """
        ppd = 0
        initvec = np.zeros(2 ** self.state_scale)
        for j in range(len(self.weight_list)):
            initvec[j] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            check_circuit.compose(self.ansatze.circuit(partial_flag=True,
                                                       pid=pid,
                                                       pn='+'),
                                  [i for i in range(self.state_scale)], inplace=True)
            job = execute(check_circuit, state_backend)
            result = job.result()
            state = result.get_statevector(check_circuit)
            ppd += self.weight_list[j] * np.dot(np.dot(state.conj(), self.Hamiltonian.hamiltonian_mat), state)
            initvec[j] = 0

        npd = 0
        for j in range(len(self.weight_list)):
            initvec[j] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            check_circuit.compose(self.ansatze.circuit(partial_flag=True,
                                                       pid=pid,
                                                       pn='-'),
                                  [i for i in range(self.state_scale)], inplace=True)
            job = execute(check_circuit, state_backend)
            result = job.result()
            state = result.get_statevector(check_circuit)
            npd += self.weight_list[j] * np.dot(np.dot(state.conj(), self.Hamiltonian.hamiltonian_mat), state)
            initvec[j] = 0

        return np.real(1 / 2 * ppd - 1 / 2 * npd)

    def GetJacobianAnalytical(self, par: list):
        if len(par) != self.ansatze.getParameterLength():
            raise SubspaceEigSolverError(
                'Error in SubspaceEigSolver GetJacobian! Incorrect parameter length')
        self.setParameter(par)
        jac = [0 for i in range(len(par))]
        for i in range(len(par)):
            jac[i] = self.PartialDerivativeAnalytical(i)
        return np.array(jac)

    def LossFunction(self, shots: int = 10000):
        res = 0
        for i in range(len(self.weight_list)):
            initvec = np.zeros(2 ** self.state_scale)
            initvec[i] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            check_circuit.compose(self.ansatze.circuit(), [i for i in range(self.state_scale)], inplace=True)
            res += self.weight_list[i] * self.Hamiltonian.ExpectationMeasurement(
                MeasurementMethod=HadamardTest,
                test_circuit=check_circuit,
                active_qubits=[i for i in range(self.state_scale)],
                shots=shots)
        return res

    def PartialDerivative(self, pid, shots: int = 10000):
        """
        Partial derivative of parameter pid. (c.f. K. Mitarai, Quantum circuit learning)
        :param pid: Parameter identifier.
        :param shots: How many times the DSWAPT measure the density matrix product trace.
        :return: Partial derivative = 1/2*ppd+1/2*npd.
        """
        ppd = 0
        for i in range(len(self.weight_list)):
            initvec = np.zeros(2 ** self.state_scale)
            initvec[i] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            check_circuit.compose(self.ansatze.circuit(partial_flag=True,
                                                       pid=pid,
                                                       pn='+'),
                                  [i for i in range(self.state_scale)], inplace=True)
            ppd += self.weight_list[i] * self.Hamiltonian.ExpectationMeasurement(
                MeasurementMethod=HadamardTest,
                test_circuit=check_circuit,
                active_qubits=[i for i in range(self.state_scale)],
                shots=shots)

        npd = 0
        for i in range(len(self.weight_list)):
            initvec = np.zeros(2 ** self.state_scale)
            initvec[i] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            check_circuit.compose(self.ansatze.circuit(partial_flag=True,
                                                       pid=pid,
                                                       pn='-'),
                                  [i for i in range(self.state_scale)], inplace=True)
            npd += self.weight_list[i] * self.Hamiltonian.ExpectationMeasurement(
                MeasurementMethod=HadamardTest,
                test_circuit=check_circuit,
                active_qubits=[i for i in range(self.state_scale)],
                shots=shots)

        return np.real(1 / 2 * ppd - 1 / 2 * npd)

    def GetJacobian(self, par: list, shots: int = 10000):
        if len(par) != self.ansatze.getParameterLength():
            raise SubspaceEigSolverError(
                'Error in SubspaceEigSolver GetJacobian! Incorrect parameter length')
        self.setParameter(par)
        jac = [0 for i in range(len(par))]
        for i in range(len(par)):
            jac[i] = self.PartialDerivative(i, shots)
        return np.array(jac)

    def EigTrace(self, getEigenstate: bool = False, getLossFunction: bool = False):
        eigval = []
        eigvec = []
        lossfun = 0
        initvec = np.zeros(2 ** self.state_scale)
        for i in range(len(self.weight_list)):
            initvec[i] = 1
            check_circuit = QuantumCircuit(self.state_scale)
            check_circuit.initialize(initvec, [i for i in range(self.state_scale)])
            initvec[i] = 0
            check_circuit.compose(self.ansatze.circuit(), [i for i in range(self.state_scale)], inplace=True)
            eigval.append(self.Hamiltonian.ExpectationMeasurement(MeasurementMethod=HadamardTest_Analytical,
                                                                  test_circuit=check_circuit,
                                                                  active_qubits=[i for i in range(self.state_scale)]))
            if getEigenstate:
                job = execute(check_circuit, state_backend)
                result = job.result()
                eigvec.append(result.get_statevector(check_circuit, decimals=3))
        if getLossFunction:
            lossfun = self.LossFunctionAnalytical()
        return {'eigval': eigval, 'eigvec': eigvec, 'lossfun': lossfun}

    def setParameter(self, new_parameter: list):
        self.ansatze.setParameter(new_parameter)

    def getLossFunction(self, parameter: np.array):
        p = [parameter[i] for i in range(self.ansatze.getParameter().__len__())]
        self.setParameter(parameter)
        return self.LossFunction()

    def getLossFunctionAnalytical(self, parameter: np.array):
        p = [parameter[i] for i in range(self.ansatze.getParameter().__len__())]
        self.setParameter(parameter)
        return self.LossFunctionAnalytical()

    def getParameter(self):
        return self.ansatze.getParameter()

    def getEigenData(self, par, vector_required: bool = True, lossfun_required: bool = True):
        self.setParameter(par)
        return self.EigTrace(vector_required, lossfun_required)

    def showStateVector(self, parameter: list):
        self.setParameter(parameter)
        backend = BasicAer.get_backend('statevector_simulator')
        qc = self.ansatze.circuit()
        print(qc.draw('text'))
        job = execute(qc, backend)
        result = job.result()
        return result.get_statevector(qc, decimals=3)
