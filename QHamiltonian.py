import numpy as np
import numbers
import scipy
import scipy.linalg
from qiskit import *

from QMeasure import HadamardTest_Analytical


class HamiltonianError(Exception):
    pass


class Hamiltonian_in_Pauli_String:
    def __init__(self, qubits, unitary: list = None, coefficient: list = None, hamiltonian_mat=None):
        """
        The Hamiltonian in the format of the sum of weighted Pauli strings.
        :param qubits: The number of qubits the Hamiltonian represented.
        :param unitary: The list of Pauli strings.
                        Valid format:
                            1. Type-Position Strings:
                                    [
                                    'P(x0)P(x1)...'
                                    ]

                            2.  Type on Fixed Position Matrix:
                                    [
                                    ['P(0)', 'P(1)', ..., 'P(qubits-1)'],
                                    ...,
                                    ['P(0)', 'P(1)', ..., 'P(qubits-1)']
                                    ]


                                        where 'P(x)' is 'I', 'X', 'Y' or 'Z',
                                              x of 'P(x)' represent the qubit on which the Pauli matrix 'P(x)' act.

                                    example:
                                    [
                                    'X0',
                                    'Z1Y2'
                                    ]

                                        which is equivalent to
                                    [
                                    ['X', 'I', 'I'],
                                    ['I', 'Z', 'Y']
                                    ]

        :param coefficient: The list of coefficient of Pauli string in corresponding index position of unitary.
        :param hamiltonian_mat: The numpy.array of Hamiltonian matrix, which is only used to analyse the correctness
                                and convergence of algorithm
        :param eigen_data: The dict of eigen values and eigen vectors of Hamiltonian matrix, which is only used to
                           analyse the correctness and convergence of algorithm
        """
        # Type on fixed position matrix
        self.qubits = qubits
        self.unitary = []
        self.coefficient = []
        # Type-position strings. Each Pauli string will be sorted based on the qubit.
        self.unitary_str_ver = []
        self.hamiltonian_mat = None
        if unitary is not None or coefficient is not None:
            # Pauli string validity check -length
            if len(unitary) == len(coefficient):
                self.coefficient = coefficient
                # Parameter unitary may be in the format of type-position strings.
                if type(unitary[0]) == str:
                    for i in range(len(unitary)):
                        # Pauli string validity check -length
                        if len(unitary[i]) % 2 != 0:
                            raise HamiltonianError('Error in Hamiltonian_in_Pauli_String! '
                                                   'Invalid observables input')
                        pauli_str = ['I' for i in range(qubits)]
                        # Iterate all type-position pairs
                        for observable_index in range(int(len(unitary[i]) / 2)):
                            # Pauli string validity check -element
                            if unitary[i][observable_index * 2] == 'I' or \
                                    unitary[i][observable_index * 2] == 'X' or \
                                    unitary[i][observable_index * 2] == 'Y' or \
                                    unitary[i][observable_index * 2] == 'Z':
                                # Check whether a conflict type of Pauli matrix is occurred to a qubit
                                if pauli_str[int(unitary[i][observable_index * 2 + 1])] == 'I' or \
                                        pauli_str[int(unitary[i][observable_index * 2 + 1])] == \
                                        unitary[i][observable_index * 2]:
                                    # Fill the pauli_str whose format satisfies unitary matrix
                                    pauli_str[int(unitary[i][observable_index * 2 + 1])] = unitary[i][
                                        observable_index * 2]
                                else:
                                    raise HamiltonianError('Error in Hamiltonian_in_Pauli_String! '
                                                           'Invalid observables input')
                            else:
                                raise HamiltonianError(
                                    'Error in Hamiltonian_in_Pauli_String! '
                                    'Invalid observables input')
                        str_ver_pauli_str = ''
                        for j in range(qubits):
                            if pauli_str[j] != 'I':
                                # Fill the unitary_str_ver whose format satisfies unitary_str_ver.
                                # In this step, the Pauli matrices are sorted based on the qubit.
                                str_ver_pauli_str += pauli_str[j] + str(j)
                        self.unitary.append(pauli_str)
                        self.unitary_str_ver.append(str_ver_pauli_str)
                # Parameter unitary may be in the format of type on fixed position matrix.
                elif type(unitary[0]) == list and type(unitary[0][0]) == str:
                    for pauli_str in unitary:
                        # Pauli string validity check -qubits
                        if len(pauli_str) != qubits:
                            raise HamiltonianError('Error in Hamiltonian_in_Pauli_String! '
                                                   'Invalid observables input')
                        str_ver_pauli_str = ''
                        for i in range(qubits):
                            # Pauli string validity check -element
                            if pauli_str[i] != 'X' and pauli_str[i] != 'Y' \
                                    and pauli_str[i] != 'Z' and pauli_str[i] != 'I':
                                raise HamiltonianError('Error in Hamiltonian_in_Pauli_String! '
                                                       'Invalid observables input')
                            else:
                                if pauli_str[i] != 'I':
                                    str_ver_pauli_str += pauli_str[i] + str(i)
                        self.unitary.append(pauli_str)
                        self.unitary_str_ver.append(str_ver_pauli_str)
        if hamiltonian_mat is not None:
            self.hamiltonian_mat = np.array(hamiltonian_mat)
        return

    def __add__(self, rhs):
        if isinstance(rhs, Hamiltonian_in_Pauli_String):
            new_unitary = self.unitary.copy()
            new_coefficient = self.coefficient.copy()
            for i in range(len(rhs.unitary)):
                if new_unitary.count(rhs.unitary[i]):
                    index = new_unitary.index(rhs.unitary[i])
                    new_coefficient[index] += rhs.coefficient[i]
                    if new_coefficient[index] == 0:
                        del new_unitary[index]
                        del new_coefficient[index]
                else:
                    new_unitary.append(rhs.unitary[i])
                    new_coefficient.append(rhs.coefficient[i])
            new_hamiltonian_mat = None
            if self.hamiltonian_mat is not None and rhs.hamiltonian_mat is not None:
                new_hamiltonian_mat = self.hamiltonian_mat + rhs.hamiltonian_mat
            return Hamiltonian_in_Pauli_String(qubits=max(self.qubits, rhs.qubits), unitary=new_unitary,
                                               coefficient=new_coefficient, hamiltonian_mat=new_hamiltonian_mat)
        else:
            raise HamiltonianError('Error! Invalid operation \'+\'.')

    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __sub__(self, rhs):
        if isinstance(rhs, Hamiltonian_in_Pauli_String):
            new_unitary = self.unitary.copy()
            new_coefficient = self.coefficient.copy()
            for i in range(len(rhs.unitary)):
                if new_unitary.count(rhs.unitary[i]):
                    index = new_unitary.index(rhs.unitary[i])
                    new_coefficient[index] -= rhs.coefficient[i]
                    if new_coefficient[index] == 0:
                        del new_unitary[index]
                        del new_coefficient[index]
                else:
                    new_unitary.append(rhs.unitary[i])
                    new_coefficient.append(-rhs.coefficient[i])
            new_hamiltonian_mat = None
            if self.hamiltonian_mat is not None and rhs.hamiltonian_mat is not None:
                new_hamiltonian_mat = self.hamiltonian_mat - rhs.hamiltonian_mat
            return Hamiltonian_in_Pauli_String(qubits=max(self.qubits, rhs.qubits), unitary=new_unitary,
                                               coefficient=new_coefficient, hamiltonian_mat=new_hamiltonian_mat)
        else:
            raise HamiltonianError('Error! Invalid operation \'-\'.')

    def __mul__(self, rhs):
        if isinstance(rhs, numbers.Real) or isinstance(rhs, numbers.Complex):
            coefficient = [self.coefficient[i] * rhs for i in range(len(self.coefficient))]
            new_hamiltonian_mat = None
            if self.hamiltonian_mat is not None:
                new_hamiltonian_mat = rhs * self.hamiltonian_mat
            return Hamiltonian_in_Pauli_String(self.qubits, unitary=self.unitary, coefficient=coefficient,
                                               hamiltonian_mat=new_hamiltonian_mat)
        elif isinstance(rhs, Hamiltonian_in_Pauli_String):
            if rhs.qubits != self.qubits:
                raise HamiltonianError('Error! The number of qubits should be equal while using operation \'*\'.')
            new_unitary = []
            new_coefficient = []
            for u_i in range(len(self.unitary)):
                for u_j in range(len(rhs.unitary)):
                    new_paulistring = ['I' for i in range(self.qubits)]
                    factor = 1
                    for i in range(self.qubits):
                        if self.unitary[u_i][i] != rhs.unitary[u_j][i]:
                            if self.unitary[u_i][i] == 'I':
                                new_paulistring[i] = rhs.unitary[u_j][i]
                            elif rhs.unitary[u_j][i] == 'I':
                                new_paulistring[i] = self.unitary[u_i][i]
                            if self.unitary[u_i][i] == 'X' and rhs.unitary[u_j][i] == 'Y':
                                new_paulistring[i] = 'Z'
                                factor *= 1j
                            if rhs.unitary[u_j][i] == 'X' and self.unitary[u_i][i] == 'Y':
                                new_paulistring[i] = 'Z'
                                factor *= -1j
                            if self.unitary[u_i][i] == 'X' and rhs.unitary[u_j][i] == 'Z':
                                new_paulistring[i] = 'Y'
                                factor *= -1j
                            if rhs.unitary[u_j][i] == 'X' and self.unitary[u_i][i] == 'Z':
                                new_paulistring[i] = 'Y'
                                factor *= 1j
                            if self.unitary[u_i][i] == 'Y' and rhs.unitary[u_j][i] == 'Z':
                                new_paulistring[i] = 'X'
                                factor *= 1j
                            if rhs.unitary[u_j][i] == 'Y' and self.unitary[u_i][i] == 'Z':
                                new_paulistring[i] = 'X'
                                factor *= -1j
                    coe = factor * self.coefficient[u_i] * rhs.coefficient[u_j]
                    if new_unitary.count(new_paulistring):
                        new_coefficient[new_unitary.index(new_paulistring)] += coe
                    else:
                        new_unitary.append(new_paulistring)
                        new_coefficient.append(coe)
            new_hamiltonian_mat = None
            if self.hamiltonian_mat is not None and rhs.hamiltonian_mat is not None:
                new_hamiltonian_mat = np.dot(self.hamiltonian_mat, rhs.hamiltonian_mat)
            return Hamiltonian_in_Pauli_String(self.qubits, unitary=new_unitary, coefficient=new_coefficient,
                                               hamiltonian_mat=new_hamiltonian_mat)
        else:
            raise HamiltonianError('Error! Invalid operation \'*\'.')

    def __rmul__(self, scalar):
        if isinstance(scalar, numbers.Real) or isinstance(scalar, numbers.Complex):
            return self.__mul__(scalar)
        else:
            raise HamiltonianError('Error! Invalid operation \'*\'.')

    def __pow__(self, power, modulo=None):
        if isinstance(power, int):
            if power >= 2:
                a = self * 1
                for i in range(power - 1):
                    a = a * self
                return a
            elif power == 1:
                return self * 1
            elif power == 0:
                return Hamiltonian_in_Pauli_String(qubits=self.qubits,
                                                   unitary=[['I' for i in range(self.qubits)]],
                                                   coefficient=[1],
                                                   hamiltonian_mat=np.eye(2**self.qubits))

    def ExpectationMeasurement(self, MeasurementMethod, test_circuit, active_qubits, shots=10000):
        # Result initialize
        res = 0
        for piece_index in range(len(self.unitary)):
            # Prepare Pauli string in the format of QuantumCircuit
            P = QuantumCircuit(self.qubits)
            for qubit_index in range(self.qubits):
                if self.unitary[piece_index][qubit_index] == 'X':
                    P.x(qubit_index)
                elif self.unitary[piece_index][qubit_index] == 'Y':
                    P.y(qubit_index)
                elif self.unitary[piece_index][qubit_index] == 'Z':
                    P.z(qubit_index)
            mea = MeasurementMethod(P=P, U=test_circuit, U_active_qubits=active_qubits, shots=shots)
            res += mea * self.coefficient[piece_index]
        return np.real(res)

    def getPauliStrings(self):
        return {'unitary': self.unitary, 'coefficient': self.coefficient}

    def getHamiltonianMatrix(self):
        return self.hamiltonian_mat

    def getEigenData(self, sort='+'):
        eigval, eigvec = scipy.linalg.eig(self.hamiltonian_mat)
        eigval = list(np.real(eigval))
        eigvec = list(eigvec)
        res = {'eigval': [],
               'eigvec': []}
        for i in range(2 ** self.qubits):
            if sort == '+':
                index = eigval.index(max(eigval))
            elif sort == '-':
                index = eigval.index(min(eigval))
            else:
                raise HamiltonianError('Error in Hamiltonian_in_Pauli_String, getEigenData. '
                                       'Invalid sort input.')
            res['eigval'].append(eigval[index])
            res['eigvec'].append(eigvec[index])
            del eigval[index]
            del eigvec[index]
        return res
