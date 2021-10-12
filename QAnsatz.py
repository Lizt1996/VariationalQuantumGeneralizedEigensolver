import numpy as np
from qiskit import *


class SelfDefinedQuantumAnsatzError(Exception):
    pass


class Ansatze:
    def __init__(self, statescale: int, layer: int, parameter: list):
        """
        Class of Ansatze for variational quantum algorithm
        This class should be inherited
        :param statescale: the number of qubit to represent the solution.
        :param layer: the number of layer
        :param parameter: the parameter list
        """
        self.statescale = statescale
        self.layer = layer
        self.parameter = parameter
        self.typelist = ['X' for i in range(self.parameter.__len__())]
        self.positionlist = [-1 for i in range(self.parameter.__len__())]
        if not self.checkParameter():
            raise SelfDefinedQuantumAnsatzError("Error in Ansatze, incorrect number of parameter.")

    def setParameter(self, parameter: list):
        self.parameter = parameter
        if not self.checkParameter():
            raise SelfDefinedQuantumAnsatzError("Error in Ansatze, incorrect number of parameter.")

    def circuit(self, partial_flag: bool = False, pid: int = 0, pn=None):
        ansatze = QuantumCircuit(self.statescale)
        return ansatze

    def checkParameter(self):
        return True

    def getParameter(self):
        return self.parameter

    def getParameterLength(self):
        return self.parameter.__len__()


def LAAnsatzeCirc(statescale: int, layer: int, parameter: list):
    parameter_num = statescale * 2 + layer * (statescale * 4)
    if parameter_num != parameter.__len__():
        raise SelfDefinedQuantumAnsatzError("Error in Ansatze, incorrect number of parameter.")
    ansatze = QuantumCircuit(statescale)
    for i in range(statescale):
        ansatze.rz(parameter[i], i)
    for i in range(statescale):
        ansatze.ry(parameter[statescale + i], i)
    for lay in range(layer):
        for i in range(statescale):
            ansatze.cnot(i, (i + 1) % statescale)
            ansatze.rz(parameter[2 * statescale + lay * statescale * 4 + 4 * i], i)
            ansatze.rz(parameter[2 * statescale + lay * statescale * 4 + 4 * i + 1], (i + 1) % statescale)
            ansatze.ry(parameter[2 * statescale + lay * statescale * 4 + 4 * i + 2], i)
            ansatze.ry(parameter[2 * statescale + lay * statescale * 4 + 4 * i + 3], (i + 1) % statescale)
    for i in range(statescale):
        ansatze.cnot((i - 1) % statescale, i)
    return ansatze


class LAAnsatze(Ansatze):
    def circuit(self):
        ansatze = QuantumCircuit(self.statescale)
        for i in range(self.statescale):
            ansatze.rz(self.parameter[i], i)
        for i in range(self.statescale):
            ansatze.ry(self.parameter[self.statescale + i], i)
        for lay in range(self.layer):
            for i in range(self.statescale):
                ansatze.cnot(i, (i + 1) % self.statescale)
                ansatze.rz(self.parameter[2 * self.statescale + lay * self.statescale * 4 + 4 * i], i)
                ansatze.rz(self.parameter[2 * self.statescale + lay * self.statescale * 4 + 4 * i + 1],
                           (i + 1) % self.statescale)
                ansatze.ry(self.parameter[2 * self.statescale + lay * self.statescale * 4 + 4 * i + 2], i)
                ansatze.ry(self.parameter[2 * self.statescale + lay * self.statescale * 4 + 4 * i + 3],
                           (i + 1) % self.statescale)
        for i in range(self.statescale):
            ansatze.cnot((i - 1) % self.statescale, i)
        return ansatze

    def checkParameter(self):
        parameter_num = self.statescale * 2 + self.layer * (self.statescale * 4)
        if parameter_num != self.parameter.__len__():
            return False
        return True


class HardwareEfficientAnsatze(Ansatze):
    """
    Class of Ansatze for variational quantum algorithm
    parameter.size = 3*systemscale+layer*3*systemscale
    """

    def circuit(self):
        ansatze = QuantumCircuit(self.statescale)
        for qn in range(self.statescale):
            ansatze.rz(self.parameter[qn], qn)
        for qn in range(self.statescale):
            ansatze.ry(self.parameter[qn + self.statescale], qn)
        for qn in range(self.statescale):
            ansatze.rz(self.parameter[qn + 2 * self.statescale], qn)

        for lay in range(self.layer):
            for qn in range(self.statescale):
                ansatze.cnot(qn, (qn + 1) % self.statescale)
            for qn in range(self.statescale):
                ansatze.rz(self.parameter[qn + 3 * self.statescale + lay * 3 * self.statescale], qn)
            for qn in range(self.statescale):
                ansatze.ry(self.parameter[qn + 4 * self.statescale + lay * 3 * self.statescale], qn)
            for qn in range(self.statescale):
                ansatze.rz(self.parameter[qn + 5 * self.statescale + lay * 3 * self.statescale], qn)
        return ansatze

    def checkParameter(self):
        parameter_num = 3 * self.statescale + self.layer * 3 * self.statescale
        if parameter_num != self.parameter.__len__():
            return False
        return True


class HardwareEfficientAnsatze_halflayer(Ansatze):
    def circuit(self, partial_flag: bool = False, pid: int = 0, pn=None):
        ang = 0
        if partial_flag:
            if pn == '+':
                ang = np.pi / 2
            elif pn == '-':
                ang = -np.pi / 2
            else:
                raise SelfDefinedQuantumAnsatzError("Error in Ansatze, incorrect parameter input -pn. Only '+' and "
                                                    "'-' are available")
        ansatze = QuantumCircuit(self.statescale)
        blocknum = np.math.floor(self.statescale / 2)
        for lay in range(self.layer):
            if lay % 2 == 0:
                for i in range(blocknum):
                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12], (2 * i) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 == pid:
                        ansatze.rz(ang, (2 * i) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 1], (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 1 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 2], (2 * i) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 2 == pid:
                        ansatze.ry(ang, (2 * i) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 3], (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 3 == pid:
                        ansatze.ry(ang, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 4], (2 * i) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 4 == pid:
                        ansatze.rz(ang, (2 * i) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 5], (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 5 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.cnot((2 * i) % self.statescale, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 6], (2 * i) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 6 == pid:
                        ansatze.rz(ang, (2 * i) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 7], (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 7 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 8], (2 * i) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 8 == pid:
                        ansatze.ry(ang, (2 * i) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 9], (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 9 == pid:
                        ansatze.ry(ang, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 10], (2 * i) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 10 == pid:
                        ansatze.rz(ang, (2 * i) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 11], (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 11 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)
            else:
                for i in range(blocknum):
                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12],
                               (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 1],
                               (2 * i + 2) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 1 == pid:
                        ansatze.rz(ang, (2 * i + 2) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 2],
                               (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 2 == pid:
                        ansatze.ry(ang, (2 * i + 1) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 3],
                               (2 * i + 2) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 3 == pid:
                        ansatze.ry(ang, (2 * i + 2) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 4],
                               (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 4 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 5],
                               (2 * i + 2) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 5 == pid:
                        ansatze.rz(ang, (2 * i + 2) % self.statescale)

                    ansatze.cnot((2 * i + 1) % self.statescale, (2 * i + 2) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 6],
                               (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 6 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 7],
                               (2 * i + 2) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 7 == pid:
                        ansatze.rz(ang, (2 * i + 2) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 8],
                               (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 8 == pid:
                        ansatze.ry(ang, (2 * i + 1) % self.statescale)

                    ansatze.ry(self.parameter[blocknum * lay * 12 + i * 12 + 9],
                               (2 * i + 2) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 9 == pid:
                        ansatze.ry(ang, (2 * i + 2) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 10],
                               (2 * i + 1) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 10 == pid:
                        ansatze.rz(ang, (2 * i + 1) % self.statescale)

                    ansatze.rz(self.parameter[blocknum * lay * 12 + i * 12 + 11],
                               (2 * i + 2) % self.statescale)
                    if partial_flag and blocknum * lay * 12 + i * 12 + 11 == pid:
                        ansatze.rz(ang, (2 * i + 2) % self.statescale)
        return ansatze

    def checkParameter(self):
        semilayer = np.math.floor(self.statescale / 2)
        parameter_num = self.layer * semilayer * 12
        if parameter_num != self.parameter.__len__():
            return False
        return True
