import numpy as np
from qiskit import *
from qiskit import Aer


class SelfDefinedAnsatzeError(Exception):
    pass


class AnsatzeElementList:
    def __init__(self, uid, position: int, type, par, partial_flag):
        """
        Class of Ansatze unit which store the information of a rotation Pauli gate with angle par.
        :param uid: An integer to identify the Ansatze unit.
        :param position: An integer to identify the Ansatze unit.
        :param type: The type of Pauli rotation gate.
        :param par: The angle of Pauli rotation gate.
        :param partial_flag: Whether the parameter is chosen to derive a partial derivative.

        Mathematical representation:
                     Pauli matrix    | type
            P =         X            | 'X'
                        Y            | 'Y'
                        Z            | 'Z'

                            matrix                      |    partial_flag
            U(par) =    exp(-i*par/2*P)                 |          0
                        exp(-i*par/2*P)exp(-i*pi/2*P)   |          1
        """
        self.uid = uid
        self.type = type
        self.par = par
        self.partial_flag = partial_flag


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
        if not self.checkParameter():
            raise SelfDefinedAnsatzeError("Error in Ansatze, incorrect number of parameter.")

    def setParameter(self, parameter: list):
        self.parameter = parameter
        if not self.checkParameter():
            raise SelfDefinedAnsatzeError("Error in Ansatze, incorrect number of parameter.")

    def circuit(self, partial_flag: bool, id: int):
        ansatze = QuantumCircuit(self.statescale)
        return ansatze

    def checkParameter(self):
        return True

    def getParameter(self):
        return self.parameter
