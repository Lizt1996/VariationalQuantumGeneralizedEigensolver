import numpy as np
from qiskit import *
from qiskit import Aer


def StateInit():
    initcirc = QuantumCircuit(5)
    initcirc.h(1)
    initcirc.h(2)
    initcirc.z(1)
    initcirc.y(2)
    initcirc.cnot(1, 2)
    initcirc.y(3)
    initcirc.x(4)
    initcirc.h(3)
    initcirc.cnot(3, 4)
    initcirc.rz(1 / 3 * np.pi, 4)
    return initcirc


class QBSTLikelihoodCircuit:
    def __init__(self, statescale: int, initcirc: QuantumCircuit):
        self.theta = 0
        self.statescale = statescale
        self.initcirc = initcirc

    def Parameter_Set_theta(self, t):
        self.theta = t % (2 * np.pi)

    def BayesianSwapTestCircuit(self):
        BayesianQC = QuantumCircuit(1+self.statescale*2, 1)
        BayesianQC.h(0)
        BayesianQC.rz(self.theta, 0)
        for i in range(1, self.statescale+1):
            BayesianQC.cswap(0, i, self.statescale+i)
        BayesianQC.h(0)
        BayesianQC.barrier(0)
        BayesianQC.measure(0, 0)
        return BayesianQC

    def BSTMeasurement(self, shotnum):
        qc = self.initcirc + self.BayesianSwapTestCircuit()
        backend_sim = Aer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend_sim, shots=shotnum)
        result_sim = job_sim.result()
        return result_sim.get_counts(qc)

    def BSTMSingleShot(self):
        res = self.BSTMeasurement(1)
        if res.get('0') is None:
            return 1
        return 0
