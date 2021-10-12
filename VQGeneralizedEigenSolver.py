from qiskit import *

from OptTracer import Record_Print_Tracer
from QMeasure import HadamardTest, HadamardTest_Analytical
from QAnsatz import Ansatze, HardwareEfficientAnsatze_halflayer
from QCircuit import MixedStateGenerationCircuit
from QHamiltonian import Hamiltonian_in_Pauli_String
from QSubspaceEigensolver import SubspaceEigSolver
from GradientBasedOptimization import steepest
import random
import numpy as np


class VQGE_Error(Exception):
    pass


class StateGenerationError(Exception):
    pass


class VQGeneralizedEigenSolver:
    """
    Hamiltonian is in the format of the sum of weighted Pauli string
    Solving Ax = 入Bx
    """

    def __init__(self, A_Hamiltonian: Hamiltonian_in_Pauli_String, B_Hamiltonian: Hamiltonian_in_Pauli_String,
                 m, ansatze: Ansatze, weight_list: list):
        """
        Class of state subspace eigen solver.
        :param A_Hamiltonian: A in Ax = 入Bx
        :param B_Hamiltonian: B in Ax = 入Bx
        :param ansatze: The quantum circuit network with respect to parameters.
        :param weight_list: The guidance of subspace searching. c.f. Von Neumann theorem.
        """
        if A_Hamiltonian.qubits != B_Hamiltonian.qubits:
            raise VQGE_Error('The qubits of Hamiltonian A and B are not equal')
        self.state_scale = A_Hamiltonian.qubits
        if len(weight_list) > 2 ** self.state_scale:
            raise VQGE_Error('Error in VQGeneralizedEigenSolver! Incorrect weight list size')
        self.ansatze = ansatze
        self.weight_list = weight_list
        self.A = A_Hamiltonian
        self.B = B_Hamiltonian
        self.m = m

    def RayleighQuotientIteration(self, m=None, iterations=5, opt=steepest,
                                  trace=True, opt_options: dict = None, shots=10000):
        if m is not None:
            self.m = m
        opt_alpha = 0.05
        opt_callback = None
        opt_iterations = 50
        opt_direct = '+'
        opt_tol = 1e-5
        if opt_options is not None:
            if opt_options.get('alpha') is not None:
                opt_alpha = opt_options['alpha']
            if opt_options.get('callback') is not None:
                opt_callback = opt_options['callback']
            if opt_options.get('iterations') is not None:
                opt_iterations = opt_options['iterations']
            if opt_options.get('direct') is not None:
                opt_direct = opt_options['direct']
            if opt_options.get('tol') is not None:
                opt_tol = opt_options['tol']
        if m > 2 ** self.state_scale:
            raise VQGE_Error('Error in VQGeneralizedEigenSolver! Incorrect m input. '
                             'Parameter m should be 2**self.state_scale at most')
        sigma = np.array([1 for i in range(self.m)])
        d_sigma = np.array([1 for i in range(self.m)])
        res_parameter = [self.getParameter() for i in range(self.m)]
        sigma_tracer = []
        print('Start generalized eigenvalue calculating iterations:')
        for it in range(iterations):
            H = [self.A - sigma[i] * self.B for i in range(self.m)]
            for eig in range(self.m):
                print('Generalized eigenvalue calculating: it = ' + str(it) + ' eig = ' + str(eig))
                self.setParameter([np.pi / 2 for i in range(self.ansatze.getParameterLength())])
                Solver = SubspaceEigSolver(Hamiltonian=H[eig], ansatze=self.ansatze, weight_list=self.weight_list)
                print('Optimizing!')
                if opt_callback is None:
                    def opt_callback(xk):
                        print(Solver.getLossFunctionAnalytical(xk))
                res_parameter[eig] = opt(Solver.getLossFunctionAnalytical,
                                         Solver.GetJacobian,
                                         self.getParameter(),
                                         alpha=opt_alpha,
                                         callback=opt_callback,
                                         iters=opt_iterations,
                                         direct=opt_direct,
                                         tol=opt_tol)
                print('Optimization finished in it = ' + str(it) + ' eig = ' + str(eig) + '!')
        print('Start generalized eigenvector calculating:')
        H = [(self.A - sigma[i] * self.B)**2 for i in range(self.m)]
        gvec_tracer = []
        for eig in range(self.m):
            print('Generalized eigenvector calculating: eig = ' + str(eig))
            Solver = SubspaceEigSolver(Hamiltonian=H[eig], ansatze=self.ansatze, weight_list=[1000, 1, 1, 1, 1, 1, 1, 1])
            self.ansatze.setParameter([np.pi / 2 for i in range(len(Solver.getParameter()))])
            state_tracer = Record_Print_Tracer([Solver.getEigenData, Solver.getLossFunctionAnalytical])
            print('Optimizing!')
            res_parameter[eig] = opt(Solver.getLossFunctionAnalytical,
                                     Solver.GetJacobian,
                                     self.getParameter(),
                                     alpha=opt_alpha,
                                     callback=state_tracer.callback,
                                     iters=opt_iterations,
                                     direct='-',
                                     tol=opt_tol)
            gvec_tracer.append([state_tracer.tracedata[i]['eigvec'] for i in range(len(state_tracer.tracedata))])
        if trace:
            return res_parameter, sigma, sigma_tracer, gvec_tracer
        return res_parameter, sigma

    def setParameter(self, new_parameter: list):
        self.ansatze.setParameter(new_parameter)

    def getParameter(self):
        return self.ansatze.getParameter()



