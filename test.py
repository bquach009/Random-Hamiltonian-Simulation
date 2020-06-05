import numpy as np
from scipy.linalg import expm
from qiskit import BasicAer
from qiskit import execute as q_execute
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.operators import MatrixOperator, op_converter
from qiskit.aqua.components.initial_states import Custom
from qiskit.quantum_info.operators.pauli import Pauli

from qdrift import time_evolve_qubits

def run_exp(num_reps=1):
    # choice of Hi
    H_basis = [Pauli.from_label('II'),
                Pauli.from_label('ZI'),
                Pauli.from_label('IZ'),
                Pauli.from_label('ZZ'),
                Pauli.from_label('YY'),
                Pauli.from_label('XX')]


    num_qubits = 2
    evo_time = 1
    epsilon = 0.01
    L = 10    ## number of sum

    ############################################################
    # generate a random Hamiltonian H as the sum of m basis Hi operators
    ############################################################

    hs = np.random.random(L)
    indexes = np.random.randint(low=0, high=6, size=L)

    ## H in matrix form
    H_matrix = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    ## H as a list of pauli operators (unweighted)
    H_list = []
    for i in range(L):
        H_matrix = H_matrix + hs[i] * H_basis[indexes[i]].to_matrix()
        H_list.append(H_basis[indexes[i]])
    print('matrix H: \n', H_matrix)
    print('\n')
    # H as a pauli operator
    H_qubitOp = op_converter.to_weighted_pauli_operator(MatrixOperator(matrix=H_matrix))

    # generate an initial state
    state_in = Custom(num_qubits, state='random')


    ############################################################
    # ground truth and benchmarks
    ############################################################

    # ground truth
    state_in_vec = state_in.construct_circuit('vector')
    groundtruth = expm(-1.j * H_matrix * evo_time) @ state_in_vec
    print('The directly computed groundtruth evolution result state is\n{}.'.format(groundtruth))
    print('\n')

    # simulated through Qiskit's evolve algorithm, which based on Trotter-Suzuki.
    quantum_registers = QuantumRegister(num_qubits)
    circuit = state_in.construct_circuit('circuit', quantum_registers)
    circuit += H_qubitOp.evolve(
        None, evo_time, num_time_slices=1,
        quantum_registers=quantum_registers,
        expansion_mode='suzuki',
        expansion_order=1
    )

    backend = BasicAer.get_backend('statevector_simulator')
    job = q_execute(circuit, backend)
    circuit_execution_result = np.asarray(job.result().get_statevector(circuit))
    print('The simulated (suzuki) evolution result state is\n{}.'.format(circuit_execution_result))
    print('\n')

    # the distance between the ground truth and the simulated state
    # measured by "Fidelity"
    fidelity_suzuki = state_fidelity(groundtruth, circuit_execution_result)
    print('Fidelity between the groundtruth and the circuit result states is {}.'.format(fidelity_suzuki))
    print('\n')


    ############################################################
    # our method
    ############################################################


    quantum_registers = QuantumRegister(num_qubits)
    circuit = state_in.construct_circuit('circuit', quantum_registers)

    # run our qdrift algorithm
    circuit = time_evolve_qubits(quantum_registers, circuit, num_qubits, H_list, hs, evo_time, epsilon, num_reps)

    backend = BasicAer.get_backend('statevector_simulator')
    job = q_execute(circuit, backend)
    circuit_execution_result = np.asarray(job.result().get_statevector(circuit))
    print('The simulated (qdrift) evolution result state is\n{}.'.format(circuit_execution_result))
    print('\n')


    # the distance between the ground truth and the simulated state
    # measured by "Fidelity"
    fidelity_qdrift = state_fidelity(groundtruth, circuit_execution_result)
    print('Fidelity between the groundtruth and the circuit result states is {}.'.format(fidelity_qdrift))
    print('\n')


    print('benchmark, suzuki:', fidelity_suzuki)
    print('qdrift:', fidelity_qdrift)
    return fidelity_qdrift, fidelity_suzuki 

if __name__ == "__main__":
    run_exp()



