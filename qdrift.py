import numpy as np
from qiskit import(
  QuantumCircuit, QuantumRegister,
  execute,
  Aer)
from scipy.linalg import expm
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.aqua.operators import EvolvedOp
from qiskit.aqua.operators import OperatorBase
from qiskit.aqua.operators import MatrixOperator

def time_evolve_qubits(qubits, circuit, n, H, h, t, epsilon, num_reps=1):
    """
    Appends operations to a QuantumCircuit which evolve the qubits register
    under the given Hamiltonian by time t with precision epsilon.

    Uses the qDrift algorithm, described in this paper by Campbell:
        https://arxiv.org/pdf/1811.08017.pdf

    @param qubits   : The QuantumRegister representing the system
    @param circuit  : The QuantumCircuit to which we will append operations
    @param n        : Number of qubits in the system
    @param H        : The Pauli operators representing local Hamiltonians
                        which make up the system's Hamiltonian
    @param h        : The coefficients of the above terms
    @param t        : The amount of time to evolve the system by
    @param epsilon  : The desired precision of the circuit
    @param num_reps : Number of times to sample a circuit

    """

    # The number of qubits in the system
    n = len(qubits)

    # Sum of all weights
    # Campbell calls this lambda, but lambda is a protected keyword in python
    h   = np.abs(h)
    h  /= np.max(h)
    lam = np.sum(h)

    # The number of local Hamiltonians to sample, as per Campbell
    N = int(2 * lam * lam * t * t / epsilon)

    print('lambda:',lam, '   N:', N)

    # The matrix we build representing the time evolution
    H_matrix = np.eye(2 ** n)

    # Constant in matrix exponentiation
    tau = lam * t / N

    # Randomly sample local hamiltonians
    for i in range(N * num_reps):

        # Choose a random local Hamiltonian
        # using the assosiated weights in h
        local_H = np.random.choice(H, p = h/lam)

        # Append chosen local Hamiltonian to the
        # cumulative Hamiltonian matrix
        H_matrix = H_matrix @ expm(-1.j * tau * local_H.to_matrix() / num_reps)

    # Convert the Hamiltonian matrix into an operator and
    # add it to the circuit
    op = Operator(H_matrix)
    circuit.append(op.to_instruction(), range(n))

    return circuit

if __name__ == '__main__':

    # Tests the qdrift method by creating a circuit
    # which time-evolves an H2 molecule

    # Set up qubits and initial circuit
    n = 2
    qubits = QuantumRegister(2)
    circuit = QuantumCircuit(qubits)

    # Perform X on the first qubit to create the initial state of the system
    circuit.x(qubits[0])

    # List of local Hamiltonians
    # See https://arxiv.org/pdf/1512.06860.pdf#page=9&zoom=100,90,196
    H = np.array([Pauli.from_label('II'),
                  Pauli.from_label('ZI'),
                  Pauli.from_label('IZ'),
                  Pauli.from_label('ZZ'),
                  Pauli.from_label('YY'),
                  Pauli.from_label('XX')])

    # List of corresponding coefficients
    h = np.array([-0.1927, 0.2048, -0.0929, 0.4588, 0.1116, 0.1116])

    t = 1
    epsilon = 0.01

    circuit = time_evolve_qubits(qubits, circuit, n, H, h, t, epsilon)
    print(circuit)
