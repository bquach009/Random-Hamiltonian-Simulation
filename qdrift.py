import numpy as np
from qiskit import(
  QuantumCircuit, QuantumRegister,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.aqua.operators import EvolvedOp
from qiskit.aqua.operators import OperatorBase

def time_evolve_qubit(t, epsilon):

    n = 2 # number of qubits
    circuit = QuantumCircuit(QuantumRegister(2))

    # List of local Hamiltonians
    # See https://arxiv.org/pdf/1512.06860.pdf#page=9&zoom=100,90,196
    H = np.array([Pauli.from_label('II'),
                  Pauli.from_label('IX'),
                  Pauli.from_label('XI'),
                  Pauli.from_label('XX'),
                  Pauli.from_label('YY'),
                  Pauli.from_label('ZZ')])

    # List of corresponding coefficients
    # TODO: I made these up, replace with the ones from
    # https://arxiv.org/pdf/1001.3855.pdf, Section 7
    h = np.array([1, 2, 3, 4])

    # Sum of all weights
    # Campbell calls this lambda, but lambda is a protected keyword in python
    lam = np.sum(h)

    # The number of local Hamiltonians to sample, as per Campbell
    N = int(2 * lam * lam * t * t / epsilon)

    print(lam, N)

    # Randomly sample local hamiltonians
    for i in range(3): # TODO: replace with N

        random_index = np.random.choice(
            range(len(h)), # The list [0, 1, ..., len(h) - 1]
            p = h/lam      # The weights are given by h
        )

        # Create an operation which is exp(i H[j] tau), as per Campbell
        op = EvolvedOp(
            H[random_index],    # Convert `Pauli` type to `Operator` type
            -1 / N * lam * t,   # `tau` coefficient, negative because EvolvedOp also applies a negative
                                #TODO: for sure about the negative?
        ).to_matrix_op()

        # print(type(op.to_instruction()))
        # print(op.to_instruction().to_matrix())

        # Add the operator to our circuit
        circuit.append(op.to_instruction(), [0, 1])

    print(circuit)

if __name__ == '__main__':
    time_evolve_qubit(1, 0.1)
