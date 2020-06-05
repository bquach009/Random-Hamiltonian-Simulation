from test import run_exp 
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

import sys
from contextlib import contextmanager

# Context manager for surpressing output of the tests into a log file
@contextmanager
def custom_redirection(fileobj):
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

num_repetitions = list(range(1, 11))
avg_qdrift_fidelity = []

# Plot the average fidelity against number of repetitions. 
for num_reps in num_repetitions:
    # Number of trials to run the qdrift algorithm
    N = 10

    # Stores the resulting fidelities with ground truth 
    fidelity_qdrift = []
    fidelity_suzuki = []

    pbar = tqdm(range(N))
    for _ in pbar:
        with open('log.txt', 'w') as out:
            with custom_redirection(out):
                res_qdrift, res_suzuki = run_exp(num_reps=num_reps)
                fidelity_qdrift.append(res_qdrift)
                fidelity_suzuki.append(res_suzuki)
        pbar.set_description("Average qdrift fidelity: {} Average suzuki fidelity: {}"\
            .format(np.mean(fidelity_qdrift), np.mean(fidelity_suzuki)))

    # Print resulting average fidelity
    print("{} Repitions!".format(num_reps))
    print("Average qdrift fidelity:", np.mean(fidelity_qdrift))
    print("Average suzuki fidelity:", np.mean(fidelity_suzuki))

    avg_qdrift_fidelity.append(np.mean(fidelity_qdrift))

    # Update the plot 
    plt.figure()
    plt.plot(list(range(1, len(avg_qdrift_fidelity) + 1)), avg_qdrift_fidelity)
    plt.xlabel("Number of Repetitions")
    plt.ylabel("Average Fidelity With Ground Truth")
    plt.savefig("fidelity_performance.png")



