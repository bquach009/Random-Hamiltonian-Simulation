from test import run_exp 
import numpy as np
from tqdm import tqdm 

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


# Number of trials to run the qdrift algorithm
N = 100

# Stores the resulting fidelities with ground truth 
fidelity_qdrift = []
fidelity_suzuki = []

pbar = tqdm(range(N))
for _ in pbar:
    with open('log.txt', 'w') as out:
        with custom_redirection(out):
            res_qdrift, res_suzuki = run_exp()
            fidelity_qdrift.append(res_qdrift)
            fidelity_suzuki.append(res_suzuki)
    pbar.set_description("Average qdrift fidelity: {} Average suzuki fidelity: {}"\
        .format(np.mean(fidelity_qdrift), np.mean(fidelity_suzuki)))

# Print resulting average fidelity
print("Average qdrift fidelity:", np.mean(fidelity_qdrift))
print("Average suzuki fidelity:", np.mean(fidelity_suzuki))

print("Saving Results!")

# Save results
np.save("qdrift_fidelities.npy", fidelity_qdrift)
np.save("suzuki_fidelities.npy", fidelity_suzuki)


