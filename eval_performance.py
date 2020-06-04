from test import run_exp 
import numpy as np
from tqdm import tqdm 

import sys
from contextlib import contextmanager


@contextmanager
def custom_redirection(fileobj):
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old


N = 100
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

print("Average qdrift fidelity:", np.mean(fidelity_qdrift))
print("Average suzuki fidelity:", np.mean(fidelity_suzuki))

print("Saving Results!")
np.save("qdrift_fidelities.npy", fidelity_qdrift)
np.save("suzuki_fidelities.npy", fidelity_suzuki)


