from test import run_exp 
import numpy as np

N = 100
fidelity_qdrift = []
fidelity_suzuki = []
for _ in range(N):
    res_qdrift, res_suzuki = run_exp()
    fidelity_qdrift.append(res_qdrift)
    fidelity_suzuki.append(res_suzuki)

print("Average qdrift fidelity:", np.mean(fidelity_qdrift))
print("Average suzuki fidelity:", np.mean(fidelity_suzuki))

print("Saving Results!")
np.save("qdrift_fidelities.npy", fidelity_qdrift)
np.save("suzuki_fidelities.npy", fidelity_suzuki)


