from quantumpoissonreadout import qiskit_readout

r = qiskit_readout(N=4, rho=10.0, eps=1e-2)          # one circuit, state vector
print(f"qubits           = {r['qubits']}")
print(f"CX gates         = {r['cx']:,}")
print(f"amplitude A      = {r['A']:.8f}")
print(f"estimate         = {r['estimate']:.8f}")
print(f"sigma_H (FE)     = {r['fe']:.8f}")
