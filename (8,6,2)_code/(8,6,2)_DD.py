import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

# Registers for [[8, 6, 2]]
def prep_qubits():
    data = QuantumRegister(8, 'data')
    anc = QuantumRegister(4, 'anc') # stabilizers
    c = ClassicalRegister(2, 'c')

    qc = QuantumCircuit(data, anc, c)
    return qc, data, anc, c 

# DFS Encoding
def encode_dfs_logical_zero(qc, data):
    for i in range(1, 8):
        qc.cx(data[0], data[i])

    for q in data:
        qc.h(q)

    for i in range(1, 8):
        qc.cx(data[0], data[i])
    
    for q in data:
        qc.h(q)

    return qc

# Dynamical Decoupling
def dynamical_decoupling(qc, data):
    for q in data:
        qc.x(q)
    
    qc.barrier()

    for q in data:
        qc.y(q)

    return qc

# Stabilizer Measurements
def stabilizer_measurements(qc, data, anc, c):
    # Z Parity
    for i in range(8):
        qc.cx(data[i], anc[0])
    
    qc.measure(anc[0], c[0])
    qc.reset(anc[0])

    # X Parity
    for i in range(8):
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])

    qc.measure(anc[1], c[1])
    qc.reset(anc[1])

    return qc

# Leakage Model
def leakage_noise_model(p = 0.02, p_leak = 0.01):
    pauli = pauli_error([
        ('X', p/3),
        ('Y', p/3),
        ('Z', p/3),
        ('I', 1-p),
    ])

    leak = pauli_error([
        ('I', 1-p_leak),
        ('Z', p_leak)
    ])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(pauli, ['h'])
    noise_model.add_all_qubit_quantum_error(pauli.tensor(pauli), ['cx'])
    noise_model.add_all_qubit_quantum_error(leak, ['h'])

    return noise_model

# Leakage Elimination Operation (LEO)
def leakage_elimination_operation(qc, data):
    for q in data:
        qc.z(q)
    return qc

# Mid-circuit measurement
def circuit_mid_circuit(shots = 2000, p = 0.02, p_leak = 0.01):
    qc, data, anc, c = prep_qubits()
    encode_dfs_logical_zero(qc, data)

    dynamical_decoupling(qc, data)
    leakage_elimination_operation(qc, data)
    qc.barrier()

    stabilizer_measurements(qc, data, anc, c)

    dynamical_decoupling(qc, data)
    leakage_elimination_operation(qc, data)
    qc.barrier()

    sim = AerSimulator(
        method = 'density_matrix',
        noise_model = leakage_noise_model(p, p_leak)
    )

    return sim.run(qc, shots = shots).result().get_counts()

# Deferred measurement
def circuit_deferred(shots = 2000, p = 0.02, p_leak = 0.01):
    qc, data, anc, c = prep_qubits()
    encode_dfs_logical_zero(qc, data)

    dynamical_decoupling(qc, data)
    leakage_elimination_operation(qc, data)
    qc.barrier()

    for i in range(8):
        qc.cx(data[i], anc[0])
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])

    qc.measure(anc[0], c[0])
    qc.measure(anc[1], c[1])

    sim = AerSimulator(
        method = 'density_matrix',
        noise_model = leakage_noise_model(p, p_leak)
    )

    return sim.run(qc, shots = shots).result().get_counts()

# Control Measurement
def circuit_control(shots = 2000, p = 0.02, p_leak = 0.01):
    qc, data, _, c = prep_qubits()
    encode_dfs_logical_zero(qc, data)

    dynamical_decoupling(qc, data)
    leakage_elimination_operation(qc, data)

    qc.measure(data[0], c[0])

    sim = AerSimulator(
        method = 'density_matrix',
        noise_model = leakage_noise_model(p, p_leak)
    )

    return sim.run(qc, shots = shots).result().get_counts()

# Postselection Metric
def postselection_rate(counts):
    total = sum(counts.values())
    valid = counts.get('00', 0)
    return valid / total if total > 0 else 0

def run_benchmarks(noise_vals, shots = 2000):
    mid, deferred, control = [], [], []

    for p in noise_vals:
        mid.append(postselection_rate(circuit_mid_circuit(shots, p)))
        deferred.append(postselection_rate(circuit_deferred(shots, p)))
        control.append(postselection_rate(circuit_control(shots, p)))

    return mid, deferred, control

if __name__ == "__main__":
    noise_vals = [0.0, 0.01, 0.02, 0.05]
    shots = 2000

    mid, deferred, control = run_benchmarks(noise_vals, shots)

    print("p_noise | mid-circuit | deferred | control")
    print("-------------------------------------------")
    for i, p in enumerate(noise_vals):
        print(
            f"{p:6.3f} | "
            f"{mid[i]:10.3f} | "
            f"{deferred[i]:8.3f} | "
            f"{control[i]:8.3f}"
        )
