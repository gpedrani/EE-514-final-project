import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import transpile

service = QiskitRuntimeService(
    token = 'QjPLVURBC3hy6ySgd_bT_IOGAyvkXcajvAG3l5IMzR4t',
    channel = 'ibm_quantum_platform'
)

def get_backend():
    backend = service.least_busy(
        operational = True,
        simulator = False,
        min_num_qubits = 11,
        dynamic_circuits = True
    )
    print("Using backend:", backend.name)
    return backend

# Registers for [[8, 6, 2]]
def prep_qubits():
    data = QuantumRegister(8, 'data')
    anc = QuantumRegister(2, 'anc') # stabilizers
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

# Leakage Elimination Operation (LEO)
def leakage_elimination_operation(qc, data):
    for q in data:
        qc.z(q)
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

# Mid-circuit measurement
def circuit_mid_circuit():
    qc, data, anc, c = prep_qubits()
    encode_dfs_logical_zero(qc, data)

    leakage_elimination_operation(qc, data)
    qc.barrier()

    stabilizer_measurements(qc, data, anc, c)
    leakage_elimination_operation(qc, data)

    return qc

# Deferred measurement
def circuit_deferred():
    qc, data, anc, c = prep_qubits()
    encode_dfs_logical_zero(qc, data)

    for i in range(8):
        qc.cx(data[i], anc[0])
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])

    qc.measure(anc[0], c[0])
    qc.measure(anc[1], c[1])

    return qc

# Control Measurement
def circuit_control():
    qc, data, _, c = prep_qubits()
    encode_dfs_logical_zero(qc, data)

    qc.measure(data[0], c[0])
    qc.measure(data[1], c[1])

    return qc

def run_on_hardware(qc, shots = 2000):
    backend = get_backend()
    tqc = transpile(qc, backend = backend, optimization_level = 1)

    sampler = Sampler(backend)
    job = sampler.run([tqc], shots = shots)
    result = job.result()

    data = result[0].data

    if 'c' in data:
        counts = data['c'].get_counts()
    else:
        counts = result[0].join_data().get_counts()

    return counts


# Postselection Metric
def postselection_rate(counts):
    total = sum(counts.values())
    valid = counts.get('00', 0)
    return valid / total if total > 0 else 0

def run_benchmarks(shots = 2000):
    circuits = {
        'mid-circuit': circuit_mid_circuit(),
        'deferred': circuit_deferred(),
        'control': circuit_control()
    }

    results = {}

    for name, qc in circuits.items():
        print(f'Running {name}...')
        counts = run_on_hardware(qc, shots)
        results[name] = postselection_rate(counts)

    return results

if __name__ == '__main__':
    results = run_benchmarks(shots = 2000)

    print('\nPostselection survival probabilities:')
    print('------------------------------------')
    for k, v in results.items():
        print(f'{k:12s}: {v:.3f}')