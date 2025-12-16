from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

def prep_qubits():
    data = QuantumRegister(6, 'data')
    anc = QuantumRegister(4, 'ancilla')
    c = ClassicalRegister(2, 'c') # to ensure mid-circuit measurement works

    qc = QuantumCircuit(data, anc, c)
    return qc, data, anc, c

def encode_logical_zero(qc, data):
    # GHZ preparation (equivalent to stabilizer projector P+)
    qc.h(data[0])
    qc.cx(data[0], data[1])
    qc.cx(data[0], data[2])
    qc.cx(data[0], data[3])
    qc.cx(data[0], data[4])
    qc.cx(data[0], data[5])
    return qc

def s1_meas(qc, q, anc, cbit):
    for i in range(6):
        qc.cx(q[i], anc)
    qc.measure(anc, cbit)

def s2_meas(qc, q, anc, cbit):
    for i in range(6):
        qc.h(q[i])
        qc.cx(q[i], anc)
        qc.h(q[i])
    qc.measure(anc, cbit)


def dynamical_decoupling(qc, data):
    for q in data:
        qc.x(q)
    qc.barrier()
    for q in data:
        qc.y(q)
    return qc


def stab_measurements(qc, data, anc, c):
    s1_meas(qc, data, anc[0], c[0])
    qc.reset(anc[0])
    s2_meas(qc, data, anc[1], c[1])
    qc.reset(anc[1])
    return qc

def leakage_ops():
    single_X = [('X', i) for i in range(6)]
    single_Z = [('Z', i) for i in range(6)]
    single_Y = [('Y', i) for i in range(6)]
    multi_examples = [[('X', 0), ('X', 1), ('X', 2)]]

    return single_X, single_Z, single_Y, multi_examples

def leakage_noise_model(p=0.05):

    # Single-qubit leakage events
    single_leak = pauli_error([
        ('X', p / 2),
        ('Z', p / 2),
        ('I', 1 - p)
    ])

    # Two-qubit correlated leakage
    two_leak = pauli_error([
        ('XX', p / 2),
        ('ZZ', p / 2),
        ('II', 1 - p)
    ])

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(single_leak, ['h'])
    noise.add_all_qubit_quantum_error(two_leak, ['cx'])

    return noise

def simulation(shots=500, p_noise=0.02, do_reset=False):
    # prep reg
    qc, data, anc, c = prep_qubits()
    # encode logical 0
    encode_logical_zero(qc, data)
    dynamical_decoupling(qc, data)
    qc.barrier()

    # mid-circuit stab meas
    stab_measurements(qc, data, anc, c)


    dynamical_decoupling(qc, data)
    qc.barrier()

    if do_reset:
        qc.reset(data)
        encode_logical_zero(qc, data)

    # simulate
    simulator = AerSimulator(
        method='density_matrix',
        noise_model=leakage_noise_model(p=p_noise)
    )

    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    return qc, counts

def circuit_control(shots=500, p_noise=0.02):
    qc, data, anc, c = prep_qubits()
    encode_logical_zero(qc, data)

    dynamical_decoupling(qc, data)
    qc.barrier()

    for i in range(6):
        qc.cx(data[i], anc[0])
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])

    qc.measure(anc[0], c[0])
    qc.measure(anc[1], c[1])

    simulator = AerSimulator(
        method='density_matrix',
        noise_model=leakage_noise_model(p=p_noise)
    )

    result = simulator.run(qc, shots=shots).result()
    return result.get_counts()


def circuit_deferred_measurement(shots=500, p_noise=0.02):
    qc, data, anc, c = prep_qubits()

    encode_logical_zero(qc, data)

    dynamical_decoupling(qc, data)
    qc.barrier()
   

    for i in range(6):
        qc.cx(data[i], anc[0])
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])

    # measure at the end
    qc.measure(anc[0], c[0])
    qc.measure(anc[1], c[1])

    simulator = AerSimulator(
        method='density_matrix',
        noise_model=leakage_noise_model(p=p_noise)
    )

    result = simulator.run(qc, shots=shots).result()
    return result.get_counts()

def postselection_rate(counts):
    total = sum(counts.values())
    valid = counts.get('00', 0)
    return valid / total

def run_benchmarks(noise_vals, shots=2000):
    mid_rates = []
    control_rates = []
    deferred_rates = []

    for p in noise_vals:
        # mid-circuit
        _, counts_mid = simulation(shots=shots, p_noise=p, do_reset=False)
        mid_rates.append(postselection_rate(counts_mid))

        # no mid-circuit
        counts_control = circuit_control(shots=shots, p_noise=p)
        control_rates.append(postselection_rate(counts_control))

        # deferred
        counts_deferred = circuit_deferred_measurement(shots=shots, p_noise=p)
        deferred_rates.append(postselection_rate(counts_deferred))

    return mid_rates, control_rates, deferred_rates

if __name__ == "__main__":
    noise_vals = [0.0, 0.01, 0.02, 0.05]
    shots = 2000

    mid, control, deferred = run_benchmarks(noise_vals, shots)

    print("p_noise | mid-circuit | deferred | control")
    for i, p in enumerate(noise_vals):
        print(f"{p:6.3f} | {mid[i]:10.3f} | {deferred[i]:8.3f} | {control[i]:8.3f}")
