from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import  PassManager, InstructionProperties, generate_preset_pass_manager
from qiskit.transpiler.passes.scheduling import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.circuit.equivalence_library import (
    SessionEquivalenceLibrary as sel,
)
from qiskit.transpiler.passes import BasisTranslator
#===========================================================================================#

service = QiskitRuntimeService(
    token = not-today-robots, 
    channel = 'ibm_quantum_platform',
    instance='EE514'
)

def get_backend():
    backend = service.backend("ibm_fez")
    target=backend.target
    print("Using backend:", backend.name)
    basis_gates = list(target.operation_names)
    return backend, basis_gates

def prep_qubits():
    data = QuantumRegister(6, "data")
    anc  = QuantumRegister(2, "anc")
    c    = ClassicalRegister(2, "c") # to ensure mid-circuit measurement works
    qc = QuantumCircuit(data, anc, c)
    return qc, data, anc, c


def encode_logical_zero(qc, data):
    # GHZ preparation (equivalent to stabilizer projector P+)
    qc.h(data[0])
    for i in range(1, 6):
        qc.cx(data[0], data[i])
    return qc

def leakage_elimination_operation(qc, data):
    for q in data:
        qc.z(q)
    return qc

def stabilizer_measurements(qc, data, anc, c):
    # ZZZZZZ
    for i in range(6):
        qc.cx(data[i], anc[0])
    qc.measure(anc[0], c[0])
    qc.reset(anc[0])

    # XXXXXX
    for i in range(6):
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])
    qc.measure(anc[1], c[1])
    qc.reset(anc[1])

    return qc

def circuit_mid_circuit():
    qc, data, anc, c = prep_qubits()
    encode_logical_zero(qc, data)

    leakage_elimination_operation(qc, data)
    qc.barrier()
    stabilizer_measurements(qc, data, anc, c)
    leakage_elimination_operation(qc, data)

    return qc


def circuit_deferred():
    qc, data, anc, c = prep_qubits()
    encode_logical_zero(qc, data)

    for i in range(6):
        qc.cx(data[i], anc[0])
        qc.h(data[i])
        qc.cx(data[i], anc[1])
        qc.h(data[i])

    qc.measure(anc[0], c[0])
    qc.measure(anc[1], c[1])
    return qc


def circuit_control():
    qc, data, _, c = prep_qubits()
    encode_logical_zero(qc, data)

    qc.measure(data[0], c[0])
    qc.measure(data[1], c[1])
    return qc


def run_on_hardware(qc, backend, shots=2000):
    tqc = transpile(qc, backend, optimization_level=1)

    sampler = Sampler(backend)
    job = sampler.run([tqc], shots=shots)
    result = job.result()

    data = result[0].data
    if "c" in data:
        counts = data["c"].get_counts()
    else:
        counts = result[0].join_data().get_counts()

    return counts

def hardware_dd(name, qc, backend, basis_gates, shots=2000):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    tqc = pm.run(qc) 
    target = backend.target
    
    X = XGate()
    Y = YGate()
    dd_sequence = [X,Y,X,Y]

    if name == "mid-circuit":
        y_gate_properties = {}
        for qubit in range(target.num_qubits):
            y_gate_properties.update(
                {
                    (qubit,): InstructionProperties(
                        duration=target["x"][(qubit,)].duration,
                        error=target["x"][(qubit,)].error,
                    )
                }
            )
     
        target.add_instruction(YGate(), y_gate_properties)

    dd_pm = PassManager([ALAPScheduleAnalysis(target=target),PadDynamicalDecoupling(target=target, dd_sequence=dd_sequence),])

    qc_dd = dd_pm.run(tqc)
    qc_dd = BasisTranslator(sel, basis_gates)(qc_dd)

    sampler = Sampler(backend)
    job = sampler.run([qc_dd], shots=shots)
    result = job.result()

    data = result[0].data
    if "c" in data:
        counts = data["c"].get_counts()
    else:
        counts = result[0].join_data().get_counts()

    return counts
    

def postselection_rate(counts):
    total = sum(counts.values())
    return counts.get("00", 0) / total if total > 0 else 0

def run_hardware_benchmarks(backend, shots=2000):
    circuits = {
        "mid-circuit": circuit_mid_circuit(),
        "deferred": circuit_deferred(),
        "control": circuit_control(),
    }

    results = {}
    for name, qc in circuits.items():
        print(f"Running {name}...")
        counts = run_on_hardware(qc, backend, shots)
        results[name] = postselection_rate(counts)

    return results

def run_hardware_benchmarks_2(backend, basis_gates, shots=2000):
    circuits = {
        "mid-circuit": circuit_mid_circuit(),
        "deferred": circuit_deferred(),
        "control": circuit_control(),
    }

    results = {}
    for name, qc in circuits.items():
        print(f"Running {name}...")
        counts = hardware_dd(name, qc, backend, basis_gates, shots)
        results[name] = postselection_rate(counts)

    return results

if __name__ == "__main__":
    backend, basis_gates = get_backend()
    shots = 2000

    dd_results = run_hardware_benchmarks_2(backend, basis_gates, shots)
    results = run_hardware_benchmarks(backend, shots)

    print("\nPostselection probabilities:")
    print("------------------------------------")
    for k, v in results.items():
        print(f"{k:12s}: {v:.4f}")

    print("\nPostselection probabilities (with DD):")
    print("------------------------------------")
    for k, v in dd_results.items():
        print(f"{k:12s}: {v:.4f}")
