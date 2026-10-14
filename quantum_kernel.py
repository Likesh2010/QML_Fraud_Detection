from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel


# ============================================================
# QML ANOMALY DETECTION
# QUANTUM FEATURE MAP + QUANTUM KERNEL
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N_QUBITS = 4
REPS = 2
ENTANGLEMENT = "linear"


# ------------------------------------------------------------
# 1. Create ZZ Quantum Feature Map
# ------------------------------------------------------------

print("=" * 70)
print("CREATING QUANTUM FEATURE MAP")
print("=" * 70)

feature_map = zz_feature_map(
    feature_dimension=N_QUBITS,
    reps=REPS,
    entanglement=ENTANGLEMENT
)


print("\nFeature map created successfully.")

print(
    f"Number of qubits       : {N_QUBITS}"
)

print(
    f"Number of repetitions  : {REPS}"
)

print(
    f"Entanglement           : {ENTANGLEMENT}"
)

print(
    f"Circuit depth          : {feature_map.depth()}"
)

print(
    f"Number of parameters   : "
    f"{feature_map.num_parameters}"
)


# ------------------------------------------------------------
# 2. Display circuit
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("QUANTUM FEATURE MAP CIRCUIT")
print("=" * 70)

print(
    feature_map.decompose().draw(
        output="text"
    )
)


# ------------------------------------------------------------
# 3. Create Fidelity Quantum Kernel
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING FIDELITY QUANTUM KERNEL")
print("=" * 70)

quantum_kernel = FidelityQuantumKernel(
    feature_map=feature_map
)

print(
    "FidelityQuantumKernel created successfully."
)


# ------------------------------------------------------------
# 4. Test the kernel with two sample vectors
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TESTING QUANTUM KERNEL")
print("=" * 70)

import numpy as np


x1 = np.array([
    [0.1, 0.2, 0.3, 0.4]
])

x2 = np.array([
    [0.5, 0.6, 0.7, 0.8]
])


kernel_value = quantum_kernel.evaluate(
    x_vec=x1,
    y_vec=x2
)

print(
    "\nKernel value:"
)

print(kernel_value)

print(
    "\nKernel test completed successfully."
)