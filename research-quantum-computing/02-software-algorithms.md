# Quantum Computing Software & Algorithms Landscape (2025-2026)

## Major Frameworks

### IBM Qiskit
- The world's most popular quantum SDK, built in Python with modular libraries: Terra (circuit building/compiling), Aer (simulation), Aqua (application-level algorithms).
- World's fastest transpiler -- 83x faster than the next leading SDK.
- **November 2025 advances:** 24% accuracy increase with dynamic circuits; 100x cost reduction for error mitigation with HPC-powered techniques.
- New C++ interface powered by C-API for native HPC integration.

### Google Cirq
- Python library for writing, manipulating, and optimizing quantum circuits.
- Focus on fine-grained, low-level control -- custom gate definitions, calibration routines, noise modeling.
- Well-suited for research on Google's quantum processors.

### Other Notable Frameworks
- **PennyLane (Xanadu):** Differentiable programming for quantum machine learning.
- **Amazon Braket:** Cloud-based access to multiple quantum hardware providers.
- **Q# (Microsoft):** Standalone quantum language integrated with the Quantum Development Kit.
- **PyQuil (Rigetti):** Python-based framework for Rigetti hardware.
- **Ocean (D-Wave):** SDK for quantum annealing applications.
- **Strawberry Fields (Xanadu):** Photonic quantum computing library.

## Key Quantum Algorithms

### Foundational Algorithms
- **Shor's Algorithm:** Integer factorization -- exponential speedup over classical methods.
- **Grover's Algorithm:** Unstructured search -- quadratic speedup.
- **Deutsch-Jozsa Algorithm:** Early proof of quantum computational advantage.

### Application Areas
- **Combinatorial Optimization:** QAOA, VQE for solving NP-hard problems.
- **Quantum Chemistry Simulation:** Simulating molecular structures and reactions.
- **Quantum Machine Learning:** Quantum kernels, variational classifiers.
- **Nonlinear Dynamics:** Quantum variational methods for complex system simulation.

## Path to Quantum Advantage
- IBM anticipates verified quantum advantage confirmed by wider community by end of 2026.
- Nighthawk iterations expected to deliver up to 7,500 gates by end of 2026 and 10,000 gates in 2027.

## Challenges
- SDKs require specialized quantum expertise -- steep learning curves for classical programmers.
- No widely adopted standard quantum programming language or consensus on abstractions.
- Fragmented workflows and limited portability across heterogeneous platforms.
- Debugging and testing quantum programs remains fundamentally difficult.

## Sources

- [IBM Qiskit](https://www.ibm.com/quantum/qiskit)
- [Google Cirq](https://quantumai.google/cirq)
- [BQPSim Quantum Software Platforms Guide](https://www.bqpsim.com/blogs/quantum-software-platforms)
- [IBM Newsroom Nov 2025](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance)
- [Open Source Frameworks Overview](https://www.opensourceforu.com/2025/11/the-top-open-source-quantum-computing-frameworks/)
