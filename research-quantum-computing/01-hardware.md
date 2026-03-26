# Quantum Computing Hardware: 2025-2026 Overview

## Industry-Wide Trends

The quantum computing industry crossed an inflection point in 2025, with vendors moving beyond theoretical fault-tolerant architectures into early engineering reality. The measure of progress is shifting from raw qubit counts to error-corrected logical qubits.

Financially, 2025 was characterized by massive investment: Quantinuum valued at $10B, PsiQuantum at $7B, and SandboxAQ at $5.75B.

## IBM

- **Nighthawk Processor (2025):** 120 qubits with 218 next-generation tunable couplers, supporting up to 5,000 two-qubit gates -- 30% more circuit complexity than the previous Heron processor.
- **Kookaburra (2026):** 1,386-qubit multi-chip processor designed to link three chips via chip-to-chip couplers, forming a combined 4,158-qubit system.
- **Quantum Loon:** Experimental processor demonstrating all key components for fault-tolerant quantum computing, with real-time error decoding in under 480 nanoseconds using qLDPC codes.
- **Starling (2028-2029):** ~200 logical qubits comprising ~10,000 physical qubits, capable of running 100 million gates.

## Google

- **Willow Chip:** 105-qubit superconducting processor with fidelities of 99.97% (single-qubit gates), 99.88% (entangling gates), and 99.5% (readout).
- **Quantum Advantage (Oct 2025):** First verifiable quantum advantage with Quantum Echoes algorithm -- approximately 13,000x faster than fastest supercomputers.
- **Caveat:** Logical error rates (~0.14% per cycle) remain orders of magnitude above the 10^-6 levels needed for large-scale algorithms.
- **Roadmap:** Targeting ~100+ logical qubits by ~2028, thousands in early 2030s, useful error-corrected quantum computer by ~2029.

## Quantinuum

- **Helios (2025):** 98 fully connected physical qubits. Single-qubit gate fidelity of 99.9975% and two-qubit gate fidelity of 99.921% -- the most accurate commercial quantum computer in the world.
- **Logical Qubits:** 94 error-detected logical qubits globally entangled; 48 fully error-corrected logical qubits at a 2:1 encoding rate.
- **Error Correction:** Logical qubits achieving 22x lower failure rates. Physical error rates of 0.024 reduced to logical error rates of 0.0011.
- **Roadmap:** Sol (2027) with 192 physical qubits; Apollo (2029) with thousands of physical qubits targeting full fault tolerance.

## IonQ

- **Tempo Architecture (2025):** Up to 100 physical qubits, with 256-qubit system planned for 2026.
- **Fidelity:** 99.9% one-qubit gate fidelity.
- **Accelerated Roadmap:** 1,600 logical qubits (2028), 8,000 (2029), 80,000 (2030) -- targeting 2 million physical qubits and 80,000 logical qubits by 2030.
- **Technology Shift:** Moved from ytterbium to barium atoms with microwave control from Oxford Ionics and Lightsync's interconnect architecture.

## Technology Comparison

- **Superconducting (IBM, Google, Rigetti):** Lead in raw qubit counts and gate speeds.
- **Trapped Ions (IonQ, Quantinuum):** Excel in fidelity and coherence times.
- **Topological (Microsoft):** Promises inherent error protection but remains unproven at scale.
- **Quantum Annealing (D-Wave):** Delivers value today but for limited problem sets.

## Other Notable Developments

- Fujitsu/RIKEN: 256-qubit superconducting computer; 1,000-qubit machine planned by 2026.
- Oxford Ionics: 99.99% fidelity for two-qubit gates (2025).
- QuEra: Error-correction-ready machine delivered to Japan's AIST; global availability planned for 2026.

## Sources

- [Riverlane QEC Predictions](https://www.riverlane.com/blog/quantum-error-correction-our-2025-trends-and-2026-predictions)
- [SpinQ Industry Trends 2025](https://www.spinquanta.com/news-detail/quantum-computing-industry-trends-2025-breakthrough-milestones-commercial-transition)
- [IEEE Spectrum - Neutral Atom Computing](https://spectrum.ieee.org/neutral-atom-quantum-computing)
- [IBM Quantum Newsroom](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance)
- [Quantinuum Helios](https://www.quantinuum.com/blog/introducing-helios-the-most-accurate-quantum-computer-in-the-world)
- [Google Willow](https://blog.google/innovation-and-ai/technology/research/google-willow-quantum-chip/)
- [Google Quantum Echoes](https://blog.google/innovation-and-ai/technology/research/quantum-echoes-willow-verifiable-quantum-advantage/)
- [IonQ Roadmap](https://www.ionq.com/roadmap)
