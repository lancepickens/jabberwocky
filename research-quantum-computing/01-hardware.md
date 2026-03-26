# Quantum Computing Hardware: 2025-2026 Overview

## IBM

At its annual Quantum Developer Conference in November 2025, IBM unveiled fundamental progress on its path to delivering both quantum advantage by the end of 2026 and fault-tolerant quantum computing by 2029.

### Current Hardware
- **IBM Nighthawk** (expected end of 2025): 120 qubits linked with 218 next-generation tunable couplers in a square lattice -- over 20% more couplers compared to IBM Quantum Heron.
- **Third-revision Heron**: Lowest median two-qubit gate errors to date -- of its 176 possible two-qubit couplings, 57 deliver less than one error in every 1,000 operations (99.9% fidelity).
- Nighthawk enables problems requiring up to 5,000 two-qubit gates, with IBM expecting future iterations to deliver up to 7,500 gates by end of 2026 and up to 10,000 gates in 2027.

### Roadmap
- **Kookaburra (2026)**: First quantum processor module capable of storing information in a qLDPC memory and processing it with an attached LPU.
- **Starling (2029)**: Will scale to run 100 million gates on 200 logical qubits.
- IBM has proven real-time error decoding in less than 480 nanoseconds using qLDPC codes.

## Google

### Current Hardware (Willow Processor)
- 105-qubit superconducting quantum computing processor (announced December 2024).
- Willow can reduce errors exponentially as qubits scale, achieving below-threshold quantum error correction.
- Fidelities: 99.97% single-qubit gates, 99.88% entangling gates, 99.5% readout.
- October 2025: First verifiable quantum advantage with Quantum Echoes -- approximately 13,000x faster than fastest supercomputers.

### Limitations & Outlook
- Logical error rates (~0.14% per cycle) remain orders of magnitude above 10^-6 levels needed for large-scale algorithms.
- Google targeting ~100+ logical qubits by ~2028, thousands of logical qubits in early 2030s.

## Quantinuum

### Current Hardware (Helios)
- 98 fully connected physical trapped-ion qubits, fidelity >99.9%.
- Squeezed 48 error-corrected logical qubits from just 98 physical qubits -- near 2:1 ratio.
- Logical qubits are 10x-100x more reliable than the physical qubits they are made of ("beyond break-even").
- Achieved quantum volume of 8,388,608 (2^23) in 2025 -- highest recorded.

### Roadmap
- Physical Helios installation expected in Singapore by 2026.
- **Sol architecture (2027)**: 2D-grid-based qubit design, 192 physical / ~96 logical qubits.
- **Apollo system (2030)**: Universal, fault-tolerant quantum computing.
- Reached $10 billion valuation in 2025.

## IonQ

### Current Hardware & Plans
- 2025 roadmap: development systems supporting 100 physical qubits for Tempo platform.
- 99.9% one-qubit gate fidelity.
- Projecting ~20,000 physical qubits by 2028, ~2,000,000 by 2030 (leveraging Oxford Ionics chip-integrated traps and Lightsynq photonic interconnects).
- ~1,600 error-corrected logical qubits in 2028; 40,000-80,000 logical qubits by 2030.
- Enterprise-grade error rates (~10^-7) by 2028; below 10^-12 by 2030.

## Industry-Wide Trends
- Focus has shifted from qubit counts to high-fidelity operations and first logical qubits.
- Superconducting qubits (IBM, Google, Rigetti) lead in raw qubit counts and gate speeds; trapped ions (IonQ, Quantinuum) excel in fidelity and coherence.
- 120 new peer-reviewed QEC papers in first 10 months of 2025 (up from 36 in 2024).
- Oxford Ionics achieved 99.99% fidelity for two-qubit gates in 2025.
- Early systems in 2026 will struggle to offer enough logical qubits at meaningful error rates, but dramatic increases in physical qubits will breathe new life into NISQ-era research.

## Sources
- [Riverlane QEC 2025 Trends and 2026 Predictions](https://www.riverlane.com/blog/quantum-error-correction-our-2025-trends-and-2026-predictions)
- [IEEE Spectrum: Neutral Atom Quantum Computing](https://spectrum.ieee.org/neutral-atom-quantum-computing)
- [IBM Newsroom Nov 2025](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance)
- [Google Willow Announcement](https://blog.google/innovation-and-ai/technology/research/google-willow-quantum-chip/)
- [Quantinuum Commercial Roadmap Milestone](https://www.nextplatform.com/2025/11/10/quantinuum-makes-another-milestone-on-commercial-quantum-roadmap/)
- [IonQ Roadmap](https://www.ionq.com/roadmap)
- [SpinQ Industry Trends 2025](https://www.spinquanta.com/news-detail/quantum-computing-industry-trends-2025-breakthrough-milestones-commercial-transition)
