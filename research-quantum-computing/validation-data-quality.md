# Data Quality Assessment: Quantum Computing Research

## Source Quality Assessment

### Tier 1: High Reliability (Primary/Institutional Sources)
- IBM Newsroom announcements -- official corporate disclosures, verifiable
- Google Research Blog -- peer-reviewed results backing claims
- Quantinuum press releases -- backed by published benchmarks
- Nature Biotechnology (St. Jude research) -- peer-reviewed
- Riverlane QEC Survey -- methodology described, 300+ respondents
- McKinsey/Forrester analyses -- professional research firms

### Tier 2: Medium Reliability (Industry Analysis)
- The Quantum Insider -- reputable industry publication but relies on company announcements
- IEEE Spectrum -- reliable technology journalism
- SpinQ -- industry participant with potential bias
- HPCwire -- technology journalism, generally reliable

### Tier 3: Lower Reliability (Aggregation/Secondary)
- Medium articles -- individual author opinions, not peer-reviewed
- BQPSim blog -- commercial entity
- Quantum AI (quantumai.co.com) -- aggregation site
- Supaboard blog -- not a primary quantum source

## Data Consistency Check

### Consistent Across Sources
- IBM's Nighthawk specs (120 qubits, 5,000 gates) -- confirmed across multiple sources
- Google Willow specs (105 qubits, fidelity numbers) -- consistent
- Quantinuum Helios specs (98 qubits, fidelity numbers) -- consistent
- Investment figures (Quantinuum $10B, PsiQuantum $7B) -- consistent
- QEC paper count growth (36 in 2024 to 120 in 2025) -- single source (Riverlane)

### Potential Inconsistencies Identified
1. **Market size:** USD 1.8B vs. USD 3.5B -- likely different market definitions
2. **IBM Kookaburra timeline:** Some sources say 2025, others 2026 -- appears to have shifted
3. **IonQ qubit counts:** "Up to 100" physical qubits in one source vs specific architecture details in another -- need to distinguish between deployed and demonstrated
4. **Quantum advantage claims:** Multiple vendors claim some form of "advantage" but with different definitions and benchmarks

### Data Gaps
1. **Rigetti** is mentioned in passing but no detailed current hardware specs
2. **Microsoft's topological approach** is noted as "unproven" but no specifics on Majorana qubit progress in 2025
3. **Chinese quantum computing efforts** (e.g., Origin Quantum, Zuchongzhi processor) are entirely absent
4. **European efforts** (IQM, Pasqal details) are underrepresented
5. **Cost data** -- no information on what it costs to access or run these quantum computers
6. **Actual performance benchmarks** comparing different systems head-to-head are missing

## Numerical Verification

### Verified
- Google Willow: 105 qubits, 99.97%/99.88%/99.5% fidelities -- matches Wikipedia and Google's own blog
- Quantinuum Helios: 98 qubits, 99.9975%/99.921% fidelities -- matches press release
- IBM Nighthawk: 120 qubits -- matches IBM newsroom
- Riverlane survey: 300+ respondents, 95% rate QEC essential -- matches original report

### Unverified/Requires Caution
- Roche Alzheimer's claim (18 months vs 4-6 years) -- single secondary source
- Goldman Sachs "25x faster" -- cited from a statistics aggregation site
- D-Wave "7 seconds vs years" -- anecdotal, single source
- IonQ "quantum advantage in drug discovery" -- company claim, awaiting independent validation

## Recommendations
1. Flag all Tier 3 source claims with [unverified] markers or remove
2. Add Chinese and European quantum efforts for completeness
3. Include cost/access information
4. Clarify market size definition
5. Add head-to-head benchmark comparisons where available
6. Note Microsoft's Majorana qubit announcement from February 2025
