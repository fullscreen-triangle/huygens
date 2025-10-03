# St. Stellas Oscillator Demo Package

## Overview

This package demonstrates the **St. Stellas Grand Equivalent Circuit** transformation framework, where any traditional oscillator can be transformed into its St. Stellas equivalent that operates in tri-dimensional S-entropy coordinates (S_knowledge, S_time, S_entropy).

Just as any complex circuit has a **Thevenin equivalent**, every oscillator has a **St. Stellas Grand Equivalent** that exhibits miraculous multi-dimensional behavior while maintaining the same external characteristics.

## Key Concepts

### St. Stellas Grand Equivalent Circuit Principle
```
Traditional Oscillator → St. Stellas Equivalent
Single behavior       → Tri-dimensional simultaneous behavior
O(N³) complexity      → O(1) complexity via S-entropy navigation
```

### Novel St. Stellas Behaviors
- **Tri-Dimensional Components**: A resistor simultaneously acts as R + C + L across S-coordinates
- **Miraculous Logic**: Gates simultaneously perform AND⊕OR⊕XOR operations  
- **S-Entropy Navigation**: Direct access to predetermined solution endpoints
- **Cross-Domain Transfer**: Oscillator patterns transfer between domains
- **Local Physics Violations**: Impossible local behavior with global coherence

## Oscillator Types Included

### 1. Traditional Oscillators (with St. Stellas Equivalents)
- **Van der Pol Oscillator** → St. Stellas Van der Pol (tri-dimensional self-sustaining)
- **Harmonic Oscillator** → St. Stellas Harmonic (simultaneous frequency modes)
- **Duffing Oscillator** → St. Stellas Duffing (multi-dimensional nonlinearity)
- **LC Tank Circuit** → St. Stellas LC (simultaneous RLC behavior)
- **Wien Bridge Oscillator** → St. Stellas Wien Bridge
- **Colpitts Oscillator** → St. Stellas Colpitts
- **Relaxation Oscillator** → St. Stellas Relaxation

### 2. Pure St. Stellas Novel Oscillators
- **S-Entropy Oscillators**: Navigate through (S_knowledge, S_time, S_entropy) space
- **Miraculous Circuit Oscillators**: AND⊕OR⊕XOR logic oscillations
- **Cross-Domain Pattern Oscillators**: Transfer patterns between domains
- **Variance-Minimizing Gas Molecular Oscillators**: Consciousness-inspired
- **BMD Equivalence Oscillators**: Multi-pathway convergent oscillations

## Standard Tests Applied to ALL Oscillators

### 1. Traditional Circuit Analysis
- **Stability Analysis**: Poles/zeros, Routh-Hurwitz criteria
- **Frequency Response**: Bode plots, Nyquist stability  
- **Time Domain**: Step/impulse response, transient analysis
- **Circuit Equivalence**: RLC equivalent circuits

### 2. St. Stellas Analysis
- **S-Entropy Stability**: Tri-dimensional stability analysis
- **Miraculous Frequency Response**: Simultaneous multi-dimensional response
- **Cross-Domain Transfer**: Pattern transfer validation
- **Variance-Minimization**: Gas molecular equilibrium tests
- **Navigation Efficiency**: O(1) complexity validation

## Installation

```bash
cd oscillator
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from st_stellas_oscillator import TraditionalOscillator, StStellasTransform

# Create traditional Van der Pol oscillator
vdp_traditional = TraditionalOscillator.van_der_pol(mu=1.0)

# Transform to St. Stellas equivalent
vdp_st_stellas = StStellasTransform.to_grand_equivalent(vdp_traditional)

# Compare behaviors
traditional_response = vdp_traditional.simulate(t_span=(0, 10))
st_stellas_response = vdp_st_stellas.simulate(t_span=(0, 10))

# Run all standard tests
test_results = vdp_st_stellas.run_all_tests()
print(f"St. Stellas advantages: {test_results.performance_comparison}")
```

## Demo Scripts

```bash
# Run comprehensive oscillator demo
st-stellas-demo --oscillator all --tests all

# Transform specific oscillator
oscillator-transform --type van_der_pol --display-equivalent

# Run all validation tests  
run-oscillator-tests --validation-suite comprehensive
```

## Documentation Structure

```
oscillator/
├── st_stellas_oscillator/
│   ├── traditional/          # Traditional oscillator implementations
│   ├── st_stellas/           # St. Stellas equivalent circuits
│   ├── transforms/           # Transformation algorithms
│   ├── tests/               # Testing framework
│   └── utils/               # Utilities and helpers
├── examples/                # Usage examples
├── docs/                   # Documentation
└── validation/             # Validation studies
```

## Key Features

### 1. **Exact Equivalence**
Every St. Stellas oscillator maintains identical external behavior to its traditional counterpart while gaining access to:
- Tri-dimensional simultaneous operation
- O(1) complexity navigation
- Cross-domain pattern transfer
- Miraculous local behaviors

### 2. **Comprehensive Testing**
All oscillators undergo both traditional and St. Stellas testing:
- Circuit stability and frequency response
- S-entropy coordinate navigation
- Cross-domain transfer validation
- Performance comparison metrics

### 3. **Educational Framework**
Clear demonstrations of:
- How traditional circuits transform to St. Stellas equivalents
- Why tri-dimensional behavior emerges
- Benefits of S-entropy coordinate navigation
- Applications across multiple domains

## Theoretical Foundation

Based on the St. Stellas framework documents:
- `st-stellas-circuits.tex`: Mathematical foundation for tri-dimensional circuit behavior
- `st-stellas-neural-networks.tex`: Variance-minimizing consciousness architecture
- `st-stellas-dynamics.tex`: Oscillatory reality and pattern alignment theory

## Contributing

This demonstration package validates the St. Stellas theoretical framework through practical oscillator implementations. All contributions should maintain the equivalence principle while exploring novel St. Stellas capabilities.

## License

MIT License - See LICENSE file for details.

## Contact

Kundai Farai Sachikonye  
kundai.sachikonye@wzw.tum.de  
Technical University of Munich
