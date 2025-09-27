# Universal Oscillatory Transformation Framework

**Transform ANY data into differential equations, analyze with tridimensional S-entropy, and discover patterns automatically.**

## 🚀 The Revolutionary Framework

This framework solves your exact problem: **analyzing GB of data without needing to know the specific oscillatory system**. Just point it at any data source, specify the time column, and it does the rest.

### ✨ Key Features

- **Universal Data Ingestion**: Works with JSON, CSV, pandas DataFrames, numpy arrays, or dictionaries
- **Automatic Differential Equation Generation**: Transforms any time series into first-order differential equations
- **Laplace Domain Transformation**: Converts to frequency domain for advanced analysis
- **Tridimensional S-Entropy**: Analyzes data in three differential forms (dt, dinfo, dentropy)
- **Interchangeable Forms**: Seamlessly switches between differential forms during analysis
- **Pattern Discovery**: Finds oscillatory patterns without prior system knowledge
- **Cross-Signal Analysis**: Discovers coupling between multiple signals
- **Comprehensive Validation**: Statistical significance testing and confidence scoring

## 🎯 Simple Usage - Exactly What You Asked For

```python
from src.universal_transformation_framework import analyze_oscillations

# That's it! One line for any data source:
results = analyze_oscillations('your_data.json', time_column='timestamp')
```

### Real Examples with Your Data

```python
# Analyze your actigraphy data
results = analyze_oscillations('../experimental-data/actigraphy/actigraphy.json', 
                              time_column='timestamp')

# Analyze sleep data
results = analyze_oscillations('../experimental-data/sleep/sleepRecords.json',
                              time_column='timestamp') 

# Analyze any CSV file
results = analyze_oscillations('data.csv', time_column='time')

# Analyze specific columns only
results = analyze_oscillations('data.json', 
                              time_column='timestamp',
                              target_columns=['heart_rate', 'activity'])
```

## 🔬 What It Does Automatically

### 1. **Data Transformation Pipeline**
- Ingests any data format
- Auto-detects time column if not specified
- Generates first-order differential equations: `dy/dt = f(y, t)`
- Transforms to Laplace domain: `ℒ{y(t)} = Y(s)`

### 2. **Tridimensional S-Entropy Analysis**
Computes entropy in three interchangeable differential forms:
- **dx/dt** (time to solution)
- **dx/dinfo** (information differential)  
- **dx/dentropy** (entropy differential)

Creates 3×3 entropy matrix:
```
[time_entropy    time_complexity    temporal_regularity  ]
[mutual_info     info_gain         info_complexity      ]
[meta_entropy    entropy_complexity entropy_derivative   ]
```

### 3. **Pattern Discovery Engine**
Automatically discovers:
- **Periodic patterns** (clear oscillations)
- **Quasi-periodic patterns** (multiple frequencies)
- **Chaotic patterns** (deterministic chaos)
- **Fractal patterns** (self-similar structures)
- **Hybrid patterns** (combined behaviors)

### 4. **Universal Analysis**
Works on **any oscillatory system**:
- Biological rhythms (heart rate, sleep, circadian)
- Physical systems (mechanical vibrations, circuits)
- Economic data (market oscillations, cycles)
- Environmental data (temperature, weather patterns)
- Signal processing (any time series)

## 📊 Comprehensive Results

```python
results = analyze_oscillations('data.json', 'timestamp')

# Individual signal analysis
for signal in results['individual_signals']:
    print(f"Signal: {signal['signal_name']}")
    print(f"Confidence: {signal['transformation'].confidence_score:.3f}")
    print(f"DE Type: {signal['transformation'].differential_equation['type']}")
    print(f"Patterns: {len(signal['transformation'].pattern_discovery['validated_patterns'])}")

# System-level insights
meta = results['meta_analysis']
print(f"System Type: {meta['system_classification']['type']}")
print(f"Complexity: {meta['system_classification']['complexity']}")
print(f"Overall Confidence: {meta['overall_confidence']:.3f}")

# Cross-signal coupling (if multiple signals)
if 'cross_signal_analysis' in results:
    coupling = results['cross_signal_analysis']['strongest_coupling']
    print(f"Strongest Coupling: {coupling['signal_pair']} ({coupling['correlation']:.3f})")
```

## 🎨 Automatic Visualizations

The framework generates comprehensive visualizations automatically:

- **Summary Plot**: Overview of all signals and classifications
- **Individual Signal Plots**: Detailed analysis for each signal
- **Cross-Analysis Plots**: Signal coupling and relationships
- **S-Entropy Matrices**: Tridimensional entropy visualization
- **Pattern Confidence**: Discovery results and confidence scores

## 🛠️ Advanced Configuration

```python
config = {
    'differential_forms': [DifferentialForm.TIME, DifferentialForm.INFO, DifferentialForm.ENTROPY],
    'pattern_discovery': True,
    'cross_validation': True,
    'generate_visualizations': True,
    'output_directory': 'my_analysis_results',
    'confidence_threshold': 0.5,
    'max_data_points': 50000,  # Memory management
    'auto_downsample': True
}

results = analyze_oscillations('data.json', 'timestamp', config=config)
```

## 📈 Command Line Interface

For quick analysis of your data files:

```bash
python demo/simple_analysis_demo.py --data ../experimental-data/actigraphy/actigraphy.json --time timestamp
python demo/simple_analysis_demo.py --data ../experimental-data/sleep/sleepRecords.json --time timestamp  
python demo/simple_analysis_demo.py --data your_data.csv --time time_column --columns signal1 signal2
```

## 🔍 Understanding Results

### Confidence Scores (0-1 scale)
- **0.8-1.0**: Strong patterns, reliable analysis
- **0.6-0.8**: Moderate patterns, good confidence  
- **0.4-0.6**: Weak patterns, consider more data
- **0.0-0.4**: Poor patterns, improve data quality

### System Classifications
- **periodic_system**: Clear oscillatory behavior
- **chaotic_system**: Deterministic chaos detected
- **fractal_system**: Self-similar patterns
- **hybrid_system**: Multiple pattern types
- **coupled_system**: Strong inter-signal coupling

### Pattern Types
- **PERIODIC**: Regular oscillations
- **QUASI_PERIODIC**: Multiple incommensurate frequencies
- **CHAOTIC**: Deterministic but unpredictable
- **FRACTAL**: Self-similar at multiple scales
- **HYBRID**: Combination of pattern types
- **NOISE**: Primarily stochastic

## 🚀 Quick Start with Your Data

1. **Install dependencies**:
```bash
pip install numpy pandas scipy scikit-learn matplotlib sympy
```

2. **Run analysis on your data**:
```python
# Replace with your actual data file
results = analyze_oscillations('../experimental-data/actigraphy/actigraphy.json', 
                              time_column='timestamp')

print(results['summary_report'])  # Full analysis report
```

3. **Check results**:
- Visualizations saved to `transformation_results/`
- Full report in `results['summary_report']`
- Detailed data in `results` dictionary

## 🎯 Perfect for Your GB of Data

This framework is designed exactly for your use case:

- **Handles large datasets** (auto-downsampling)
- **Works with any data format** (JSON, CSV, etc.)
- **No system knowledge required** (universal analysis)
- **Discovers patterns automatically** (no manual tuning)
- **Provides actionable insights** (recommendations, confidence scores)
- **Scalable processing** (memory-efficient algorithms)

## 📊 Example Output

```
=== UNIVERSAL OSCILLATORY TRANSFORMATION FRAMEWORK ===
COMPREHENSIVE ANALYSIS REPORT

EXECUTIVE SUMMARY:
- Data processed: 3 signal(s), 48,000 data points
- Analysis duration: 2.34 seconds
- Overall system confidence: 0.847
- System classification: oscillatory_system_coupled
- System complexity: high

DIFFERENTIAL EQUATION ANALYSIS:
- Dominant equation type: oscillatory
- heart_rate: oscillatory (R² = 0.876)
- activity: nonlinear (R² = 0.743)
- temperature: linear (R² = 0.612)

PATTERN DISCOVERY:
- Dominant pattern type: periodic
- Total patterns discovered: 7
  
  heart_rate patterns:
    - periodic: confidence 0.891
    - quasi_periodic: confidence 0.743
    
RECOMMENDATIONS:
1. Detailed harmonic analysis
2. Phase-amplitude coupling analysis
3. Proceed with predictive modeling
```

This framework transforms **any data** into actionable oscillatory insights with **zero system-specific configuration required**!
