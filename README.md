# Cardiovascular Oscillatory Analysis Framework

A comprehensive computational framework for multi-sensor cardiovascular data analysis, integrating consumer-grade heart rate sensors with professional cardiovascular testing through advanced signal processing and Kalman filtering techniques.

## Overview

This framework provides tools for analyzing cardiovascular data from multiple consumer sensors simultaneously, with capabilities for:
- Multi-sensor heart rate data fusion using Kalman filtering
- QRS complex detection and analysis from consumer ECG devices  
- Heart rate variability analysis with entropy measures
- Personalization through calibration with professional cardiovascular testing
- Precision assessment and improvement recommendations

The system addresses the challenge of obtaining professional-grade cardiovascular measurements from consumer devices by implementing statistical fusion algorithms and personalized calibration techniques.

## Framework Components

### Core Analysis Modules

1. **Cardiovascular Oscillatory Suite** (`src/cardiovascular_oscillatory_suite.py`)
   - Multi-sensor Kalman filtering for heart rate fusion
   - QRS complex detection using Pan-Tompkins algorithm
   - Heart rate variability analysis with time and frequency domain metrics
   - Personalization system using professional cardiovascular data

2. **Analysis Scripts** (`analyze_cardiovascular_data.py`)
   - Command-line interface for comprehensive cardiovascular analysis
   - Automated report generation and visualization
   - Configurable sensor input and sampling rates

3. **Supporting Frameworks** 
   - Entropy-oscillation coupling analysis
   - Universal transformation framework for oscillatory systems
   - Biomechanical oscillatory system integration

### Data Processing Pipeline

The framework implements a sequential processing pipeline:

1. **Multi-sensor data ingestion** from various consumer devices
2. **Kalman filter fusion** with adaptive noise estimation  
3. **QRS complex detection** where ECG signals are available
4. **Heart rate variability computation** using established metrics
5. **Professional data calibration** when reference measurements exist
6. **Precision assessment** with uncertainty quantification
7. **Report generation** with recommendations for improvement

## Installation and Setup

### Prerequisites

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### Quick Start

Run the demonstration with sample data:

```bash
python analyze_cardiovascular_data.py --demo
```

Analyze your multi-sensor heart rate data:

```bash
python analyze_cardiovascular_data.py \
  --heart-rate-data sample_heart_rate_sensors_config.json \
  --professional-data sample_professional_cardiovascular_data.json
```

## Data Configuration

### Sensor Configuration Format

Create a JSON file specifying your sensor data sources:

```json
{
  "chest_strap": {
    "file_path": "data/polar_h10_data.json",
    "column_mapping": {
      "hr_bpm": "heart_rate",
      "timestamp": "timestamp"
    },
    "sensor_info": {
      "model": "Polar H10",
      "measurement_type": "ECG",
      "expected_accuracy": "±1 bpm"
    }
  },
  "watch_ppg": {
    "file_path": "data/apple_watch_data.json", 
    "column_mapping": {
      "heart_rate": "heart_rate",
      "time": "timestamp"
    },
    "sensor_info": {
      "model": "Apple Watch Series 8",
      "measurement_type": "PPG",
      "expected_accuracy": "±3 bpm"
    }
  }
}
```

### Heart Rate Data Format

Individual sensor data files should contain:

```json
[
  {"timestamp": "2024-12-27T10:00:00", "heart_rate": 72, "signal_quality": 0.95},
  {"timestamp": "2024-12-27T10:00:01", "heart_rate": 73, "signal_quality": 0.92},
  {"timestamp": "2024-12-27T10:00:02", "heart_rate": 71, "signal_quality": 0.98}
]
```

### Professional Cardiovascular Data Format

Reference data from professional testing:

```json
{
  "ecg_parameters": {
    "heart_rate": 72.0,
    "hrv_rmssd": 35.2,
    "hrv_sdnn": 45.8,
    "qrs_duration": 0.098,
    "pr_interval": 0.156,
    "qt_interval": 0.384
  },
  "echocardiography_results": {
    "ejection_fraction": 62.0,
    "stroke_volume": 78.0,
    "cardiac_output": 5.8
  },
  "stress_test_data": {
    "max_heart_rate_achieved": 185.0,
    "anaerobic_threshold": 156.0,
    "heart_rate_zones": [72, 95, 125, 155, 180]
  },
  "resting_heart_rate": 70.0,
  "max_heart_rate": 185.0,
  "anaerobic_threshold": 156.0,
  "cardiac_output": 5.8,
  "ejection_fraction": 62.0
}
```

## Analysis Methods

### Multi-Sensor Kalman Filtering

The framework implements an adaptive Kalman filter for optimal sensor fusion:

- **State vector**: `[heart_rate, heart_rate_derivative]`
- **Process model**: Constant velocity with cardiac variability noise
- **Measurement model**: Weighted combination based on sensor quality ratings
- **Adaptive estimation**: Real-time measurement noise parameter updates

Sensor quality weights are assigned based on device type:
- Professional ECG: 1.0
- Chest strap ECG: 0.9  
- Smart ring PPG: 0.8
- Watch PPG: 0.7
- Fitness tracker: 0.6
- Smartphone camera: 0.4

### QRS Complex Detection

QRS detection utilizes the Pan-Tompkins algorithm:

1. **Bandpass filtering** (5-15 Hz) for QRS frequency isolation
2. **Derivative filtering** for slope enhancement
3. **Signal squaring** for positive amplitude conversion
4. **Moving window integration** (150ms window)
5. **Adaptive thresholding** with 300ms refractory period

### Heart Rate Variability Analysis

Time domain metrics:
- **RMSSD**: Root mean square of successive RR interval differences
- **SDNN**: Standard deviation of RR intervals
- **Triangular Index**: RR interval distribution geometry

Frequency domain metrics:
- **LF power**: Low frequency band (0.04-0.15 Hz)
- **HF power**: High frequency band (0.15-0.4 Hz) 
- **LF/HF ratio**: Autonomic balance indicator

Entropy measures:
- **Sample entropy**: Signal regularity quantification
- **Approximate entropy**: Pattern predictability measure
- **Shannon entropy**: Information content analysis

### Personalization and Calibration

The system creates individual calibration curves by comparing consumer measurements with professional references:

- **Linear calibration**: `professional = slope * consumer + offset`
- **Quality assessment**: Calibration confidence scoring
- **Individual signatures**: Personalized cardiovascular parameter ranges
- **Training zones**: Heart rate zones based on professional testing

## Output and Results

### Analysis Report

The framework generates comprehensive reports including:

- **Multi-sensor fusion results** with uncertainty quantification
- **QRS analysis outcomes** with signal quality assessment
- **Heart rate variability metrics** with autonomic assessment
- **Personalization results** with calibration quality scores
- **Precision assessment** with improvement recommendations

### Visualizations

Automatically generated plots include:
- Multi-sensor heart rate fusion with uncertainty bands
- Sensor contribution pie charts
- Kalman filter performance metrics
- Heart rate variability frequency analysis
- Precision assessment summaries

### Performance Metrics

Expected precision levels:
- **Kalman filter uncertainty**: ±1-3 bpm (target: <±2 bpm)
- **QRS detection accuracy**: >95% for good quality signals
- **HRV correlation**: r > 0.9 with professional ECG reference
- **Overall precision**: Graded as excellent/good/fair/poor

## Data Requirements

### Minimum Requirements
- Heart rate measurements from at least one sensor
- Minimum 100 data points for basic analysis
- Temporal sampling ≤60 seconds for rhythm detection

### Recommended Requirements
- Multiple sensor types for optimal fusion
- ≥1000 measurements spanning multiple hours
- Sampling interval ≤5 seconds for detailed analysis
- Professional reference data for calibration

### Optional Enhancements
- ECG waveform data for QRS analysis
- Environmental data for context
- Activity labels for state-dependent analysis

## Command Line Interface

### Basic Analysis
```bash
python analyze_cardiovascular_data.py --demo
```

### Custom Configuration
```bash
python analyze_cardiovascular_data.py \
  --heart-rate-data sensors.json \
  --professional-data professional.json \
  --sampling-rate 100.0 \
  --output results_directory
```

### Parameters
- `--heart-rate-data`: Path to sensor configuration file
- `--professional-data`: Path to professional cardiovascular data
- `--demo`: Run with synthetic demonstration data
- `--sampling-rate`: Analysis sampling rate in Hz (default: 100.0)
- `--output`: Output directory for results (default: cardiovascular_results)

## Algorithmic Details

### Kalman Filter Implementation

State transition matrix:
```
F = [[1, dt],
     [0, 1 ]]
```

Process noise covariance:
```
Q = [[1.0, 0.5],
     [0.5, 1.0]]
```

Measurement matrix:
```
H = [1, 0]
```

The filter adapts measurement noise based on innovation sequence statistics.

### Entropy Calculations

Sample entropy computation for RR interval sequence X with pattern length m=2 and tolerance r=0.2:

1. Count template matches within tolerance
2. Calculate conditional probabilities 
3. Compute negative log likelihood ratio

Approximate entropy uses similar methodology with self-matching included.

### Quality Assessment

Signal quality score combines:
- **Signal-to-noise ratio**: Signal power / noise power estimate
- **Amplitude consistency**: Coefficient of variation
- **Baseline stability**: Drift measurement

Overall quality = min(1.0, SNR/10) × min(1.0, 1/CV) × min(1.0, 1/drift)

## Validation and Testing

### Statistical Validation
- Cross-correlation analysis between sensors
- Significance testing with p < 0.05 threshold  
- Bootstrap resampling for confidence intervals
- Leave-one-out validation for robustness

### Performance Benchmarks
- Processing time: <1 second per 1000 measurements
- Memory usage: <100MB for typical datasets
- Accuracy: Within 5% of professional references when calibrated

## Limitations and Considerations

### Technical Limitations
- Requires synchronized timestamps across sensors
- Motion artifacts can degrade signal quality
- Consumer sensor accuracy varies by device and conditions
- Calibration quality depends on professional reference data

### Methodological Considerations
- Statistical fusion assumes independent sensor errors
- QRS detection requires minimum signal quality
- HRV analysis needs sufficient data duration
- Personalization requires individual calibration data

## Scientific Background

This framework builds on established principles:
- Kalman filtering theory for optimal estimation
- Digital signal processing for biological signals
- Heart rate variability analysis standards
- Consumer device measurement characteristics

The approach addresses limitations in consumer cardiovascular monitoring through:
- Statistical sensor fusion techniques
- Individual calibration procedures  
- Comprehensive quality assessment
- Uncertainty quantification methods

## Contributing and Extension

The framework is designed for extension and modification:
- Modular architecture for adding new sensors
- Configurable algorithm parameters
- Extensible analysis pipeline
- Standard data format compatibility

Potential enhancements include:
- Additional signal quality metrics
- Extended HRV analysis methods
- Real-time processing capabilities
- Integration with additional device types

## File Structure

```
cardiovascular-analysis/
├── src/
│   ├── cardiovascular_oscillatory_suite.py    # Core analysis framework
│   ├── entropy_oscillation_coupling_framework.py
│   └── universal_transformation_framework.py
├── analyze_cardiovascular_data.py             # Main analysis script
├── sample_heart_rate_sensors_config.json      # Example configuration
├── sample_professional_cardiovascular_data.json
├── README_Cardiovascular_Analysis_Suite.md    # Detailed documentation
└── README.md                                  # This file
```

## References and Standards

- Heart rate variability measurement standards (Task Force, 1996)
- Pan-Tompkins QRS detection algorithm (Pan & Tompkins, 1985)
- Kalman filtering theory (Kalman, 1960)
- Consumer device accuracy studies (various)

For detailed methodology and validation results, see the comprehensive documentation in `README_Cardiovascular_Analysis_Suite.md`.
