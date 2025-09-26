"""
Circulatory System Oscillatory Dynamics Validator

This module validates the theoretical predictions about cardiovascular oscillations,
heart rate variability, and multi-scale circulatory coupling within the unified
biological oscillations framework.

Key Validations:
1. Heart Rate Variability as Multi-Scale Coupling
2. Blood Pressure Oscillation Coupling 
3. Cardiovascular System Integration
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.integrate import odeint
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime

class CirculatoryOscillatoryValidator:
    """Validates oscillatory dynamics in cardiovascular systems"""
    
    def __init__(self, results_dir: str = "results/circulatory"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Cardiovascular parameters
        self.heart_rate_base = 70  # beats/min
        self.cardiac_period = 60 / self.heart_rate_base  # seconds
        self.systolic_pressure = 120  # mmHg
        self.diastolic_pressure = 80   # mmHg
        
        # Oscillatory coupling parameters
        self.respiratory_coupling = 0.15  # RSA strength
        self.autonomic_coupling = 0.25    # ANS modulation
        self.baroreceptor_coupling = 0.30 # Baroreceptor sensitivity
        
        # Multi-scale frequency ranges (Hz)
        self.frequency_bands = {
            'ultra_low': (0.0001, 0.003),   # Circadian influences
            'very_low': (0.003, 0.04),      # Renin-angiotensin system
            'low': (0.04, 0.15),            # Baroreceptor control
            'high': (0.15, 0.40),           # Respiratory coupling
            'very_high': (0.40, 2.0)        # Cardiac dynamics
        }
        
        self.validation_results = {}
        
    def experiment_1_heart_rate_variability_coupling(self) -> Dict[str, Any]:
        """
        Experiment 1: Heart Rate Variability as Multi-Scale Oscillatory Coupling
        
        Validates that HRV emerges from synchronized coupling between:
        - Cardiac pacemaker oscillations
        - Respiratory rhythm coupling (RSA)
        - Autonomic nervous system modulation
        - Baroreceptor feedback loops
        """
        print("🫀 Experiment 1: Heart Rate Variability Multi-Scale Coupling")
        
        # Simulation parameters
        duration = 300  # 5 minutes
        fs = 4.0  # 4 Hz sampling
        t = np.linspace(0, duration, int(duration * fs))
        
        # Generate multi-scale oscillatory components
        # 1. Base cardiac oscillation
        cardiac_base = np.sin(2 * np.pi * (self.heart_rate_base / 60) * t)
        
        # 2. Respiratory coupling (RSA - Respiratory Sinus Arrhythmia)
        resp_freq = 0.25  # 15 breaths/min
        respiratory_modulation = self.respiratory_coupling * np.sin(2 * np.pi * resp_freq * t)
        
        # 3. Low-frequency autonomic modulation
        lf_freq = 0.1  # 0.1 Hz
        autonomic_modulation = self.autonomic_coupling * np.sin(2 * np.pi * lf_freq * t)
        
        # 4. Very low frequency circadian influence
        vlf_freq = 0.02  # 0.02 Hz
        circadian_modulation = 0.1 * np.sin(2 * np.pi * vlf_freq * t)
        
        # 5. Baroreceptor feedback (coupling to blood pressure)
        bp_oscillation = np.sin(2 * np.pi * 0.08 * t)  # Mayer waves
        baroreceptor_modulation = self.baroreceptor_coupling * bp_oscillation
        
        # Combined HRV signal with multi-scale coupling
        hrv_modulation = (respiratory_modulation + 
                         autonomic_modulation + 
                         circadian_modulation + 
                         baroreceptor_modulation)
        
        # Generate RR intervals (heart rate variability)
        instantaneous_hr = self.heart_rate_base + 10 * hrv_modulation
        rr_intervals = 60.0 / instantaneous_hr  # Convert to RR intervals
        
        # Calculate HRV metrics
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000  # ms
        sdnn = np.std(rr_intervals) * 1000  # ms
        pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(np.diff(rr_intervals)) * 100
        
        # Frequency domain analysis
        freqs, psd = signal.welch(rr_intervals, fs=fs, nperseg=min(256, len(rr_intervals)//4))
        
        # Extract power in different frequency bands
        band_powers = {}
        total_power = np.trapz(psd, freqs)
        
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            band_mask = (freqs >= f_min) & (freqs <= f_max)
            if np.any(band_mask):
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers[band_name] = band_power / total_power * 100  # Percentage
            else:
                band_powers[band_name] = 0
        
        # Coupling analysis between frequency bands
        coupling_coefficients = {}
        for i, (band1, (f1_min, f1_max)) in enumerate(self.frequency_bands.items()):
            for j, (band2, (f2_min, f2_max)) in enumerate(self.frequency_bands.items()):
                if i < j:  # Avoid duplicate pairs
                    # Extract signals in each band
                    b1, a1 = signal.butter(4, [f1_min, f1_max], btype='band', fs=fs)
                    b2, a2 = signal.butter(4, [f2_min, f2_max], btype='band', fs=fs)
                    
                    sig1 = signal.filtfilt(b1, a1, rr_intervals)
                    sig2 = signal.filtfilt(b2, a2, rr_intervals)
                    
                    # Calculate coupling strength (correlation)
                    coupling = np.corrcoef(sig1, sig2)[0, 1]
                    coupling_coefficients[f"{band1}_{band2}"] = coupling
        
        # Theoretical predictions validation
        theoretical_predictions = {
            'respiratory_coupling_strength': 0.15,
            'autonomic_coupling_strength': 0.25,
            'baroreceptor_coupling_strength': 0.30,
            'expected_rmssd_range': (20, 60),  # ms
            'expected_lf_hf_ratio': (0.5, 2.0)
        }
        
        # Calculate LF/HF ratio
        lf_hf_ratio = band_powers['low'] / max(band_powers['high'], 1e-6)
        
        # Validation metrics
        validation_success = (
            theoretical_predictions['expected_rmssd_range'][0] <= rmssd <= 
            theoretical_predictions['expected_rmssd_range'][1] and
            theoretical_predictions['expected_lf_hf_ratio'][0] <= lf_hf_ratio <= 
            theoretical_predictions['expected_lf_hf_ratio'][1]
        )
        
        results = {
            'experiment': 'Heart Rate Variability Multi-Scale Coupling',
            'validation_success': validation_success,
            'hrv_metrics': {
                'rmssd_ms': rmssd,
                'sdnn_ms': sdnn,
                'pnn50_percent': pnn50,
                'lf_hf_ratio': lf_hf_ratio
            },
            'band_powers': band_powers,
            'coupling_coefficients': coupling_coefficients,
            'theoretical_predictions': theoretical_predictions,
            'oscillatory_components': {
                'time': t,
                'rr_intervals': rr_intervals,
                'respiratory_component': respiratory_modulation,
                'autonomic_component': autonomic_modulation,
                'baroreceptor_component': baroreceptor_modulation
            }
        }
        
        # Visualization
        self._plot_hrv_analysis(results)
        
        return results
    
    def experiment_2_blood_pressure_oscillation_coupling(self) -> Dict[str, Any]:
        """
        Experiment 2: Blood Pressure Oscillation Coupling
        
        Validates oscillatory coupling between:
        - Cardiac pump oscillations  
        - Vascular resistance oscillations
        - Autonomic control oscillations
        - Mayer wave generation through coupling
        """
        print("🩸 Experiment 2: Blood Pressure Oscillation Coupling")
        
        # Simulation parameters
        duration = 400  # seconds
        fs = 10.0  # 10 Hz sampling
        t = np.linspace(0, duration, int(duration * fs))
        
        # Cardiovascular system model with oscillatory coupling
        def cardiovascular_oscillator(state, t):
            """
            Coupled cardiovascular oscillator model:
            - Heart rate oscillations
            - Vascular resistance oscillations  
            - Blood pressure dynamics
            - Autonomic feedback coupling
            """
            hr, bp_systolic, bp_diastolic, vascular_tone = state
            
            # Heart rate dynamics with autonomic coupling
            hr_target = self.heart_rate_base + 15 * np.sin(0.1 * 2 * np.pi * t)  # LF modulation
            hr_dot = 0.5 * (hr_target - hr) + 2 * np.sin(0.25 * 2 * np.pi * t)  # Respiratory coupling
            
            # Blood pressure dynamics coupled to heart rate
            stroke_volume = 70 + 10 * np.sin(hr * 2 * np.pi * t / 60)  # mL
            cardiac_output = hr * stroke_volume / 1000  # L/min
            
            # Vascular resistance oscillations (Mayer waves)
            resistance_base = 1200  # dyn⋅s/cm^5
            resistance_oscillation = resistance_base * (1 + 0.15 * np.sin(0.08 * 2 * np.pi * t))
            
            # Blood pressure calculation with oscillatory coupling
            mean_bp = cardiac_output * resistance_oscillation / 80
            pulse_pressure = 40 + 10 * np.cos(hr * 2 * np.pi * t / 60)
            
            bp_systolic_new = mean_bp + pulse_pressure / 2
            bp_diastolic_new = mean_bp - pulse_pressure / 2
            
            # Baroreceptor feedback coupling
            bp_error = mean_bp - 100  # Target MAP = 100 mmHg
            vascular_tone_dot = -0.1 * bp_error + 0.05 * np.random.randn()
            
            # State derivatives
            bp_systolic_dot = 2 * (bp_systolic_new - bp_systolic)
            bp_diastolic_dot = 2 * (bp_diastolic_new - bp_diastolic)
            
            return [hr_dot, bp_systolic_dot, bp_diastolic_dot, vascular_tone_dot]
        
        # Initial conditions
        initial_state = [self.heart_rate_base, self.systolic_pressure, 
                        self.diastolic_pressure, 1.0]
        
        # Solve coupled oscillator system
        solution = odeint(cardiovascular_oscillator, initial_state, t)
        hr_trace = solution[:, 0]
        bp_systolic_trace = solution[:, 1]
        bp_diastolic_trace = solution[:, 2]
        vascular_tone_trace = solution[:, 3]
        
        # Calculate pulse pressure and mean arterial pressure
        pulse_pressure = bp_systolic_trace - bp_diastolic_trace
        mean_arterial_pressure = bp_diastolic_trace + pulse_pressure / 3
        
        # Frequency domain analysis of blood pressure oscillations
        freqs, bp_psd = signal.welch(mean_arterial_pressure, fs=fs, nperseg=512)
        
        # Identify Mayer waves (0.08-0.12 Hz)
        mayer_mask = (freqs >= 0.08) & (freqs <= 0.12)
        mayer_power = np.trapz(bp_psd[mayer_mask], freqs[mayer_mask])
        total_bp_power = np.trapz(bp_psd, freqs)
        mayer_wave_strength = mayer_power / total_bp_power
        
        # Cross-coupling analysis between HR and BP
        hr_bp_coupling = np.corrcoef(hr_trace, mean_arterial_pressure)[0, 1]
        
        # Phase coupling analysis
        from scipy.signal import hilbert
        hr_analytic = hilbert(hr_trace - np.mean(hr_trace))
        bp_analytic = hilbert(mean_arterial_pressure - np.mean(mean_arterial_pressure))
        
        hr_phase = np.angle(hr_analytic)
        bp_phase = np.angle(bp_analytic)
        phase_difference = hr_phase - bp_phase
        
        # Calculate phase coupling strength
        phase_coupling_strength = np.abs(np.mean(np.exp(1j * phase_difference)))
        
        # Theoretical validation
        theoretical_expectations = {
            'mayer_wave_frequency_range': (0.08, 0.12),
            'expected_mayer_power_percentage': (10, 30),
            'hr_bp_coupling_range': (-0.8, -0.3),  # Negative due to baroreceptor reflex
            'phase_coupling_threshold': 0.3
        }
        
        validation_success = (
            theoretical_expectations['expected_mayer_power_percentage'][0] <= 
            mayer_wave_strength * 100 <= 
            theoretical_expectations['expected_mayer_power_percentage'][1] and
            phase_coupling_strength >= theoretical_expectations['phase_coupling_threshold']
        )
        
        results = {
            'experiment': 'Blood Pressure Oscillation Coupling',
            'validation_success': validation_success,
            'oscillatory_metrics': {
                'mayer_wave_strength': mayer_wave_strength,
                'hr_bp_coupling': hr_bp_coupling,
                'phase_coupling_strength': phase_coupling_strength
            },
            'physiological_traces': {
                'time': t,
                'heart_rate': hr_trace,
                'systolic_bp': bp_systolic_trace,
                'diastolic_bp': bp_diastolic_trace,
                'mean_arterial_pressure': mean_arterial_pressure,
                'pulse_pressure': pulse_pressure
            },
            'frequency_analysis': {
                'frequencies': freqs,
                'bp_power_spectral_density': bp_psd
            },
            'theoretical_expectations': theoretical_expectations
        }
        
        # Visualization
        self._plot_bp_oscillation_analysis(results)
        
        return results
    
    def experiment_3_cardiovascular_system_integration(self) -> Dict[str, Any]:
        """
        Experiment 3: Integrated Cardiovascular System Oscillations
        
        Validates system-level integration of:
        - Cardiac output oscillations
        - Venous return oscillations  
        - Peripheral circulation oscillations
        - Respiratory-cardiovascular coupling
        """
        print("❤️ Experiment 3: Cardiovascular System Integration")
        
        # Advanced cardiovascular model parameters
        duration = 600  # 10 minutes
        fs = 8.0
        t = np.linspace(0, duration, int(duration * fs))
        
        # Multi-compartment cardiovascular model
        def integrated_cv_system(state, t):
            """
            Integrated cardiovascular system with multiple coupled oscillators:
            - Left ventricle
            - Aortic compliance
            - Systemic circulation
            - Venous return
            - Right heart
            - Pulmonary circulation
            """
            (lv_volume, aortic_pressure, systemic_flow, venous_pressure,
             rv_volume, pulmonary_pressure, respiratory_phase) = state
            
            # Respiratory coupling (central pattern)
            resp_freq = 0.25  # 15 breaths/min
            respiratory_phase_dot = 2 * np.pi * resp_freq
            resp_amplitude = np.sin(respiratory_phase)
            
            # Left ventricular dynamics
            lv_elastance_base = 2.5  # mmHg/mL
            lv_elastance = lv_elastance_base * (1 + 0.3 * resp_amplitude)  # Respiratory coupling
            
            lv_pressure = lv_elastance * (lv_volume - 10)  # V0 = 10 mL
            
            # Aortic valve and systemic circulation
            if lv_pressure > aortic_pressure:
                aortic_flow = 200 * (lv_pressure - aortic_pressure)  # mL/s
                lv_volume_dot = -aortic_flow
            else:
                aortic_flow = 0
                lv_volume_dot = venous_pressure * 5  # Venous return
            
            # Aortic compliance and windkessel effect
            aortic_compliance = 1.5 + 0.3 * resp_amplitude  # mL/mmHg
            aortic_pressure_dot = (aortic_flow - systemic_flow) / aortic_compliance
            
            # Systemic circulation with oscillatory resistance
            systemic_resistance_base = 1.2  # mmHg⋅s/mL
            # Mayer wave modulation
            systemic_resistance = systemic_resistance_base * (1 + 0.2 * np.sin(0.1 * 2 * np.pi * t))
            systemic_flow_dot = (aortic_pressure - venous_pressure) / systemic_resistance - systemic_flow
            
            # Venous return dynamics
            venous_compliance = 50 + 10 * resp_amplitude  # Respiratory coupling
            venous_pressure_dot = (systemic_flow - lv_volume_dot) / venous_compliance
            
            # Right heart dynamics (simplified)
            rv_elastance = 0.5 + 0.2 * resp_amplitude
            rv_pressure = rv_elastance * (rv_volume - 5)
            
            if venous_pressure > rv_pressure:
                rv_inflow = 100 * (venous_pressure - rv_pressure)
                rv_volume_dot = rv_inflow
            else:
                rv_volume_dot = -pulmonary_pressure * 3
            
            # Pulmonary circulation
            pulmonary_resistance = 0.2
            pulmonary_pressure_dot = (rv_volume_dot - systemic_flow) / pulmonary_resistance
            
            return [lv_volume_dot, aortic_pressure_dot, systemic_flow_dot, 
                   venous_pressure_dot, rv_volume_dot, pulmonary_pressure_dot, 
                   respiratory_phase_dot]
        
        # Initial conditions for integrated system
        initial_conditions = [80, 100, 5000, 5, 30, 15, 0]  # Physiological values
        
        # Solve integrated cardiovascular system
        solution = odeint(integrated_cv_system, initial_conditions, t)
        
        lv_volume = solution[:, 0]
        aortic_pressure = solution[:, 1] 
        systemic_flow = solution[:, 2]
        venous_pressure = solution[:, 3]
        rv_volume = solution[:, 4]
        pulmonary_pressure = solution[:, 5]
        respiratory_phase = solution[:, 6]
        
        # Calculate derived cardiovascular metrics
        cardiac_output = np.gradient(lv_volume, t) * -1 * 60 / 1000  # L/min
        stroke_volume = np.max(lv_volume) - np.min(lv_volume)  # mL
        heart_rate = 60000 / np.mean(np.diff(np.where(np.diff(lv_volume) > 0)[0])) * fs  # bpm
        
        # Multi-scale coupling analysis
        signals = {
            'cardiac_output': cardiac_output,
            'aortic_pressure': aortic_pressure,
            'systemic_flow': systemic_flow,
            'venous_pressure': venous_pressure,
            'respiratory_signal': np.sin(respiratory_phase)
        }
        
        # Cross-correlation analysis between subsystems
        coupling_matrix = np.zeros((len(signals), len(signals)))
        signal_names = list(signals.keys())
        
        for i, sig1_name in enumerate(signal_names):
            for j, sig2_name in enumerate(signal_names):
                sig1 = signals[sig1_name]
                sig2 = signals[sig2_name]
                
                # Normalize signals
                sig1_norm = (sig1 - np.mean(sig1)) / np.std(sig1)
                sig2_norm = (sig2 - np.mean(sig2)) / np.std(sig2)
                
                coupling_matrix[i, j] = np.corrcoef(sig1_norm, sig2_norm)[0, 1]
        
        # Respiratory-cardiovascular coupling quantification
        resp_cardiac_coupling = np.corrcoef(
            np.sin(respiratory_phase), cardiac_output)[0, 1]
        resp_venous_coupling = np.corrcoef(
            np.sin(respiratory_phase), venous_pressure)[0, 1]
        
        # System-level oscillatory coherence
        # Calculate phase coherence across all subsystems
        from scipy.signal import hilbert
        
        phase_coherence_scores = []
        for sig_name, signal_data in signals.items():
            if sig_name != 'respiratory_signal':
                analytic_signal = hilbert(signal_data - np.mean(signal_data))
                signal_phase = np.angle(analytic_signal)
                resp_phase_wrapped = np.angle(np.exp(1j * respiratory_phase))
                
                phase_coherence = np.abs(np.mean(np.exp(1j * (signal_phase - resp_phase_wrapped))))
                phase_coherence_scores.append(phase_coherence)
        
        system_phase_coherence = np.mean(phase_coherence_scores)
        
        # Theoretical predictions for integrated system
        theoretical_predictions = {
            'expected_co_range': (4, 8),  # L/min cardiac output
            'expected_sv_range': (60, 100),  # mL stroke volume
            'resp_cardiac_coupling_min': 0.3,
            'system_coherence_min': 0.25,
            'coupling_strength_threshold': 0.4
        }
        
        # Validation metrics
        validation_success = (
            theoretical_predictions['expected_co_range'][0] <= np.mean(cardiac_output) <= 
            theoretical_predictions['expected_co_range'][1] and
            abs(resp_cardiac_coupling) >= theoretical_predictions['resp_cardiac_coupling_min'] and
            system_phase_coherence >= theoretical_predictions['system_coherence_min']
        )
        
        results = {
            'experiment': 'Cardiovascular System Integration',
            'validation_success': validation_success,
            'system_metrics': {
                'mean_cardiac_output': np.mean(cardiac_output),
                'mean_stroke_volume': stroke_volume,
                'estimated_heart_rate': heart_rate,
                'system_phase_coherence': system_phase_coherence
            },
            'coupling_analysis': {
                'coupling_matrix': coupling_matrix,
                'signal_names': signal_names,
                'resp_cardiac_coupling': resp_cardiac_coupling,
                'resp_venous_coupling': resp_venous_coupling
            },
            'physiological_traces': {
                'time': t,
                'lv_volume': lv_volume,
                'aortic_pressure': aortic_pressure,
                'systemic_flow': systemic_flow,
                'venous_pressure': venous_pressure,
                'cardiac_output': cardiac_output,
                'respiratory_phase': respiratory_phase
            },
            'theoretical_predictions': theoretical_predictions
        }
        
        # Visualization
        self._plot_integrated_cv_analysis(results)
        
        return results
    
    def _plot_hrv_analysis(self, results: Dict[str, Any]):
        """Create comprehensive HRV analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        data = results['oscillatory_components']
        metrics = results['hrv_metrics']
        bands = results['band_powers']
        
        # Plot 1: RR interval time series
        axes[0, 0].plot(data['time'][:1000], data['rr_intervals'][:1000], 'b-', linewidth=1)
        axes[0, 0].set_title('RR Interval Time Series')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('RR Interval (s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Oscillatory components
        axes[0, 1].plot(data['time'][:500], data['respiratory_component'][:500], 
                       label='Respiratory', linewidth=2)
        axes[0, 1].plot(data['time'][:500], data['autonomic_component'][:500], 
                       label='Autonomic', linewidth=2)
        axes[0, 1].plot(data['time'][:500], data['baroreceptor_component'][:500], 
                       label='Baroreceptor', linewidth=2)
        axes[0, 1].set_title('Multi-Scale Oscillatory Components')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Modulation Amplitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Frequency band powers
        band_names = list(bands.keys())
        band_values = list(bands.values())
        colors = plt.cm.viridis(np.linspace(0, 1, len(band_names)))
        
        bars = axes[0, 2].bar(band_names, band_values, color=colors)
        axes[0, 2].set_title('HRV Frequency Band Powers')
        axes[0, 2].set_ylabel('Power (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, band_values):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 4: HRV metrics
        metric_names = ['RMSSD (ms)', 'SDNN (ms)', 'pNN50 (%)', 'LF/HF Ratio']
        metric_values = [metrics['rmssd_ms'], metrics['sdnn_ms'], 
                        metrics['pnn50_percent'], metrics['lf_hf_ratio']]
        
        axes[1, 0].bar(metric_names, metric_values, color='lightcoral')
        axes[1, 0].set_title('HRV Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Coupling coefficients heatmap
        coupling_data = results['coupling_coefficients']
        if coupling_data:
            coupling_names = list(coupling_data.keys())
            coupling_values = list(coupling_data.values())
            
            # Create coupling matrix for heatmap
            n_bands = len(self.frequency_bands)
            coupling_matrix = np.zeros((n_bands, n_bands))
            band_list = list(self.frequency_bands.keys())
            
            for coupling_name, coupling_value in coupling_data.items():
                bands_pair = coupling_name.split('_')
                if len(bands_pair) == 2:
                    try:
                        i = band_list.index(bands_pair[0])
                        j = band_list.index(bands_pair[1])
                        coupling_matrix[i, j] = coupling_value
                        coupling_matrix[j, i] = coupling_value  # Symmetric
                    except ValueError:
                        continue
            
            # Fill diagonal with 1s
            np.fill_diagonal(coupling_matrix, 1.0)
            
            im = axes[1, 1].imshow(coupling_matrix, cmap='RdBu_r', aspect='auto',
                                 vmin=-1, vmax=1)
            axes[1, 1].set_xticks(range(len(band_list)))
            axes[1, 1].set_yticks(range(len(band_list)))
            axes[1, 1].set_xticklabels(band_list, rotation=45)
            axes[1, 1].set_yticklabels(band_list)
            axes[1, 1].set_title('Inter-Band Coupling Matrix')
            plt.colorbar(im, ax=axes[1, 1])
        
        # Plot 6: Validation summary
        validation_text = (
            f"Validation Success: {'✓' if results['validation_success'] else '✗'}\n\n"
            f"RMSSD: {metrics['rmssd_ms']:.1f} ms\n"
            f"Expected: {results['theoretical_predictions']['expected_rmssd_range']}\n\n"
            f"LF/HF Ratio: {metrics['lf_hf_ratio']:.2f}\n"
            f"Expected: {results['theoretical_predictions']['expected_lf_hf_ratio']}"
        )
        
        axes[1, 2].text(0.05, 0.95, validation_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Validation Summary')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/hrv_multi_scale_coupling_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bp_oscillation_analysis(self, results: Dict[str, Any]):
        """Create blood pressure oscillation analysis visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        traces = results['physiological_traces']
        freq_data = results['frequency_analysis']
        metrics = results['oscillatory_metrics']
        
        # Plot 1: Blood pressure traces
        axes[0, 0].plot(traces['time'], traces['systolic_bp'], 'r-', 
                       label='Systolic', linewidth=1.5)
        axes[0, 0].plot(traces['time'], traces['diastolic_bp'], 'b-', 
                       label='Diastolic', linewidth=1.5)
        axes[0, 0].plot(traces['time'], traces['mean_arterial_pressure'], 'g-', 
                       label='MAP', linewidth=2)
        axes[0, 0].set_title('Blood Pressure Oscillations')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Pressure (mmHg)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Heart rate trace
        axes[0, 1].plot(traces['time'], traces['heart_rate'], 'purple', linewidth=2)
        axes[0, 1].set_title('Heart Rate Oscillations')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Heart Rate (bpm)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Power spectral density
        axes[0, 2].semilogy(freq_data['frequencies'], freq_data['bp_power_spectral_density'])
        
        # Highlight Mayer wave band
        mayer_mask = (freq_data['frequencies'] >= 0.08) & (freq_data['frequencies'] <= 0.12)
        axes[0, 2].fill_between(freq_data['frequencies'][mayer_mask], 
                               freq_data['bp_power_spectral_density'][mayer_mask],
                               alpha=0.3, color='red', label='Mayer Waves')
        
        axes[0, 2].set_title('Blood Pressure Power Spectral Density')
        axes[0, 2].set_xlabel('Frequency (Hz)')
        axes[0, 2].set_ylabel('PSD (mmHg²/Hz)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: HR-BP coupling scatter
        axes[1, 0].scatter(traces['heart_rate'], traces['mean_arterial_pressure'], 
                          alpha=0.6, s=10)
        
        # Add trend line
        z = np.polyfit(traces['heart_rate'], traces['mean_arterial_pressure'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(traces['heart_rate'], p(traces['heart_rate']), 
                       "r--", alpha=0.8, linewidth=2)
        
        axes[1, 0].set_title(f'HR-BP Coupling (r={metrics["hr_bp_coupling"]:.3f})')
        axes[1, 0].set_xlabel('Heart Rate (bpm)')
        axes[1, 0].set_ylabel('Mean Arterial Pressure (mmHg)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Pulse pressure dynamics
        axes[1, 1].plot(traces['time'], traces['pulse_pressure'], 'orange', linewidth=2)
        axes[1, 1].set_title('Pulse Pressure Oscillations')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Pulse Pressure (mmHg)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Validation metrics
        metric_names = ['Mayer Wave\nStrength (%)', 'HR-BP\nCoupling', 'Phase\nCoupling']
        metric_values = [metrics['mayer_wave_strength'] * 100, 
                        abs(metrics['hr_bp_coupling']),
                        metrics['phase_coupling_strength']]
        colors = ['green' if results['validation_success'] else 'red'] * 3
        
        bars = axes[1, 2].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[1, 2].set_title('Oscillatory Coupling Metrics')
        axes[1, 2].set_ylabel('Metric Value')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Add validation status
        status_text = f"Validation: {'PASS' if results['validation_success'] else 'FAIL'}"
        axes[1, 2].text(0.5, 0.9, status_text, transform=axes[1, 2].transAxes,
                       ha='center', fontsize=14, weight='bold',
                       color='green' if results['validation_success'] else 'red')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/bp_oscillation_coupling_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_integrated_cv_analysis(self, results: Dict[str, Any]):
        """Create integrated cardiovascular system analysis visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        traces = results['physiological_traces']
        coupling = results['coupling_analysis']
        metrics = results['system_metrics']
        
        # Plot 1: Left ventricular volume
        axes[0, 0].plot(traces['time'], traces['lv_volume'], 'red', linewidth=2)
        axes[0, 0].set_title('Left Ventricular Volume')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Volume (mL)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Aortic pressure
        axes[0, 1].plot(traces['time'], traces['aortic_pressure'], 'blue', linewidth=2)
        axes[0, 1].set_title('Aortic Pressure')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Pressure (mmHg)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Systemic flow
        axes[0, 2].plot(traces['time'], traces['systemic_flow'], 'green', linewidth=2)
        axes[0, 2].set_title('Systemic Flow')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Flow (mL/s)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Venous pressure
        axes[1, 0].plot(traces['time'], traces['venous_pressure'], 'purple', linewidth=2)
        axes[1, 0].set_title('Venous Pressure')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Pressure (mmHg)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cardiac output with respiratory coupling
        axes[1, 1].plot(traces['time'], traces['cardiac_output'], 'orange', 
                       linewidth=2, label='Cardiac Output')
        
        # Overlay respiratory signal (scaled)
        resp_scaled = np.sin(traces['respiratory_phase']) * 2 + np.mean(traces['cardiac_output'])
        axes[1, 1].plot(traces['time'], resp_scaled, 'cyan', alpha=0.7, 
                       linewidth=1, label='Respiratory')
        
        axes[1, 1].set_title(f'Cardiac Output (Mean: {metrics["mean_cardiac_output"]:.1f} L/min)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Cardiac Output (L/min)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: System coupling matrix heatmap
        im = axes[1, 2].imshow(coupling['coupling_matrix'], cmap='RdBu_r', 
                              aspect='auto', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(coupling['signal_names'])))
        axes[1, 2].set_yticks(range(len(coupling['signal_names'])))
        axes[1, 2].set_xticklabels(coupling['signal_names'], rotation=45)
        axes[1, 2].set_yticklabels(coupling['signal_names'])
        axes[1, 2].set_title('System Coupling Matrix')
        plt.colorbar(im, ax=axes[1, 2])
        
        # Plot 7: Respiratory-cardiovascular coupling
        axes[2, 0].scatter(np.sin(traces['respiratory_phase']), 
                          traces['cardiac_output'], alpha=0.6, s=5)
        
        # Add trend line
        resp_signal = np.sin(traces['respiratory_phase'])
        z = np.polyfit(resp_signal, traces['cardiac_output'], 1)
        p = np.poly1d(z)
        axes[2, 0].plot(resp_signal, p(resp_signal), "r--", alpha=0.8, linewidth=2)
        
        axes[2, 0].set_title(f'Resp-Cardiac Coupling (r={coupling["resp_cardiac_coupling"]:.3f})')
        axes[2, 0].set_xlabel('Respiratory Phase')
        axes[2, 0].set_ylabel('Cardiac Output (L/min)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: System metrics summary
        metric_names = ['CO\n(L/min)', 'SV\n(mL)', 'HR\n(bpm)', 'Phase\nCoherence']
        metric_values = [metrics['mean_cardiac_output'], metrics['mean_stroke_volume'],
                        metrics['estimated_heart_rate'], metrics['system_phase_coherence']]
        colors = plt.cm.viridis(np.linspace(0, 1, len(metric_values)))
        
        bars = axes[2, 1].bar(metric_names, metric_values, color=colors)
        axes[2, 1].set_title('System Performance Metrics')
        axes[2, 1].set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 9: Validation summary
        predictions = results['theoretical_predictions']
        validation_text = (
            f"🫀 CARDIOVASCULAR SYSTEM VALIDATION\n\n"
            f"✓ Validation Success: {'PASS' if results['validation_success'] else 'FAIL'}\n\n"
            f"Cardiac Output: {metrics['mean_cardiac_output']:.1f} L/min\n"
            f"Expected: {predictions['expected_co_range']}\n\n"
            f"System Coherence: {metrics['system_phase_coherence']:.3f}\n"
            f"Required: ≥{predictions['system_coherence_min']}\n\n"
            f"Resp-Cardiac Coupling: {abs(coupling['resp_cardiac_coupling']):.3f}\n"
            f"Required: ≥{predictions['resp_cardiac_coupling_min']}"
        )
        
        axes[2, 2].text(0.05, 0.95, validation_text, transform=axes[2, 2].transAxes,
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round', 
                               facecolor='lightgreen' if results['validation_success'] else 'lightcoral', 
                               alpha=0.8))
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('Validation Results')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/integrated_cardiovascular_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all circulatory oscillatory validation experiments"""
        print("🫀 Running Circulatory System Oscillatory Validation Suite")
        print("=" * 60)
        
        all_results = {}
        
        # Run experiments
        all_results['experiment_1'] = self.experiment_1_heart_rate_variability_coupling()
        all_results['experiment_2'] = self.experiment_2_blood_pressure_oscillation_coupling()
        all_results['experiment_3'] = self.experiment_3_cardiovascular_system_integration()
        
        # Compile validation summary
        validations = [result['validation_success'] for result in all_results.values()]
        overall_success = all(validations)
        
        summary = {
            'domain': 'Circulatory System Oscillations',
            'total_experiments': len(all_results),
            'successful_validations': sum(validations),
            'overall_validation_success': overall_success,
            'timestamp': datetime.now().isoformat(),
            'detailed_results': all_results
        }
        
        # Save comprehensive results
        results_file = f"{self.results_dir}/circulatory_validation_summary.json"
        import json
        with open(results_file, 'w') as f:
            json.dump({k: v for k, v in summary.items() 
                      if k != 'detailed_results'}, f, indent=2)
        
        print(f"\n🫀 Circulatory Validation Complete:")
        print(f"   ✓ Successful: {sum(validations)}/{len(validations)} experiments")
        print(f"   ✓ Overall Status: {'PASS' if overall_success else 'FAIL'}")
        print(f"   ✓ Results saved to: {self.results_dir}/")
        
        self.validation_results = summary
        return summary

def main():
    """Run circulatory oscillatory validation as standalone"""
    validator = CirculatoryOscillatoryValidator()
    return validator.run_all_experiments()

if __name__ == "__main__":
    results = main()
    print(f"Circulatory validation completed: {results['overall_validation_success']}")
