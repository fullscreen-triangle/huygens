"""
Respiratory System Oscillatory Dynamics Validator

This module validates the theoretical predictions about respiratory oscillations,
breathing rhythms, gas exchange dynamics, and atmospheric coupling within the unified
biological oscillations framework.

Key Validations:
1. Breathing Rhythm Oscillatory Coupling
2. Gas Exchange Oscillation Dynamics
3. Atmospheric-Respiratory Integration
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

class RespiratoryOscillatoryValidator:
    """Validates oscillatory dynamics in respiratory systems"""
    
    def __init__(self, results_dir: str = "results/respiratory"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Respiratory oscillation parameters
        self.respiratory_params = {
            'resting_frequency': 0.25,      # 15 breaths/min
            'tidal_volume': 500,            # mL
            'functional_residual_capacity': 2300,  # mL
            'dead_space': 150,              # mL
            'alveolar_ventilation': 4200,   # mL/min
        }
        
        # Gas exchange parameters
        self.gas_params = {
            'o2_atmospheric': 21.0,         # % oxygen
            'co2_atmospheric': 0.04,        # % CO2
            'alveolar_o2': 13.8,           # % oxygen
            'alveolar_co2': 5.2,           # % CO2
            'blood_o2_capacity': 20,        # mL O2/100mL blood
            'hemoglobin_saturation': 98,    # %
        }
        
        # Oscillatory coupling parameters
        self.coupling_params = {
            'cardiac_respiratory_ratio': 4.0,  # HR:RR ratio
            'vagal_coupling_strength': 0.3,    # Parasympathetic coupling
            'chemoreceptor_sensitivity': 0.8,   # CO2/O2 sensitivity
            'atmospheric_coupling': 0.15,      # Environmental coupling
        }
        
        # Respiratory control centers
        self.control_centers = {
            'pre_botzinger_freq': 0.25,     # Hz - primary rhythm generator
            'pontomedullary_freq': 0.1,     # Hz - modulation
            'cortical_freq': 0.05,          # Hz - voluntary control
            'chemoreceptor_freq': 0.02,     # Hz - chemical drive
        }
        
        self.validation_results = {}
        
    def experiment_1_breathing_rhythm_oscillatory_coupling(self) -> Dict[str, Any]:
        """
        Experiment 1: Breathing Rhythm Oscillatory Coupling
        
        Validates coupling between respiratory oscillatory components:
        - Pre-Bötzinger complex pacemaker
        - Pontomedullary control
        - Vagal modulation
        - Cardiac-respiratory synchronization
        """
        print("🫁 Experiment 1: Breathing Rhythm Oscillatory Coupling")
        
        # Simulation parameters
        duration = 300  # 5 minutes
        fs = 20.0  # 20 Hz sampling
        t = np.linspace(0, duration, int(duration * fs))
        
        # Respiratory control system model
        def respiratory_oscillator_system(state, t):
            """
            Multi-scale respiratory control system:
            - Pre-Bötzinger complex (primary oscillator)
            - Ponto-medullary centers (modulation)
            - Vagal control (parasympathetic)
            - Chemical drive (chemoreceptors)
            """
            pre_botzinger, ponto_medullary, vagal_control, chemical_drive = state
            
            # Pre-Bötzinger complex (primary respiratory pacemaker)
            baseline_freq = self.control_centers['pre_botzinger_freq']
            pre_botz_oscillation = np.sin(baseline_freq * 2 * np.pi * t)
            pre_botz_dot = 0.5 * (0.8 - pre_botzinger) + 0.3 * pre_botz_oscillation
            
            # Ponto-medullary modulation (breath pattern)
            ponto_freq = self.control_centers['pontomedullary_freq']
            ponto_modulation = np.sin(ponto_freq * 2 * np.pi * t)
            ponto_dot = 0.8 * (ponto_modulation - ponto_medullary)
            
            # Vagal control (heart-lung coupling)
            cardiac_freq = baseline_freq * self.coupling_params['cardiac_respiratory_ratio']
            cardiac_modulation = 0.2 * np.sin(cardiac_freq * 2 * np.pi * t)
            vagal_coupling = self.coupling_params['vagal_coupling_strength']
            vagal_dot = 1.2 * (vagal_coupling * cardiac_modulation - vagal_control)
            
            # Chemical drive (CO2/O2 chemoreceptors)
            chemoreceptor_freq = self.control_centers['chemoreceptor_freq']
            chemical_oscillation = 0.1 * np.sin(chemoreceptor_freq * 2 * np.pi * t)
            metabolic_demand = 0.6 + 0.2 * np.sin(0.01 * 2 * np.pi * t)  # Slow variation
            chemical_dot = 0.3 * (metabolic_demand - chemical_drive) + chemical_oscillation
            
            return [pre_botz_dot, ponto_dot, vagal_dot, chemical_dot]
        
        # Initial conditions
        initial_state = [0.8, 0.0, 0.15, 0.6]
        
        # Solve respiratory control system
        solution = odeint(respiratory_oscillator_system, initial_state, t)
        pre_botzinger_trace = solution[:, 0]
        ponto_medullary_trace = solution[:, 1]
        vagal_trace = solution[:, 2]
        chemical_drive_trace = solution[:, 3]
        
        # Generate breathing pattern from control signals
        # Respiratory drive (combination of control centers)
        respiratory_drive = (0.6 * pre_botzinger_trace + 
                            0.2 * ponto_medullary_trace + 
                            0.1 * vagal_trace + 
                            0.1 * chemical_drive_trace)
        
        # Breathing pattern generation
        breath_threshold = 0.7
        breathing_signal = np.where(respiratory_drive > breath_threshold, 1, 0)
        
        # Generate realistic airflow pattern
        airflow = np.zeros_like(t)
        for i in range(1, len(breathing_signal)):
            if breathing_signal[i] == 1 and breathing_signal[i-1] == 0:  # Inspiration start
                # Create inspiratory flow pattern
                inspiration_duration = int(1.5 * fs)  # 1.5 seconds
                end_idx = min(i + inspiration_duration, len(airflow))
                inspiration_pattern = np.sin(np.linspace(0, np.pi, end_idx - i))
                airflow[i:end_idx] = inspiration_pattern * 500  # mL/s peak flow
            elif breathing_signal[i] == 0 and breathing_signal[i-1] == 1:  # Expiration start
                expiration_duration = int(2.5 * fs)  # 2.5 seconds
                end_idx = min(i + expiration_duration, len(airflow))
                expiration_pattern = -np.sin(np.linspace(0, np.pi, end_idx - i))
                airflow[i:end_idx] = expiration_pattern * 300  # mL/s peak flow
        
        # Calculate lung volume
        lung_volume = np.zeros_like(t)
        lung_volume[0] = self.respiratory_params['functional_residual_capacity']
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            lung_volume[i] = lung_volume[i-1] + airflow[i] * dt
            # Prevent negative volumes
            lung_volume[i] = max(lung_volume[i], 1000)  # Minimum 1L
        
        # Extract breathing metrics
        # Find breath cycles
        breath_starts = np.where(np.diff(breathing_signal) == 1)[0]
        if len(breath_starts) > 1:
            inter_breath_intervals = np.diff(breath_starts) / fs
            breathing_rate = 60 / np.mean(inter_breath_intervals)  # breaths/min
            breathing_variability = np.std(inter_breath_intervals)
        else:
            breathing_rate = 15  # Default
            breathing_variability = 0
        
        # Tidal volume calculation
        if len(breath_starts) > 0:
            tidal_volumes = []
            for start_idx in breath_starts:
                end_idx = min(start_idx + int(4 * fs), len(lung_volume))  # 4 second window
                breath_volume = lung_volume[start_idx:end_idx]
                if len(breath_volume) > 0:
                    tidal_vol = np.max(breath_volume) - np.min(breath_volume)
                    tidal_volumes.append(tidal_vol)
            
            mean_tidal_volume = np.mean(tidal_volumes) if tidal_volumes else 500
            tidal_volume_variability = np.std(tidal_volumes) if len(tidal_volumes) > 1 else 0
        else:
            mean_tidal_volume = 500
            tidal_volume_variability = 0
        
        # Frequency domain analysis of respiratory control
        freqs, drive_psd = signal.welch(respiratory_drive, fs=fs, nperseg=256)
        
        # Extract power in different control frequency bands
        control_band_powers = {}
        total_power = np.trapz(drive_psd, freqs)
        
        control_bands = {
            'primary_rhythm': (0.2, 0.35),     # Pre-Bötzinger
            'modulation': (0.05, 0.15),        # Ponto-medullary
            'chemical_drive': (0.01, 0.05),    # Chemoreceptors
            'vagal_coupling': (0.8, 1.5),      # Cardiac coupling
        }
        
        for band_name, (f_min, f_max) in control_bands.items():
            band_mask = (freqs >= f_min) & (freqs <= f_max)
            if np.any(band_mask):
                band_power = np.trapz(drive_psd[band_mask], freqs[band_mask])
                control_band_powers[band_name] = band_power / total_power * 100
            else:
                control_band_powers[band_name] = 0
        
        # Cross-coupling analysis between control centers
        control_signals = {
            'pre_botzinger': pre_botzinger_trace,
            'ponto_medullary': ponto_medullary_trace,
            'vagal': vagal_trace,
            'chemical_drive': chemical_drive_trace
        }
        
        coupling_matrix = np.zeros((len(control_signals), len(control_signals)))
        signal_names = list(control_signals.keys())
        
        for i, sig1_name in enumerate(signal_names):
            for j, sig2_name in enumerate(signal_names):
                sig1 = control_signals[sig1_name]
                sig2 = control_signals[sig2_name]
                coupling_matrix[i, j] = np.corrcoef(sig1, sig2)[0, 1]
        
        # Cardiac-respiratory coupling analysis
        cardiac_frequency = breathing_rate / 60 * self.coupling_params['cardiac_respiratory_ratio']
        synthetic_cardiac_signal = np.sin(cardiac_frequency * 2 * np.pi * t)
        
        cardiac_respiratory_coupling = np.corrcoef(respiratory_drive, synthetic_cardiac_signal)[0, 1]
        
        # Theoretical predictions
        theoretical_predictions = {
            'expected_breathing_rate': (12, 18),    # breaths/min
            'expected_tidal_volume': (400, 600),    # mL
            'primary_rhythm_power_min': 30,         # % of total power
            'cardiac_respiratory_coupling_min': 0.4,
            'breathing_variability_max': 1.0        # seconds std
        }
        
        # Validation
        validation_success = (
            theoretical_predictions['expected_breathing_rate'][0] <= breathing_rate <= 
            theoretical_predictions['expected_breathing_rate'][1] and
            theoretical_predictions['expected_tidal_volume'][0] <= mean_tidal_volume <= 
            theoretical_predictions['expected_tidal_volume'][1] and
            control_band_powers['primary_rhythm'] >= theoretical_predictions['primary_rhythm_power_min'] and
            abs(cardiac_respiratory_coupling) >= theoretical_predictions['cardiac_respiratory_coupling_min'] and
            breathing_variability <= theoretical_predictions['breathing_variability_max']
        )
        
        results = {
            'experiment': 'Breathing Rhythm Oscillatory Coupling',
            'validation_success': validation_success,
            'respiratory_metrics': {
                'breathing_rate': breathing_rate,
                'mean_tidal_volume': mean_tidal_volume,
                'breathing_variability': breathing_variability,
                'tidal_volume_variability': tidal_volume_variability
            },
            'control_traces': {
                'time': t,
                'pre_botzinger': pre_botzinger_trace,
                'ponto_medullary': ponto_medullary_trace,
                'vagal': vagal_trace,
                'chemical_drive': chemical_drive_trace,
                'respiratory_drive': respiratory_drive
            },
            'breathing_pattern': {
                'airflow': airflow,
                'lung_volume': lung_volume,
                'breathing_signal': breathing_signal
            },
            'coupling_analysis': {
                'coupling_matrix': coupling_matrix,
                'signal_names': signal_names,
                'cardiac_respiratory_coupling': cardiac_respiratory_coupling
            },
            'frequency_analysis': {
                'frequencies': freqs,
                'power_spectral_density': drive_psd,
                'control_band_powers': control_band_powers
            },
            'theoretical_predictions': theoretical_predictions
        }
        
        # Visualization
        self._plot_breathing_rhythm_analysis(results)
        
        return results
    
    def experiment_2_gas_exchange_oscillation_dynamics(self) -> Dict[str, Any]:
        """
        Experiment 2: Gas Exchange Oscillation Dynamics
        
        Validates oscillatory patterns in gas exchange:
        - Alveolar ventilation oscillations
        - O2/CO2 concentration cycles
        - Hemoglobin saturation oscillations
        - Diffusion rate oscillations
        """
        print("🔄 Experiment 2: Gas Exchange Oscillation Dynamics")
        
        # Simulation parameters
        duration = 240  # 4 minutes
        fs = 10.0   # 10 Hz sampling
        t = np.linspace(0, duration, int(duration * fs))
        
        # Gas exchange system model
        def gas_exchange_system(state, t):
            """
            Multi-compartment gas exchange model:
            - Alveolar gas concentrations
            - Blood gas concentrations
            - Hemoglobin binding dynamics
            - Tissue gas exchange
            """
            (alveolar_o2, alveolar_co2, blood_o2, blood_co2, 
             hb_saturation, tissue_o2) = state
            
            # Breathing pattern (oscillatory ventilation)
            breath_freq = self.respiratory_params['resting_frequency']
            ventilation_rate = 5000 + 1000 * np.sin(breath_freq * 2 * np.pi * t)  # mL/min
            
            # Alveolar ventilation dynamics
            # O2 uptake from atmosphere
            atmospheric_o2_flow = ventilation_rate * self.gas_params['o2_atmospheric'] / 100
            # O2 uptake by blood
            o2_uptake_rate = 250 * (alveolar_o2 - blood_o2) / 100  # mL/min
            
            alveolar_o2_dot = (atmospheric_o2_flow - o2_uptake_rate) / 3000  # %/min
            
            # CO2 elimination
            co2_production_rate = 200 + 50 * np.sin(0.05 * 2 * np.pi * t)  # mL/min (metabolic variation)
            co2_elimination_rate = ventilation_rate * alveolar_co2 / 100
            
            alveolar_co2_dot = (co2_production_rate - co2_elimination_rate) / 3000  # %/min
            
            # Blood gas dynamics
            # O2 transport
            o2_delivery_rate = blood_o2 * 5000 / 100  # mL/min (cardiac output effect)
            blood_o2_dot = (o2_uptake_rate - o2_delivery_rate) / 5000  # %/min
            
            # CO2 transport
            co2_pickup_rate = tissue_o2 * 200 / 100  # CO2 production from tissue metabolism
            blood_co2_dot = (co2_pickup_rate - co2_production_rate) / 5000  # %/min
            
            # Hemoglobin saturation dynamics (oxygen dissociation curve)
            p50 = 27  # mmHg (P50 for hemoglobin)
            po2_estimate = blood_o2 * 7  # Rough conversion to mmHg
            hill_coefficient = 2.7
            
            target_saturation = 100 * (po2_estimate**hill_coefficient) / (p50**hill_coefficient + po2_estimate**hill_coefficient)
            hb_saturation_dot = 2 * (target_saturation - hb_saturation)  # %/min
            
            # Tissue oxygen dynamics
            o2_consumption_rate = 250 + 50 * np.sin(0.1 * 2 * np.pi * t)  # mL/min (metabolic oscillation)
            o2_supply_rate = o2_delivery_rate * (hb_saturation / 100)
            
            tissue_o2_dot = (o2_supply_rate - o2_consumption_rate) / 1000  # %/min
            
            return [alveolar_o2_dot, alveolar_co2_dot, blood_o2_dot, 
                   blood_co2_dot, hb_saturation_dot, tissue_o2_dot]
        
        # Initial conditions (physiological values)
        initial_state = [
            self.gas_params['alveolar_o2'],      # 13.8% O2
            self.gas_params['alveolar_co2'],     # 5.2% CO2
            19.0,   # Blood O2 %
            6.0,    # Blood CO2 %
            self.gas_params['hemoglobin_saturation'],  # 98% Hb saturation
            15.0    # Tissue O2 %
        ]
        
        # Solve gas exchange system
        solution = odeint(gas_exchange_system, initial_state, t)
        alveolar_o2_trace = solution[:, 0]
        alveolar_co2_trace = solution[:, 1]
        blood_o2_trace = solution[:, 2]
        blood_co2_trace = solution[:, 3]
        hb_saturation_trace = solution[:, 4]
        tissue_o2_trace = solution[:, 5]
        
        # Calculate derived gas exchange metrics
        # Alveolar-arterial O2 gradient
        aa_gradient = alveolar_o2_trace - blood_o2_trace
        
        # Respiratory exchange ratio (RER)
        co2_elimination = np.gradient(alveolar_co2_trace, t[1] - t[0])
        o2_consumption = -np.gradient(alveolar_o2_trace, t[1] - t[0])
        rer = np.abs(co2_elimination) / (np.abs(o2_consumption) + 1e-6)
        
        # Gas exchange efficiency
        ventilation_perfusion_ratio = 0.8 + 0.2 * np.sin(0.1 * 2 * np.pi * t)  # V/Q ratio
        gas_exchange_efficiency = hb_saturation_trace * ventilation_perfusion_ratio / 100
        
        # Oscillatory analysis of gas concentrations
        gas_signals = {
            'alveolar_o2': alveolar_o2_trace,
            'alveolar_co2': alveolar_co2_trace,
            'blood_o2': blood_o2_trace,
            'blood_co2': blood_co2_trace,
            'hb_saturation': hb_saturation_trace
        }
        
        # Frequency domain analysis
        gas_oscillation_powers = {}
        
        for gas_name, gas_signal in gas_signals.items():
            freqs, psd = signal.welch(gas_signal, fs=fs, nperseg=128)
            
            # Extract power in respiratory frequency band
            resp_mask = (freqs >= 0.2) & (freqs <= 0.4)  # Breathing frequency band
            if np.any(resp_mask):
                resp_power = np.trapz(psd[resp_mask], freqs[resp_mask])
                total_power = np.trapz(psd, freqs)
                gas_oscillation_powers[gas_name] = resp_power / total_power * 100
            else:
                gas_oscillation_powers[gas_name] = 0
        
        # Cross-coupling analysis between gas compartments
        gas_coupling_matrix = np.zeros((len(gas_signals), len(gas_signals)))
        gas_names = list(gas_signals.keys())
        
        for i, gas1_name in enumerate(gas_names):
            for j, gas2_name in enumerate(gas_names):
                gas1 = gas_signals[gas1_name]
                gas2 = gas_signals[gas2_name]
                gas_coupling_matrix[i, j] = np.corrcoef(gas1, gas2)[0, 1]
        
        # O2-CO2 inverse coupling (physiological antagonism)
        o2_co2_coupling = np.corrcoef(alveolar_o2_trace, alveolar_co2_trace)[0, 1]
        
        # Hemoglobin-tissue coupling (oxygen delivery efficiency)
        hb_tissue_coupling = np.corrcoef(hb_saturation_trace, tissue_o2_trace)[0, 1]
        
        # Phase relationships between compartments
        from scipy.signal import hilbert
        
        # Alveolar-blood phase coupling
        alveolar_o2_analytic = hilbert(alveolar_o2_trace - np.mean(alveolar_o2_trace))
        blood_o2_analytic = hilbert(blood_o2_trace - np.mean(blood_o2_trace))
        
        alveolar_phase = np.angle(alveolar_o2_analytic)
        blood_phase = np.angle(blood_o2_analytic)
        
        alveolar_blood_phase_coupling = np.abs(np.mean(np.exp(1j * (alveolar_phase - blood_phase))))
        
        # Gas exchange oscillation stability
        o2_variability = np.std(alveolar_o2_trace)
        co2_variability = np.std(alveolar_co2_trace)
        saturation_variability = np.std(hb_saturation_trace)
        
        # Theoretical predictions
        theoretical_predictions = {
            'o2_co2_anticorrelation_range': (-0.8, -0.3),  # Inverse relationship
            'hb_tissue_coupling_min': 0.6,
            'alveolar_blood_phase_coupling_min': 0.7,
            'respiratory_oscillation_power_min': 40,  # % in breathing frequency band
            'o2_variability_max': 2.0,  # % std
            'saturation_variability_max': 5.0  # % std
        }
        
        # Validation
        validation_success = (
            theoretical_predictions['o2_co2_anticorrelation_range'][0] <= o2_co2_coupling <= 
            theoretical_predictions['o2_co2_anticorrelation_range'][1] and
            hb_tissue_coupling >= theoretical_predictions['hb_tissue_coupling_min'] and
            alveolar_blood_phase_coupling >= theoretical_predictions['alveolar_blood_phase_coupling_min'] and
            gas_oscillation_powers['alveolar_o2'] >= theoretical_predictions['respiratory_oscillation_power_min'] and
            o2_variability <= theoretical_predictions['o2_variability_max'] and
            saturation_variability <= theoretical_predictions['saturation_variability_max']
        )
        
        results = {
            'experiment': 'Gas Exchange Oscillation Dynamics',
            'validation_success': validation_success,
            'gas_traces': {
                'time': t,
                'alveolar_o2': alveolar_o2_trace,
                'alveolar_co2': alveolar_co2_trace,
                'blood_o2': blood_o2_trace,
                'blood_co2': blood_co2_trace,
                'hb_saturation': hb_saturation_trace,
                'tissue_o2': tissue_o2_trace
            },
            'derived_metrics': {
                'aa_gradient': aa_gradient,
                'respiratory_exchange_ratio': rer,
                'gas_exchange_efficiency': gas_exchange_efficiency
            },
            'oscillation_analysis': {
                'gas_oscillation_powers': gas_oscillation_powers,
                'o2_variability': o2_variability,
                'co2_variability': co2_variability,
                'saturation_variability': saturation_variability
            },
            'coupling_analysis': {
                'gas_coupling_matrix': gas_coupling_matrix,
                'gas_names': gas_names,
                'o2_co2_coupling': o2_co2_coupling,
                'hb_tissue_coupling': hb_tissue_coupling,
                'alveolar_blood_phase_coupling': alveolar_blood_phase_coupling
            },
            'theoretical_predictions': theoretical_predictions
        }
        
        # Visualization
        self._plot_gas_exchange_analysis(results)
        
        return results
    
    def experiment_3_atmospheric_respiratory_integration(self) -> Dict[str, Any]:
        """
        Experiment 3: Atmospheric-Respiratory Integration
        
        Validates coupling between atmospheric conditions and respiratory function:
        - Atmospheric pressure oscillations
        - Environmental oxygen variations
        - Barometric pressure effects
        - Altitude adaptation responses
        """
        print("🌍 Experiment 3: Atmospheric-Respiratory Integration")
        
        # Simulation parameters
        duration = 480  # 8 minutes
        fs = 5.0  # 5 Hz sampling  
        t = np.linspace(0, duration, int(duration * fs))
        
        # Atmospheric conditions model
        def atmospheric_respiratory_system(state, t):
            """
            Integrated atmospheric-respiratory system:
            - Atmospheric pressure variations
            - Oxygen partial pressure changes
            - Respiratory adaptation responses
            - Chemoreceptor sensitivity adjustments
            """
            (atmospheric_pressure, partial_pressure_o2, respiratory_drive,
             chemoreceptor_sensitivity, ventilation_response) = state
            
            # Atmospheric oscillations (weather patterns, altitude changes)
            pressure_base = 760  # mmHg sea level
            # Simulate atmospheric pressure variations (weather fronts, altitude)
            pressure_variation = 30 * np.sin(0.005 * 2 * np.pi * t)  # Slow weather changes
            pressure_rapid = 5 * np.sin(0.02 * 2 * np.pi * t)  # Faster variations
            
            atm_pressure_target = pressure_base + pressure_variation + pressure_rapid
            atmospheric_pressure_dot = 0.1 * (atm_pressure_target - atmospheric_pressure)
            
            # Partial pressure O2 changes with atmospheric pressure
            o2_fraction = 0.209  # 20.9% oxygen
            po2_target = atmospheric_pressure * o2_fraction
            partial_pressure_o2_dot = 0.15 * (po2_target - partial_pressure_o2)
            
            # Respiratory drive response to atmospheric changes
            # Hypoxic response (decreased O2)
            hypoxic_threshold = 100  # mmHg
            hypoxic_stimulus = np.maximum(0, (hypoxic_threshold - partial_pressure_o2) / 100)
            
            # Barometric pressure response
            pressure_stimulus = (atmospheric_pressure - pressure_base) / 100
            
            # Combined atmospheric stimulus
            atmospheric_stimulus = 0.7 * hypoxic_stimulus + 0.3 * pressure_stimulus
            respiratory_drive_dot = 0.5 * (1.0 + atmospheric_stimulus - respiratory_drive)
            
            # Chemoreceptor sensitivity adaptation
            # Acclimatization to atmospheric changes
            base_sensitivity = self.coupling_params['chemoreceptor_sensitivity']
            adaptation_factor = 1 + 0.5 * hypoxic_stimulus  # Increased sensitivity at altitude
            sensitivity_target = base_sensitivity * adaptation_factor
            
            chemoreceptor_sensitivity_dot = 0.02 * (sensitivity_target - chemoreceptor_sensitivity)
            
            # Ventilation response (compensatory)
            ventilation_base = 6000  # mL/min
            hypoxic_ventilation_response = ventilation_base * (1 + 2 * hypoxic_stimulus)
            chemoreceptor_modulation = chemoreceptor_sensitivity * respiratory_drive
            
            ventilation_target = hypoxic_ventilation_response * chemoreceptor_modulation
            ventilation_response_dot = 0.3 * (ventilation_target - ventilation_response)
            
            return [atmospheric_pressure_dot, partial_pressure_o2_dot, 
                   respiratory_drive_dot, chemoreceptor_sensitivity_dot, 
                   ventilation_response_dot]
        
        # Initial conditions (sea level)
        initial_state = [
            760,    # Atmospheric pressure (mmHg)
            159,    # Partial pressure O2 (mmHg) 
            1.0,    # Respiratory drive
            0.8,    # Chemoreceptor sensitivity
            6000    # Ventilation response (mL/min)
        ]
        
        # Solve atmospheric-respiratory system
        solution = odeint(atmospheric_respiratory_system, initial_state, t)
        atmospheric_pressure_trace = solution[:, 0]
        po2_trace = solution[:, 1]
        respiratory_drive_trace = solution[:, 2]
        chemoreceptor_sensitivity_trace = solution[:, 3]
        ventilation_response_trace = solution[:, 4]
        
        # Calculate respiratory performance metrics
        # Oxygen saturation estimation (altitude effects)
        def estimate_spo2(po2):
            """Estimate SpO2 from partial pressure O2"""
            p50 = 27  # mmHg
            hill_coeff = 2.7
            spo2 = 100 * (po2**hill_coeff) / (p50**hill_coeff + po2**hill_coeff)
            return np.clip(spo2, 0, 100)
        
        estimated_spo2 = estimate_spo2(po2_trace)
        
        # Alveolar ventilation efficiency
        dead_space_ratio = 0.3  # VD/VT ratio
        alveolar_ventilation = ventilation_response_trace * (1 - dead_space_ratio)
        
        # Respiratory work (effort required)
        pressure_gradient = 760 - atmospheric_pressure_trace
        respiratory_work = ventilation_response_trace * (1 + pressure_gradient / 1000)
        
        # Atmospheric coupling analysis
        # Pressure-ventilation coupling
        pressure_ventilation_coupling = np.corrcoef(atmospheric_pressure_trace, 
                                                   ventilation_response_trace)[0, 1]
        
        # O2-saturation coupling
        po2_saturation_coupling = np.corrcoef(po2_trace, estimated_spo2)[0, 1]
        
        # Adaptation rate analysis (chemoreceptor sensitivity changes)
        sensitivity_adaptation_rate = np.mean(np.abs(np.gradient(chemoreceptor_sensitivity_trace)))
        
        # Oscillatory response analysis
        atmospheric_signals = {
            'atmospheric_pressure': atmospheric_pressure_trace,
            'partial_pressure_o2': po2_trace,
            'respiratory_drive': respiratory_drive_trace,
            'ventilation_response': ventilation_response_trace,
            'chemoreceptor_sensitivity': chemoreceptor_sensitivity_trace
        }
        
        # Cross-correlation with time lags (response delays)
        def calculate_cross_correlation_with_lag(sig1, sig2, max_lag=50):
            """Calculate cross-correlation with different time lags"""
            correlations = []
            lags = range(-max_lag, max_lag + 1)
            
            for lag in lags:
                if lag == 0:
                    corr = np.corrcoef(sig1, sig2)[0, 1]
                elif lag > 0:
                    if len(sig1) > lag:
                        corr = np.corrcoef(sig1[lag:], sig2[:-lag])[0, 1]
                    else:
                        corr = 0
                else:
                    if len(sig2) > abs(lag):
                        corr = np.corrcoef(sig1[:lag], sig2[abs(lag):])[0, 1]
                    else:
                        corr = 0
                correlations.append(corr)
            
            max_corr_idx = np.argmax(np.abs(correlations))
            optimal_lag = lags[max_corr_idx]
            max_correlation = correlations[max_corr_idx]
            
            return optimal_lag, max_correlation, correlations, lags
        
        # Atmospheric-respiratory response delay
        pressure_lag, pressure_max_corr, _, _ = calculate_cross_correlation_with_lag(
            atmospheric_pressure_trace, ventilation_response_trace)
        
        response_delay = abs(pressure_lag) / fs  # Convert to seconds
        
        # Environmental adaptation metrics
        # Altitude simulation (pressure drop)
        altitude_periods = t > duration / 2  # Second half simulates altitude
        sea_level_spo2 = np.mean(estimated_spo2[~altitude_periods])
        altitude_spo2 = np.mean(estimated_spo2[altitude_periods])
        altitude_spo2_drop = sea_level_spo2 - altitude_spo2
        
        # Ventilation compensation
        sea_level_ventilation = np.mean(ventilation_response_trace[~altitude_periods])
        altitude_ventilation = np.mean(ventilation_response_trace[altitude_periods])
        ventilation_compensation = (altitude_ventilation - sea_level_ventilation) / sea_level_ventilation * 100
        
        # System stability under atmospheric stress
        pressure_variability = np.std(atmospheric_pressure_trace)
        ventilation_stability = 1.0 / (1.0 + np.std(ventilation_response_trace) / np.mean(ventilation_response_trace))
        
        # Theoretical predictions
        theoretical_predictions = {
            'pressure_ventilation_coupling_range': (-0.8, -0.2),  # Inverse: low pressure → high ventilation
            'po2_saturation_coupling_min': 0.8,
            'response_delay_max': 30,  # seconds
            'altitude_spo2_drop_range': (2, 15),  # % drop
            'ventilation_compensation_min': 20,  # % increase
            'sensitivity_adaptation_rate_min': 0.001,  # adaptation per time unit
            'ventilation_stability_min': 0.7
        }
        
        # Validation
        validation_success = (
            theoretical_predictions['pressure_ventilation_coupling_range'][0] <= 
            pressure_ventilation_coupling <= 
            theoretical_predictions['pressure_ventilation_coupling_range'][1] and
            po2_saturation_coupling >= theoretical_predictions['po2_saturation_coupling_min'] and
            response_delay <= theoretical_predictions['response_delay_max'] and
            theoretical_predictions['altitude_spo2_drop_range'][0] <= altitude_spo2_drop <= 
            theoretical_predictions['altitude_spo2_drop_range'][1] and
            ventilation_compensation >= theoretical_predictions['ventilation_compensation_min'] and
            sensitivity_adaptation_rate >= theoretical_predictions['sensitivity_adaptation_rate_min'] and
            ventilation_stability >= theoretical_predictions['ventilation_stability_min']
        )
        
        results = {
            'experiment': 'Atmospheric-Respiratory Integration',
            'validation_success': validation_success,
            'atmospheric_traces': {
                'time': t,
                'atmospheric_pressure': atmospheric_pressure_trace,
                'partial_pressure_o2': po2_trace,
                'respiratory_drive': respiratory_drive_trace,
                'chemoreceptor_sensitivity': chemoreceptor_sensitivity_trace,
                'ventilation_response': ventilation_response_trace
            },
            'performance_metrics': {
                'estimated_spo2': estimated_spo2,
                'alveolar_ventilation': alveolar_ventilation,
                'respiratory_work': respiratory_work
            },
            'coupling_analysis': {
                'pressure_ventilation_coupling': pressure_ventilation_coupling,
                'po2_saturation_coupling': po2_saturation_coupling,
                'response_delay_seconds': response_delay,
                'pressure_max_correlation': pressure_max_corr
            },
            'adaptation_metrics': {
                'altitude_spo2_drop': altitude_spo2_drop,
                'ventilation_compensation': ventilation_compensation,
                'sensitivity_adaptation_rate': sensitivity_adaptation_rate,
                'ventilation_stability': ventilation_stability
            },
            'environmental_conditions': {
                'sea_level_spo2': sea_level_spo2,
                'altitude_spo2': altitude_spo2,
                'sea_level_ventilation': sea_level_ventilation,
                'altitude_ventilation': altitude_ventilation,
                'pressure_variability': pressure_variability
            },
            'theoretical_predictions': theoretical_predictions
        }
        
        # Visualization
        self._plot_atmospheric_integration_analysis(results)
        
        return results
    
    def _plot_breathing_rhythm_analysis(self, results: Dict[str, Any]):
        """Create breathing rhythm analysis visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        control_traces = results['control_traces']
        breathing = results['breathing_pattern']
        coupling = results['coupling_analysis']
        freq_data = results['frequency_analysis']
        metrics = results['respiratory_metrics']
        
        # Plot 1: Respiratory control centers
        axes[0, 0].plot(control_traces['time'], control_traces['pre_botzinger'], 
                       'red', label='Pre-Bötzinger', linewidth=2)
        axes[0, 0].plot(control_traces['time'], control_traces['ponto_medullary'], 
                       'blue', label='Ponto-medullary', linewidth=2)
        axes[0, 0].plot(control_traces['time'], control_traces['vagal'], 
                       'green', label='Vagal', linewidth=2)
        axes[0, 0].plot(control_traces['time'], control_traces['chemical_drive'], 
                       'purple', label='Chemical Drive', linewidth=2)
        
        axes[0, 0].set_title('Respiratory Control Centers')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Activity Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Respiratory drive and breathing pattern
        time_window = slice(0, int(60 * 20))  # First 60 seconds
        
        ax1 = axes[0, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(control_traces['time'][time_window], 
                        control_traces['respiratory_drive'][time_window], 
                        'red', linewidth=2, label='Respiratory Drive')
        line2 = ax2.plot(control_traces['time'][time_window], 
                        breathing['breathing_signal'][time_window], 
                        'blue', linewidth=1, label='Breathing Signal')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Respiratory Drive', color='red')
        ax2.set_ylabel('Breathing Signal', color='blue')
        ax1.tick_params(axis='y', labelcolor='red')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.set_title('Respiratory Drive and Breathing Pattern')
        ax1.grid(True, alpha=0.3)
        
        # Plot 3: Airflow and lung volume
        ax3 = axes[0, 2]
        ax4 = ax3.twinx()
        
        line3 = ax3.plot(control_traces['time'][time_window], 
                        breathing['airflow'][time_window], 
                        'green', linewidth=1.5, label='Airflow')
        line4 = ax4.plot(control_traces['time'][time_window], 
                        breathing['lung_volume'][time_window], 
                        'orange', linewidth=2, label='Lung Volume')
        
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Airflow (mL/s)', color='green')
        ax4.set_ylabel('Lung Volume (mL)', color='orange')
        ax3.tick_params(axis='y', labelcolor='green')
        ax4.tick_params(axis='y', labelcolor='orange')
        
        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
        ax3.set_title('Airflow and Lung Volume')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Control system coupling matrix
        im = axes[1, 0].imshow(coupling['coupling_matrix'], cmap='RdBu_r', 
                              aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(coupling['signal_names'])))
        axes[1, 0].set_yticks(range(len(coupling['signal_names'])))
        axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in coupling['signal_names']], 
                                  rotation=45)
        axes[1, 0].set_yticklabels([name.replace('_', '\n') for name in coupling['signal_names']])
        axes[1, 0].set_title('Control System Coupling')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 5: Frequency domain analysis
        axes[1, 1].semilogy(freq_data['frequencies'], freq_data['power_spectral_density'], 
                           'b-', linewidth=2)
        
        # Highlight control frequency bands
        band_colors = {'primary_rhythm': 'red', 'modulation': 'green', 
                      'chemical_drive': 'orange', 'vagal_coupling': 'purple'}
        
        control_bands = {
            'primary_rhythm': (0.2, 0.35),
            'modulation': (0.05, 0.15),
            'chemical_drive': (0.01, 0.05),
            'vagal_coupling': (0.8, 1.5),
        }
        
        for band_name, (f_min, f_max) in control_bands.items():
            if f_max <= max(freq_data['frequencies']):
                band_mask = ((freq_data['frequencies'] >= f_min) & 
                           (freq_data['frequencies'] <= f_max))
                if np.any(band_mask):
                    axes[1, 1].fill_between(freq_data['frequencies'][band_mask],
                                           freq_data['power_spectral_density'][band_mask],
                                           alpha=0.3, color=band_colors.get(band_name, 'gray'),
                                           label=band_name.replace('_', ' ').title())
        
        axes[1, 1].set_title('Respiratory Control Power Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('PSD')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 2)
        
        # Plot 6: Control band powers
        band_powers = freq_data['control_band_powers']
        band_names = list(band_powers.keys())
        band_values = list(band_powers.values())
        colors = [band_colors.get(name, 'gray') for name in band_names]
        
        bars = axes[1, 2].bar(band_names, band_values, color=colors, alpha=0.7)
        axes[1, 2].set_title('Control Band Powers')
        axes[1, 2].set_ylabel('Power (%)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, band_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 7: Respiratory metrics
        metric_names = ['Breathing Rate\n(breaths/min)', 'Tidal Volume\n(mL)', 
                       'Breathing Variability\n(s)', 'TV Variability\n(mL)']
        metric_values = [metrics['breathing_rate'], metrics['mean_tidal_volume'],
                        metrics['breathing_variability'], metrics['tidal_volume_variability']]
        
        bars = axes[2, 0].bar(metric_names, metric_values, 
                             color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[2, 0].set_title('Respiratory Performance Metrics')
        axes[2, 0].set_ylabel('Metric Value')
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[2, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # Plot 8: Cardiac-respiratory coupling
        # Generate synthetic cardiac signal for visualization
        cardiac_freq = metrics['breathing_rate'] / 60 * 4  # 4:1 ratio
        cardiac_time = control_traces['time'][:1000]  # First portion for clarity
        cardiac_signal = np.sin(cardiac_freq * 2 * np.pi * cardiac_time)
        resp_signal = control_traces['respiratory_drive'][:1000]
        
        axes[2, 1].plot(cardiac_time, cardiac_signal, 'red', 
                       label='Cardiac (synthetic)', linewidth=2)
        axes[2, 1].plot(cardiac_time, resp_signal, 'blue', 
                       label='Respiratory Drive', linewidth=2)
        
        axes[2, 1].set_title(f'Cardiac-Respiratory Coupling (r={coupling["cardiac_respiratory_coupling"]:.3f})')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Signal Amplitude')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # Plot 9: Validation summary
        predictions = results['theoretical_predictions']
        validation_text = (
            f"🫁 BREATHING RHYTHM VALIDATION\n\n"
            f"✓ Status: {'PASS' if results['validation_success'] else 'FAIL'}\n\n"
            f"Breathing Rate: {metrics['breathing_rate']:.1f} bpm\n"
            f"Expected: {predictions['expected_breathing_rate']}\n\n"
            f"Tidal Volume: {metrics['mean_tidal_volume']:.0f} mL\n"
            f"Expected: {predictions['expected_tidal_volume']}\n\n"
            f"Primary Rhythm Power: {band_powers['primary_rhythm']:.1f}%\n"
            f"Required: ≥{predictions['primary_rhythm_power_min']}%\n\n"
            f"Cardiac-Resp Coupling: {coupling['cardiac_respiratory_coupling']:.3f}\n"
            f"Required: ≥{predictions['cardiac_respiratory_coupling_min']}"
        )
        
        axes[2, 2].text(0.05, 0.95, validation_text, transform=axes[2, 2].transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', 
                               facecolor='lightgreen' if results['validation_success'] else 'lightcoral', 
                               alpha=0.8))
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('Validation Results')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/breathing_rhythm_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gas_exchange_analysis(self, results: Dict[str, Any]):
        """Create gas exchange dynamics analysis visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        gas_traces = results['gas_traces']
        derived = results['derived_metrics']
        oscillation = results['oscillation_analysis']
        coupling = results['coupling_analysis']
        
        # Plot 1: Gas concentration traces
        axes[0, 0].plot(gas_traces['time'], gas_traces['alveolar_o2'], 
                       'red', label='Alveolar O₂', linewidth=2)
        axes[0, 0].plot(gas_traces['time'], gas_traces['blood_o2'], 
                       'blue', label='Blood O₂', linewidth=2)
        axes[0, 0].set_title('Oxygen Concentrations')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('O₂ Concentration (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: CO2 concentrations
        axes[0, 1].plot(gas_traces['time'], gas_traces['alveolar_co2'], 
                       'green', label='Alveolar CO₂', linewidth=2)
        axes[0, 1].plot(gas_traces['time'], gas_traces['blood_co2'], 
                       'orange', label='Blood CO₂', linewidth=2)
        axes[0, 1].set_title('Carbon Dioxide Concentrations')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('CO₂ Concentration (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Hemoglobin saturation and tissue O2
        ax1 = axes[0, 2]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(gas_traces['time'], gas_traces['hb_saturation'], 
                        'purple', linewidth=2, label='Hb Saturation')
        line2 = ax2.plot(gas_traces['time'], gas_traces['tissue_o2'], 
                        'brown', linewidth=2, label='Tissue O₂')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Hb Saturation (%)', color='purple')
        ax2.set_ylabel('Tissue O₂ (%)', color='brown')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='brown')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.set_title('Hemoglobin & Tissue Oxygen')
        ax1.grid(True, alpha=0.3)
        
        # Plot 4: Gas coupling matrix heatmap
        im = axes[1, 0].imshow(coupling['gas_coupling_matrix'], cmap='RdBu_r', 
                              aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(coupling['gas_names'])))
        axes[1, 0].set_yticks(range(len(coupling['gas_names'])))
        axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in coupling['gas_names']], 
                                  rotation=45)
        axes[1, 0].set_yticklabels([name.replace('_', '\n') for name in coupling['gas_names']])
        axes[1, 0].set_title('Gas Exchange Coupling Matrix')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 5: Gas oscillation powers
        osc_powers = oscillation['gas_oscillation_powers']
        gas_names = list(osc_powers.keys())
        power_values = list(osc_powers.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(gas_names)))
        
        bars = axes[1, 1].bar(gas_names, power_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Respiratory Oscillation Powers')
        axes[1, 1].set_ylabel('Power in Breathing Band (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, power_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Plot 6: O2-CO2 relationship
        axes[1, 2].scatter(gas_traces['alveolar_o2'], gas_traces['alveolar_co2'], 
                          alpha=0.6, s=20, c=gas_traces['time'], cmap='viridis')
        
        # Add trend line
        z = np.polyfit(gas_traces['alveolar_o2'], gas_traces['alveolar_co2'], 1)
        p = np.poly1d(z)
        axes[1, 2].plot(gas_traces['alveolar_o2'], p(gas_traces['alveolar_o2']), 
                       "r--", alpha=0.8, linewidth=2)
        
        axes[1, 2].set_title(f'O₂-CO₂ Coupling (r={coupling["o2_co2_coupling"]:.3f})')
        axes[1, 2].set_xlabel('Alveolar O₂ (%)')
        axes[1, 2].set_ylabel('Alveolar CO₂ (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 7: Derived gas exchange metrics
        axes[2, 0].plot(gas_traces['time'], derived['aa_gradient'], 
                       'red', linewidth=2, label='A-a Gradient')
        axes[2, 0].fill_between(gas_traces['time'], derived['aa_gradient'], 
                               alpha=0.3, color='red')
        axes[2, 0].set_title('Alveolar-Arterial O₂ Gradient')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('A-a Gradient (%)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: Respiratory exchange ratio and efficiency
        ax1 = axes[2, 1]
        ax2 = ax1.twinx()
        
        # Filter extreme RER values for visualization
        rer_filtered = np.clip(derived['respiratory_exchange_ratio'], 0.5, 2.0)
        
        line1 = ax1.plot(gas_traces['time'], rer_filtered, 
                        'blue', linewidth=2, label='RER')
        line2 = ax2.plot(gas_traces['time'], derived['gas_exchange_efficiency'], 
                        'green', linewidth=2, label='Exchange Efficiency')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RER', color='blue')
        ax2.set_ylabel('Exchange Efficiency', color='green')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='green')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.set_title('Gas Exchange Performance')
        ax1.grid(True, alpha=0.3)
        
        # Plot 9: Validation summary
        predictions = results['theoretical_predictions']
        validation_text = (
            f"🔄 GAS EXCHANGE VALIDATION\n\n"
            f"✓ Status: {'PASS' if results['validation_success'] else 'FAIL'}\n\n"
            f"O₂-CO₂ Coupling: {coupling['o2_co2_coupling']:.3f}\n"
            f"Expected: {predictions['o2_co2_anticorrelation_range']}\n\n"
            f"Hb-Tissue Coupling: {coupling['hb_tissue_coupling']:.3f}\n"
            f"Required: ≥{predictions['hb_tissue_coupling_min']}\n\n"
            f"Phase Coupling: {coupling['alveolar_blood_phase_coupling']:.3f}\n"
            f"Required: ≥{predictions['alveolar_blood_phase_coupling_min']}\n\n"
            f"O₂ Variability: {oscillation['o2_variability']:.2f}%\n"
            f"Max Allowed: {predictions['o2_variability_max']}%"
        )
        
        axes[2, 2].text(0.05, 0.95, validation_text, transform=axes[2, 2].transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', 
                               facecolor='lightgreen' if results['validation_success'] else 'lightcoral', 
                               alpha=0.8))
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('Validation Results')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/gas_exchange_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_atmospheric_integration_analysis(self, results: Dict[str, Any]):
        """Create atmospheric-respiratory integration analysis visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        atm_traces = results['atmospheric_traces']
        performance = results['performance_metrics']
        coupling = results['coupling_analysis']
        adaptation = results['adaptation_metrics']
        environment = results['environmental_conditions']
        
        # Plot 1: Atmospheric conditions
        ax1 = axes[0, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(atm_traces['time'], atm_traces['atmospheric_pressure'], 
                        'blue', linewidth=2, label='Pressure')
        line2 = ax2.plot(atm_traces['time'], atm_traces['partial_pressure_o2'], 
                        'red', linewidth=2, label='PO₂')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Atmospheric Pressure (mmHg)', color='blue')
        ax2.set_ylabel('Partial Pressure O₂ (mmHg)', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax1.set_title('Atmospheric Conditions')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Respiratory response traces
        axes[0, 1].plot(atm_traces['time'], atm_traces['respiratory_drive'], 
                       'green', label='Respiratory Drive', linewidth=2)
        axes[0, 1].plot(atm_traces['time'], atm_traces['chemoreceptor_sensitivity'], 
                       'orange', label='Chemoreceptor Sensitivity', linewidth=2)
        
        axes[0, 1].set_title('Respiratory Control Response')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Response Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Ventilation response and SpO2
        ax1 = axes[0, 2]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(atm_traces['time'], atm_traces['ventilation_response'], 
                        'purple', linewidth=2, label='Ventilation')
        line2 = ax2.plot(atm_traces['time'], performance['estimated_spo2'], 
                        'brown', linewidth=2, label='SpO₂')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Ventilation (mL/min)', color='purple')
        ax2.set_ylabel('SpO₂ (%)', color='brown')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='brown')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        ax1.set_title('Ventilation Response & Oxygen Saturation')
        ax1.grid(True, alpha=0.3)
        
        # Plot 4: Pressure-ventilation coupling
        axes[1, 0].scatter(atm_traces['atmospheric_pressure'], 
                          atm_traces['ventilation_response'], 
                          alpha=0.6, s=20, c=atm_traces['time'], cmap='plasma')
        
        # Add trend line
        z = np.polyfit(atm_traces['atmospheric_pressure'], atm_traces['ventilation_response'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(atm_traces['atmospheric_pressure'], 
                       p(atm_traces['atmospheric_pressure']), 
                       "r--", alpha=0.8, linewidth=2)
        
        axes[1, 0].set_title(f'Pressure-Ventilation Coupling (r={coupling["pressure_ventilation_coupling"]:.3f})')
        axes[1, 0].set_xlabel('Atmospheric Pressure (mmHg)')
        axes[1, 0].set_ylabel('Ventilation Response (mL/min)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: PO2-SpO2 relationship
        axes[1, 1].scatter(atm_traces['partial_pressure_o2'], 
                          performance['estimated_spo2'], 
                          alpha=0.6, s=20, c=atm_traces['time'], cmap='viridis')
        
        # Add theoretical oxygen dissociation curve
        po2_theory = np.linspace(50, 200, 100)
        p50 = 27
        hill_coeff = 2.7
        spo2_theory = 100 * (po2_theory**hill_coeff) / (p50**hill_coeff + po2_theory**hill_coeff)
        axes[1, 1].plot(po2_theory, spo2_theory, 'k--', linewidth=2, 
                       label='Theoretical ODC', alpha=0.8)
        
        axes[1, 1].set_title(f'PO₂-SpO₂ Coupling (r={coupling["po2_saturation_coupling"]:.3f})')
        axes[1, 1].set_xlabel('Partial Pressure O₂ (mmHg)')
        axes[1, 1].set_ylabel('SpO₂ (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Adaptation metrics over time
        midpoint = len(atm_traces['time']) // 2
        sea_level_period = atm_traces['time'] < atm_traces['time'][midpoint]
        altitude_period = ~sea_level_period
        
        metrics = ['SpO₂', 'Ventilation', 'Respiratory Drive', 'Sensitivity']
        sea_level_values = [
            np.mean(performance['estimated_spo2'][sea_level_period]),
            np.mean(atm_traces['ventilation_response'][sea_level_period]) / 1000,
            np.mean(atm_traces['respiratory_drive'][sea_level_period]),
            np.mean(atm_traces['chemoreceptor_sensitivity'][sea_level_period])
        ]
        altitude_values = [
            np.mean(performance['estimated_spo2'][altitude_period]),
            np.mean(atm_traces['ventilation_response'][altitude_period]) / 1000,
            np.mean(atm_traces['respiratory_drive'][altitude_period]),
            np.mean(atm_traces['chemoreceptor_sensitivity'][altitude_period])
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axes[1, 2].bar(x - width/2, sea_level_values, width, 
                              label='Sea Level', color='lightblue', alpha=0.7)
        bars2 = axes[1, 2].bar(x + width/2, altitude_values, width, 
                              label='Altitude', color='lightcoral', alpha=0.7)
        
        axes[1, 2].set_title('Sea Level vs Altitude Adaptation')
        axes[1, 2].set_ylabel('Normalized Values')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 7: Respiratory work over time
        axes[2, 0].plot(atm_traces['time'], performance['respiratory_work'], 
                       'red', linewidth=2)
        axes[2, 0].fill_between(atm_traces['time'], performance['respiratory_work'], 
                               alpha=0.3, color='red')
        axes[2, 0].axvline(atm_traces['time'][midpoint], color='black', 
                          linestyle='--', alpha=0.5, label='Altitude Transition')
        axes[2, 0].set_title('Respiratory Work')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Work (arbitrary units)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: System adaptation summary
        adaptation_names = ['SpO₂ Drop\n(%)', 'Ventilation\nIncrease (%)', 
                           'Sensitivity\nAdaptation', 'Response Delay\n(s)']
        adaptation_values = [adaptation['altitude_spo2_drop'], 
                            adaptation['ventilation_compensation'],
                            adaptation['sensitivity_adaptation_rate'] * 1000,  # Scale for visibility
                            coupling['response_delay_seconds']]
        
        colors = ['lightcoral', 'lightgreen', 'gold', 'lightblue']
        bars = axes[2, 1].bar(adaptation_names, adaptation_values, color=colors, alpha=0.7)
        axes[2, 1].set_title('Atmospheric Adaptation Metrics')
        axes[2, 1].set_ylabel('Metric Value')
        
        for bar, value in zip(bars, adaptation_values):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Plot 9: Validation summary
        predictions = results['theoretical_predictions']
        validation_text = (
            f"🌍 ATMOSPHERIC INTEGRATION VALIDATION\n\n"
            f"✓ Status: {'PASS' if results['validation_success'] else 'FAIL'}\n\n"
            f"Pressure-Ventilation: {coupling['pressure_ventilation_coupling']:.3f}\n"
            f"Expected: {predictions['pressure_ventilation_coupling_range']}\n\n"
            f"Response Delay: {coupling['response_delay_seconds']:.1f}s\n"
            f"Max Allowed: {predictions['response_delay_max']}s\n\n"
            f"Altitude SpO₂ Drop: {adaptation['altitude_spo2_drop']:.1f}%\n"
            f"Expected: {predictions['altitude_spo2_drop_range']}\n\n"
            f"Ventilation Compensation: {adaptation['ventilation_compensation']:.1f}%\n"
            f"Required: ≥{predictions['ventilation_compensation_min']}%"
        )
        
        axes[2, 2].text(0.05, 0.95, validation_text, transform=axes[2, 2].transAxes,
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', 
                               facecolor='lightgreen' if results['validation_success'] else 'lightcoral', 
                               alpha=0.8))
        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].axis('off')
        axes[2, 2].set_title('Validation Results')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/atmospheric_integration_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all respiratory oscillatory validation experiments"""
        print("🫁 Running Respiratory System Oscillatory Validation Suite")
        print("=" * 60)
        
        all_results = {}
        
        # Run experiments
        all_results['experiment_1'] = self.experiment_1_breathing_rhythm_oscillatory_coupling()
        all_results['experiment_2'] = self.experiment_2_gas_exchange_oscillation_dynamics()
        all_results['experiment_3'] = self.experiment_3_atmospheric_respiratory_integration()
        
        # Compile validation summary
        validations = [result['validation_success'] for result in all_results.values()]
        overall_success = all(validations)
        
        summary = {
            'domain': 'Respiratory System Oscillations',
            'total_experiments': len(all_results),
            'successful_validations': sum(validations),
            'overall_validation_success': overall_success,
            'timestamp': datetime.now().isoformat(),
            'detailed_results': all_results
        }
        
        # Save comprehensive results
        results_file = f"{self.results_dir}/respiratory_validation_summary.json"
        import json
        with open(results_file, 'w') as f:
            json.dump({k: v for k, v in summary.items() 
                      if k != 'detailed_results'}, f, indent=2)
        
        print(f"\n🫁 Respiratory Validation Complete:")
        print(f"   ✓ Successful: {sum(validations)}/{len(validations)} experiments")
        print(f"   ✓ Overall Status: {'PASS' if overall_success else 'FAIL'}")
        print(f"   ✓ Results saved to: {self.results_dir}/")
        
        self.validation_results = summary
        return summary

def main():
    """Run respiratory oscillatory validation as standalone"""
    validator = RespiratoryOscillatoryValidator()
    return validator.run_all_experiments()

if __name__ == "__main__":
    results = main()
    print(f"Respiratory validation completed: {results['overall_validation_success']}")
