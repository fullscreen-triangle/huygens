"""
Technological Infrastructure Oscillatory Model
Models satellite/cell tower/device oscillatory infrastructure coupling with biology
Part of the Technological-Biological Meta-Oscillatory Coupling Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnologicalOscillatoryInfrastructure:
    """
    Models technological infrastructure oscillatory systems and their coupling with biology
    """
    
    def __init__(self):
        # Technological oscillatory frequencies (Hz)
        self.tech_frequencies = {
            'gps_l1': 1.57542e9,           # GPS L1 carrier frequency
            'gps_l2': 1.22760e9,           # GPS L2 carrier frequency
            'cesium_atomic': 9.192631770e9, # Cesium atomic clock standard
            'rubidium_atomic': 6.834682611e9, # Rubidium atomic clock
            'cellular_850': 850e6,          # Cellular 850 MHz band
            'cellular_1900': 1.9e9,        # Cellular 1900 MHz band
            'cellular_2100': 2.1e9,        # Cellular 2100 MHz band
            'cellular_2600': 2.6e9,        # Cellular 2600 MHz band
            'wifi_2_4': 2.4e9,             # WiFi 2.4 GHz
            'wifi_5_0': 5.0e9,             # WiFi 5.0 GHz
            'bluetooth': 2.4e9,            # Bluetooth 2.4 GHz
            'device_crystal': 32.768e3,    # Device RTC crystal oscillator
            'cpu_base': 1e9,               # Base CPU frequency (~1 GHz)
            'satellite_orbit': 1/(12*3600) # GPS satellite orbital frequency
        }
        
        # Biological oscillatory frequencies (Hz)
        self.bio_frequencies = {
            'circadian': 1.157e-5,         # 24 hour circadian rhythm
            'ultradian': 1.667e-4,         # 1.5 hour ultradian rhythm
            'cardiac_rest': 1.0,           # Resting heart rate ~60 BPM
            'cardiac_active': 1.67,        # Active heart rate ~100 BPM
            'respiratory': 0.25,           # Breathing rate ~15 BPM
            'neural_delta': 2.0,           # Delta brain waves (1-4 Hz)
            'neural_theta': 6.0,           # Theta brain waves (4-8 Hz)
            'neural_alpha': 10.5,          # Alpha brain waves (8-13 Hz)
            'neural_beta': 21.5,           # Beta brain waves (13-30 Hz)
            'neural_gamma': 65.0,          # Gamma brain waves (30-100 Hz)
            'weekly': 1.653e-6,            # Weekly rhythm (7 days)
            'monthly': 3.86e-7             # Monthly rhythm (30 days)
        }
        
        # Infrastructure parameters
        self.infrastructure_params = {
            'gps_satellites': 31,          # Number of GPS satellites
            'cellular_towers_per_km2': 1.5, # Cellular tower density
            'device_oscillator_stability': 20e-6, # Crystal oscillator stability (ppm)
            'atomic_clock_stability': 1e-13, # Atomic clock fractional stability
            'satellite_altitude_km': 20200,   # GPS satellite altitude
            'cellular_power_watts': 20,       # Typical cellular tower power
            'device_power_milliwatts': 100    # Typical device power consumption
        }
        
        self.coupling_coefficients = {}
        self._calculate_coupling_coefficients()
    
    def _calculate_coupling_coefficients(self):
        """Calculate coupling coefficients between technological and biological systems"""
        # Harmonic coupling coefficients
        for tech_name, tech_freq in self.tech_frequencies.items():
            for bio_name, bio_freq in self.bio_frequencies.items():
                # Calculate harmonic ratio
                ratio = tech_freq / bio_freq
                
                # Coupling strength based on harmonic relationships
                # Stronger coupling for simple integer ratios
                if ratio > 1:
                    harmonic_strength = 1 / (1 + np.log10(ratio))
                else:
                    harmonic_strength = 1 / (1 + np.log10(1/ratio))
                
                coupling_key = f"{tech_name}_{bio_name}"
                self.coupling_coefficients[coupling_key] = {
                    'harmonic_ratio': ratio,
                    'coupling_strength': harmonic_strength,
                    'frequency_difference': abs(tech_freq - bio_freq),
                    'normalized_diff': abs(tech_freq - bio_freq) / max(tech_freq, bio_freq)
                }
    
    def model_gps_satellite_system(self, time_hours: float = 24, sampling_rate: float = 1.0) -> Dict:
        """Model GPS satellite system oscillatory behavior"""
        t_seconds = np.linspace(0, time_hours * 3600, int(time_hours * 3600 * sampling_rate))
        
        # GPS satellite constellation
        n_satellites = self.infrastructure_params['gps_satellites']
        orbital_freq = self.tech_frequencies['satellite_orbit']
        l1_freq = self.tech_frequencies['gps_l1']
        atomic_freq = self.tech_frequencies['cesium_atomic']
        
        # Model satellite orbital dynamics
        satellite_positions = []
        satellite_signals = []
        
        for sat_id in range(n_satellites):
            # Each satellite has slightly different orbital phase
            orbital_phase = 2 * np.pi * sat_id / n_satellites
            
            # Orbital position (simplified circular orbit)
            orbital_signal = np.sin(2 * np.pi * orbital_freq * t_seconds + orbital_phase)
            satellite_positions.append(orbital_signal)
            
            # L1 carrier signal modulated by atomic clock stability
            atomic_stability = self.infrastructure_params['atomic_clock_stability']
            clock_drift = atomic_stability * np.cumsum(np.random.normal(0, 1, len(t_seconds)))
            
            # High-frequency L1 carrier (downsampled representation)
            l1_phase = 2 * np.pi * orbital_freq * t_seconds + orbital_phase  # Use orbital freq as proxy
            l1_signal = np.sin(l1_phase) * (1 + clock_drift)
            satellite_signals.append(l1_signal)
        
        # Calculate constellation coherence
        constellation_coherence = np.mean([np.corrcoef(satellite_positions[0], pos)[0,1] 
                                         for pos in satellite_positions[1:]])
        
        # Calculate total GPS signal power
        total_signal = np.sum(satellite_signals, axis=0)
        signal_power = np.mean(total_signal**2)
        
        return {
            'time_seconds': t_seconds,
            'satellite_positions': np.array(satellite_positions),
            'satellite_signals': np.array(satellite_signals),
            'total_signal': total_signal,
            'constellation_coherence': constellation_coherence,
            'signal_power': signal_power,
            'orbital_frequency': orbital_freq,
            'l1_frequency': l1_freq,
            'n_satellites': n_satellites
        }
    
    def model_cellular_network(self, time_hours: float = 24, sampling_rate: float = 1.0,
                              coverage_area_km2: float = 100) -> Dict:
        """Model cellular network oscillatory infrastructure"""
        t_seconds = np.linspace(0, time_hours * 3600, int(time_hours * 3600 * sampling_rate))
        
        # Calculate number of towers in coverage area
        tower_density = self.infrastructure_params['cellular_towers_per_km2']
        n_towers = int(coverage_area_km2 * tower_density)
        
        # Multiple frequency bands
        freq_bands = ['cellular_850', 'cellular_1900', 'cellular_2100', 'cellular_2600']
        tower_signals = {}
        
        for band in freq_bands:
            band_freq = self.tech_frequencies[band]
            band_signals = []
            
            for tower_id in range(n_towers):
                # Each tower has slightly different phase and amplitude
                phase = 2 * np.pi * np.random.random()
                amplitude = 1 + 0.1 * np.random.normal()  # ±10% amplitude variation
                
                # Use lower frequency oscillation as proxy for high RF
                proxy_freq = band_freq / 1e9  # Scale down for computation
                signal = amplitude * np.sin(2 * np.pi * proxy_freq * t_seconds + phase)
                
                # Add power variation due to usage patterns
                usage_pattern = 0.5 + 0.3 * np.sin(2 * np.pi * self.bio_frequencies['circadian'] * t_seconds)
                signal *= usage_pattern
                
                band_signals.append(signal)
            
            tower_signals[band] = np.array(band_signals)
        
        # Calculate total cellular network power
        total_network_signal = np.zeros(len(t_seconds))
        for band_signals in tower_signals.values():
            total_network_signal += np.sum(band_signals, axis=0)
        
        # Calculate network coherence across bands
        band_powers = [np.mean(signals**2) for signals in tower_signals.values()]
        network_coherence = 1 / (1 + np.var(band_powers) / np.mean(band_powers)**2)
        
        return {
            'time_seconds': t_seconds,
            'tower_signals': tower_signals,
            'total_network_signal': total_network_signal,
            'network_coherence': network_coherence,
            'n_towers': n_towers,
            'frequency_bands': freq_bands,
            'band_powers': band_powers
        }
    
    def model_smart_device_ensemble(self, time_hours: float = 24, sampling_rate: float = 1.0,
                                  n_devices: int = 100) -> Dict:
        """Model ensemble of smart device oscillators"""
        t_seconds = np.linspace(0, time_hours * 3600, int(time_hours * 3600 * sampling_rate))
        
        # Device oscillator types
        oscillator_types = ['device_crystal', 'cpu_base', 'wifi_2_4', 'wifi_5_0', 'bluetooth']
        device_signals = {}
        
        for osc_type in oscillator_types:
            osc_freq = self.tech_frequencies[osc_type]
            type_signals = []
            
            for device_id in range(n_devices):
                # Each device has crystal oscillator tolerance
                stability_ppm = self.infrastructure_params['device_oscillator_stability']
                freq_variation = 1 + stability_ppm * (2 * np.random.random() - 1)
                actual_freq = osc_freq * freq_variation
                
                # Use scaled frequency for computation
                if actual_freq > 1e6:  # Scale down high frequencies
                    proxy_freq = actual_freq / 1e6
                else:
                    proxy_freq = actual_freq
                
                # Generate oscillator signal
                phase = 2 * np.pi * np.random.random()
                signal = np.sin(2 * np.pi * proxy_freq * t_seconds + phase)
                
                # Add device usage patterns (correlated with circadian rhythms)
                if osc_type in ['wifi_2_4', 'wifi_5_0', 'bluetooth']:
                    usage_pattern = 0.3 + 0.7 * (0.5 + 0.4 * np.sin(2 * np.pi * self.bio_frequencies['circadian'] * t_seconds + np.pi/6))
                    signal *= usage_pattern
                
                type_signals.append(signal)
            
            device_signals[osc_type] = np.array(type_signals)
        
        # Calculate device ensemble coherence
        ensemble_coherence = {}
        for osc_type, signals in device_signals.items():
            # Coherence within oscillator type
            correlations = []
            for i in range(min(10, len(signals))):  # Sample 10 devices
                for j in range(i+1, min(10, len(signals))):
                    corr = np.corrcoef(signals[i], signals[j])[0,1]
                    correlations.append(abs(corr))
            ensemble_coherence[osc_type] = np.mean(correlations) if correlations else 0
        
        # Calculate total device oscillatory power
        total_device_signal = np.zeros(len(t_seconds))
        for osc_signals in device_signals.values():
            total_device_signal += np.sum(osc_signals, axis=0)
        
        return {
            'time_seconds': t_seconds,
            'device_signals': device_signals,
            'total_device_signal': total_device_signal,
            'ensemble_coherence': ensemble_coherence,
            'n_devices': n_devices,
            'oscillator_types': oscillator_types
        }
    
    def calculate_biological_coupling(self, gps_model: Dict, cellular_model: Dict, 
                                    device_model: Dict, bio_signal_type: str = 'circadian') -> Dict:
        """Calculate coupling between technological infrastructure and biological rhythms"""
        
        if bio_signal_type not in self.bio_frequencies:
            raise ValueError(f"Unknown biological signal type: {bio_signal_type}")
        
        bio_freq = self.bio_frequencies[bio_signal_type]
        time_seconds = gps_model['time_seconds']
        
        # Generate biological reference signal
        bio_signal = np.sin(2 * np.pi * bio_freq * time_seconds)
        
        coupling_results = {}
        
        # GPS-biological coupling
        gps_signal = gps_model['total_signal']
        gps_bio_correlation = np.corrcoef(gps_signal, bio_signal)[0,1]
        
        # Cross-correlation for phase relationship
        cross_corr_gps = signal.correlate(gps_signal - np.mean(gps_signal), 
                                         bio_signal - np.mean(bio_signal), mode='full')
        max_corr_idx_gps = np.argmax(np.abs(cross_corr_gps))
        phase_lag_gps = (max_corr_idx_gps - len(bio_signal) + 1) / len(bio_signal) * 2 * np.pi
        
        coupling_results['gps'] = {
            'correlation': abs(gps_bio_correlation),
            'phase_lag_rad': phase_lag_gps,
            'coupling_strength': abs(gps_bio_correlation) * np.cos(phase_lag_gps),
            'harmonic_ratio': self.tech_frequencies['gps_l1'] / bio_freq
        }
        
        # Cellular-biological coupling
        cellular_signal = cellular_model['total_network_signal']
        cellular_bio_correlation = np.corrcoef(cellular_signal, bio_signal)[0,1]
        
        cross_corr_cellular = signal.correlate(cellular_signal - np.mean(cellular_signal), 
                                             bio_signal - np.mean(bio_signal), mode='full')
        max_corr_idx_cellular = np.argmax(np.abs(cross_corr_cellular))
        phase_lag_cellular = (max_corr_idx_cellular - len(bio_signal) + 1) / len(bio_signal) * 2 * np.pi
        
        coupling_results['cellular'] = {
            'correlation': abs(cellular_bio_correlation),
            'phase_lag_rad': phase_lag_cellular,
            'coupling_strength': abs(cellular_bio_correlation) * np.cos(phase_lag_cellular),
            'band_specific': {}
        }
        
        # Band-specific cellular coupling
        for band, signals in cellular_model['tower_signals'].items():
            band_signal = np.sum(signals, axis=0)
            band_correlation = abs(np.corrcoef(band_signal, bio_signal)[0,1])
            coupling_results['cellular']['band_specific'][band] = {
                'correlation': band_correlation,
                'frequency': self.tech_frequencies[band],
                'harmonic_ratio': self.tech_frequencies[band] / bio_freq
            }
        
        # Device-biological coupling
        device_signal = device_model['total_device_signal']
        device_bio_correlation = np.corrcoef(device_signal, bio_signal)[0,1]
        
        cross_corr_device = signal.correlate(device_signal - np.mean(device_signal), 
                                           bio_signal - np.mean(bio_signal), mode='full')
        max_corr_idx_device = np.argmax(np.abs(cross_corr_device))
        phase_lag_device = (max_corr_idx_device - len(bio_signal) + 1) / len(bio_signal) * 2 * np.pi
        
        coupling_results['device'] = {
            'correlation': abs(device_bio_correlation),
            'phase_lag_rad': phase_lag_device,
            'coupling_strength': abs(device_bio_correlation) * np.cos(phase_lag_device),
            'oscillator_specific': {}
        }
        
        # Oscillator-specific device coupling
        for osc_type, signals in device_model['device_signals'].items():
            osc_signal = np.sum(signals, axis=0)
            osc_correlation = abs(np.corrcoef(osc_signal, bio_signal)[0,1])
            coupling_results['device']['oscillator_specific'][osc_type] = {
                'correlation': osc_correlation,
                'frequency': self.tech_frequencies[osc_type],
                'harmonic_ratio': self.tech_frequencies[osc_type] / bio_freq
            }
        
        # Calculate meta-oscillatory network properties
        all_tech_signals = [gps_signal, cellular_signal, device_signal]
        
        # Network synchronization
        sync_matrix = np.corrcoef(all_tech_signals + [bio_signal])
        network_synchronization = np.mean(np.abs(sync_matrix[:-1, -1]))  # Tech-bio correlations
        
        # Network coherence
        tech_coherence = np.mean([np.abs(sync_matrix[i,j]) for i in range(3) for j in range(i+1,3)])
        
        coupling_results['meta_network'] = {
            'synchronization': network_synchronization,
            'tech_coherence': tech_coherence,
            'overall_coupling': np.mean([
                coupling_results['gps']['coupling_strength'],
                coupling_results['cellular']['coupling_strength'],
                coupling_results['device']['coupling_strength']
            ]),
            'biological_signal': bio_signal_type,
            'biological_frequency': bio_freq
        }
        
        return coupling_results
    
    def simulate_feedback_dynamics(self, time_hours: float = 48, coupling_strength: float = 0.1) -> Dict:
        """Simulate bidirectional feedback dynamics between technological and biological systems"""
        
        def coupled_oscillator_system(y, t, coupling):
            """System of coupled oscillators: biological and technological"""
            bio_freq = self.bio_frequencies['circadian']
            tech_freq = self.tech_frequencies['satellite_orbit']  # Use orbital freq as proxy
            
            # y = [bio_amplitude, bio_phase, tech_amplitude, tech_phase]
            bio_amp, bio_phase, tech_amp, tech_phase = y
            
            # Coupled oscillator equations with feedback
            dbio_amp = -0.01 * bio_amp + coupling * tech_amp * np.cos(tech_phase - bio_phase)
            dbio_phase = 2 * np.pi * bio_freq + coupling * (tech_amp / bio_amp) * np.sin(tech_phase - bio_phase)
            
            dtech_amp = -0.005 * tech_amp + 0.5 * coupling * bio_amp * np.cos(bio_phase - tech_phase)
            dtech_phase = 2 * np.pi * tech_freq + 0.5 * coupling * (bio_amp / tech_amp) * np.sin(bio_phase - tech_phase)
            
            return [dbio_amp, dbio_phase, dtech_amp, dtech_phase]
        
        # Time array
        t_seconds = np.linspace(0, time_hours * 3600, int(time_hours * 100))  # 0.01s resolution
        
        # Initial conditions: [bio_amp, bio_phase, tech_amp, tech_phase]
        y0 = [1.0, 0.0, 1.0, 0.0]
        
        # Solve coupled system
        solution = odeint(coupled_oscillator_system, y0, t_seconds, args=(coupling_strength,))
        
        bio_amplitude = solution[:, 0]
        bio_phase = solution[:, 1]
        tech_amplitude = solution[:, 2]
        tech_phase = solution[:, 3]
        
        # Reconstruct oscillatory signals
        bio_signal = bio_amplitude * np.sin(bio_phase)
        tech_signal = tech_amplitude * np.sin(tech_phase)
        
        # Calculate coupling metrics
        phase_difference = bio_phase - tech_phase
        phase_locking = np.abs(np.mean(np.exp(1j * phase_difference)))
        
        amplitude_correlation = np.corrcoef(bio_amplitude, tech_amplitude)[0,1]
        signal_correlation = np.corrcoef(bio_signal, tech_signal)[0,1]
        
        # Analyze frequency entrainment
        bio_freq_inst = np.diff(bio_phase) / (2 * np.pi * np.diff(t_seconds))
        tech_freq_inst = np.diff(tech_phase) / (2 * np.pi * np.diff(t_seconds))
        
        freq_entrainment = 1 - np.std(bio_freq_inst - tech_freq_inst) / np.mean(bio_freq_inst + tech_freq_inst)
        
        return {
            'time_seconds': t_seconds,
            'bio_signal': bio_signal,
            'tech_signal': tech_signal,
            'bio_amplitude': bio_amplitude,
            'tech_amplitude': tech_amplitude,
            'bio_phase': bio_phase,
            'tech_phase': tech_phase,
            'phase_difference': phase_difference,
            'phase_locking': phase_locking,
            'amplitude_correlation': amplitude_correlation,
            'signal_correlation': signal_correlation,
            'frequency_entrainment': freq_entrainment,
            'coupling_strength': coupling_strength
        }
    
    def generate_infrastructure_report(self, gps_model: Dict, cellular_model: Dict, 
                                     device_model: Dict, coupling_results: Dict) -> str:
        """Generate comprehensive infrastructure analysis report"""
        
        report = """
=== TECHNOLOGICAL INFRASTRUCTURE OSCILLATORY ANALYSIS ===

GPS Satellite System:
- Number of satellites: {n_sats}
- Orbital frequency: {orbital_freq:.6f} Hz ({orbital_period:.2f} hours)
- L1 carrier frequency: {l1_freq:.3e} Hz
- Constellation coherence: {constellation_coherence:.4f}
- Signal power: {signal_power:.4f}

Cellular Network Infrastructure:
- Number of towers: {n_towers}
- Frequency bands: {freq_bands}
- Network coherence: {network_coherence:.4f}
- Band powers: {band_powers}

Smart Device Ensemble:
- Number of devices: {n_devices}
- Oscillator types: {osc_types}
- Ensemble coherence: {device_coherence}

Biological Coupling Analysis ({bio_type}):
""".format(
            n_sats=gps_model['n_satellites'],
            orbital_freq=gps_model['orbital_frequency'],
            orbital_period=1/gps_model['orbital_frequency']/3600,
            l1_freq=gps_model['l1_frequency'],
            constellation_coherence=gps_model['constellation_coherence'],
            signal_power=gps_model['signal_power'],
            n_towers=cellular_model['n_towers'],
            freq_bands=cellular_model['frequency_bands'],
            network_coherence=cellular_model['network_coherence'],
            band_powers=[f"{p:.3f}" for p in cellular_model['band_powers']],
            n_devices=device_model['n_devices'],
            osc_types=device_model['oscillator_types'],
            device_coherence={k: f"{v:.3f}" for k, v in device_model['ensemble_coherence'].items()},
            bio_type=coupling_results['meta_network']['biological_signal']
        )
        
        # GPS coupling
        gps_coupling = coupling_results['gps']
        report += f"\nGPS-Biological Coupling:\n"
        report += f"  - Correlation: {gps_coupling['correlation']:.4f}\n"
        report += f"  - Phase lag: {gps_coupling['phase_lag_rad']:.4f} radians\n"
        report += f"  - Coupling strength: {gps_coupling['coupling_strength']:.4f}\n"
        report += f"  - Harmonic ratio: {gps_coupling['harmonic_ratio']:.2e}\n"
        
        # Cellular coupling
        cellular_coupling = coupling_results['cellular']
        report += f"\nCellular-Biological Coupling:\n"
        report += f"  - Overall correlation: {cellular_coupling['correlation']:.4f}\n"
        report += f"  - Phase lag: {cellular_coupling['phase_lag_rad']:.4f} radians\n"
        report += f"  - Coupling strength: {cellular_coupling['coupling_strength']:.4f}\n"
        
        report += f"  - Band-specific coupling:\n"
        for band, band_coupling in cellular_coupling['band_specific'].items():
            report += f"    * {band}: correlation = {band_coupling['correlation']:.4f}, "
            report += f"harmonic ratio = {band_coupling['harmonic_ratio']:.2e}\n"
        
        # Device coupling
        device_coupling = coupling_results['device']
        report += f"\nDevice-Biological Coupling:\n"
        report += f"  - Overall correlation: {device_coupling['correlation']:.4f}\n"
        report += f"  - Phase lag: {device_coupling['phase_lag_rad']:.4f} radians\n"
        report += f"  - Coupling strength: {device_coupling['coupling_strength']:.4f}\n"
        
        report += f"  - Oscillator-specific coupling:\n"
        for osc_type, osc_coupling in device_coupling['oscillator_specific'].items():
            report += f"    * {osc_type}: correlation = {osc_coupling['correlation']:.4f}, "
            report += f"harmonic ratio = {osc_coupling['harmonic_ratio']:.2e}\n"
        
        # Meta-network analysis
        meta_network = coupling_results['meta_network']
        report += f"\nMeta-Oscillatory Network Properties:\n"
        report += f"  - Network synchronization: {meta_network['synchronization']:.4f}\n"
        report += f"  - Technological coherence: {meta_network['tech_coherence']:.4f}\n"
        report += f"  - Overall coupling strength: {meta_network['overall_coupling']:.4f}\n"
        report += f"  - Biological frequency: {meta_network['biological_frequency']:.6f} Hz\n"
        
        return report
    
    def plot_infrastructure_analysis(self, gps_model: Dict, cellular_model: Dict, 
                                   device_model: Dict, coupling_results: Dict,
                                   save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of infrastructure analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Technological Infrastructure Oscillatory Analysis', fontsize=16)
        
        time_hours = gps_model['time_seconds'] / 3600
        
        # Plot 1: GPS constellation signals
        axes[0, 0].plot(time_hours, gps_model['total_signal'], alpha=0.7, label='Total GPS Signal')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Signal Amplitude')
        axes[0, 0].set_title(f'GPS Constellation Signal\n({gps_model["n_satellites"]} satellites)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cellular network signals
        total_cellular = cellular_model['total_network_signal']
        axes[0, 1].plot(time_hours, total_cellular, alpha=0.7, color='orange', label='Total Cellular')
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Signal Amplitude')
        axes[0, 1].set_title(f'Cellular Network Signal\n({cellular_model["n_towers"]} towers)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Device ensemble signals
        total_device = device_model['total_device_signal']
        axes[0, 2].plot(time_hours, total_device, alpha=0.7, color='green', label='Total Device')
        axes[0, 2].set_xlabel('Time (hours)')
        axes[0, 2].set_ylabel('Signal Amplitude')
        axes[0, 2].set_title(f'Smart Device Ensemble\n({device_model["n_devices"]} devices)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Biological coupling comparison
        bio_type = coupling_results['meta_network']['biological_signal']
        bio_freq = coupling_results['meta_network']['biological_frequency']
        bio_signal = np.sin(2 * np.pi * bio_freq * gps_model['time_seconds'])
        
        axes[1, 0].plot(time_hours, bio_signal, 'k-', alpha=0.8, label=f'{bio_type.title()} Rhythm')
        axes[1, 0].plot(time_hours, gps_model['total_signal'] / np.max(gps_model['total_signal']), 
                       '--', alpha=0.6, label='GPS (normalized)')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Normalized Amplitude')
        axes[1, 0].set_title('GPS-Biological Coupling')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Coupling strength comparison
        coupling_strengths = [
            coupling_results['gps']['coupling_strength'],
            coupling_results['cellular']['coupling_strength'],
            coupling_results['device']['coupling_strength']
        ]
        infrastructure_types = ['GPS', 'Cellular', 'Devices']
        
        bars = axes[1, 1].bar(infrastructure_types, coupling_strengths, 
                             color=['blue', 'orange', 'green'], alpha=0.7)
        axes[1, 1].set_ylabel('Coupling Strength')
        axes[1, 1].set_title('Infrastructure-Biological Coupling')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, strength in zip(bars, coupling_strengths):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{strength:.3f}', ha='center', va='bottom')
        
        # Plot 6: Meta-network properties
        network_metrics = {
            'Synchronization': coupling_results['meta_network']['synchronization'],
            'Tech Coherence': coupling_results['meta_network']['tech_coherence'],
            'Overall Coupling': coupling_results['meta_network']['overall_coupling']
        }
        
        axes[1, 2].bar(range(len(network_metrics)), list(network_metrics.values()), 
                      color='purple', alpha=0.7)
        axes[1, 2].set_xticks(range(len(network_metrics)))
        axes[1, 2].set_xticklabels(list(network_metrics.keys()), rotation=45)
        axes[1, 2].set_ylabel('Metric Value')
        axes[1, 2].set_title('Meta-Oscillatory Network Properties')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Infrastructure analysis plot saved to {save_path}")
        
        plt.show()

def main():
    """Demonstration of technological infrastructure oscillatory modeling"""
    
    logger.info("Initializing Technological Infrastructure Oscillatory Model...")
    
    # Create infrastructure model
    infrastructure = TechnologicalOscillatoryInfrastructure()
    
    # Model parameters
    time_hours = 48  # 2 days of simulation
    sampling_rate = 1/60  # 1 sample per minute
    
    logger.info("Modeling GPS satellite system...")
    gps_model = infrastructure.model_gps_satellite_system(time_hours, sampling_rate)
    
    logger.info("Modeling cellular network infrastructure...")
    cellular_model = infrastructure.model_cellular_network(time_hours, sampling_rate, coverage_area_km2=50)
    
    logger.info("Modeling smart device ensemble...")
    device_model = infrastructure.model_smart_device_ensemble(time_hours, sampling_rate, n_devices=50)
    
    logger.info("Calculating biological coupling...")
    coupling_results = infrastructure.calculate_biological_coupling(
        gps_model, cellular_model, device_model, bio_signal_type='circadian'
    )
    
    logger.info("Simulating feedback dynamics...")
    feedback_results = infrastructure.simulate_feedback_dynamics(time_hours=24, coupling_strength=0.05)
    
    # Generate comprehensive report
    report = infrastructure.generate_infrastructure_report(
        gps_model, cellular_model, device_model, coupling_results
    )
    
    print("=== TECHNOLOGICAL INFRASTRUCTURE ANALYSIS COMPLETE ===\n")
    print(report)
    
    # Additional feedback analysis
    print(f"\nFeedback Dynamics Analysis:")
    print(f"  - Phase locking strength: {feedback_results['phase_locking']:.4f}")
    print(f"  - Amplitude correlation: {feedback_results['amplitude_correlation']:.4f}")
    print(f"  - Signal correlation: {feedback_results['signal_correlation']:.4f}")
    print(f"  - Frequency entrainment: {feedback_results['frequency_entrainment']:.4f}")
    
    # Create visualizations
    infrastructure.plot_infrastructure_analysis(
        gps_model, cellular_model, device_model, coupling_results
    )
    
    logger.info("Technological infrastructure analysis complete!")

if __name__ == "__main__":
    main()
