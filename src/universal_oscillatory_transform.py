"""
Universal Oscillatory Transformation Framework
Transforms any input data into differential equations and applies s-entropy analysis
Handles tridimensional differential forms: dx/dt, dx/dinfo, dx/dentropy
"""

import numpy as np
import pandas as pd
from scipy import signal, integrate, optimize, fftpack
from scipy.linalg import solve
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifferentialForm(Enum):
    """Tridimensional differential forms"""
    TIME = "dt"           # dx/dt - time to solution  
    INFO = "dinfo"        # dx/dinfo - information differential
    ENTROPY = "dentropy"  # dx/dentropy - entropy differential

@dataclass
class TransformationResult:
    """Results from universal transformation"""
    original_data: np.ndarray
    differential_equation: Dict[str, Any]
    laplace_transform: Dict[str, Any]
    s_entropy_analysis: Dict[str, Any]
    pattern_discovery: Dict[str, Any]
    oscillatory_features: Dict[str, Any]
    confidence_score: float

class UniversalDataIngestion:
    """Universal data ingestion engine for any data source"""
    
    @staticmethod
    def ingest_data(data_source: Union[str, pd.DataFrame, np.ndarray, Dict], 
                   time_column: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Ingest any data source and extract time series"""
        
        if isinstance(data_source, str):
            # File path - try to load
            if data_source.endswith('.json'):
                import json
                with open(data_source, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            else:
                raise ValueError("Unsupported file format")
                
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            
        elif isinstance(data_source, np.ndarray):
            if data_source.ndim == 1:
                df = pd.DataFrame({'value': data_source, 'index': range(len(data_source))})
                time_column = 'index'
            else:
                df = pd.DataFrame(data_source)
                
        elif isinstance(data_source, dict):
            df = pd.DataFrame(data_source)
        else:
            raise ValueError("Unsupported data source type")
        
        # Auto-detect time column if not specified
        if time_column is None:
            time_candidates = ['timestamp', 'time', 'datetime', 'date', 'index']
            for candidate in time_candidates:
                if candidate in df.columns:
                    time_column = candidate
                    break
            else:
                # Use index as time
                df['auto_time'] = range(len(df))
                time_column = 'auto_time'
        
        # Extract time and data vectors
        time_vector = df[time_column].values
        
        # Convert time to numeric if needed
        if not np.issubdtype(time_vector.dtype, np.number):
            try:
                time_vector = pd.to_datetime(time_vector)
                time_vector = (time_vector - time_vector.min()).dt.total_seconds()
            except:
                time_vector = np.arange(len(time_vector), dtype=float)
        
        # Get all numeric columns except time
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        data_columns = [col for col in numeric_columns if col != time_column]
        
        if not data_columns:
            raise ValueError("No numeric data columns found")
        
        # Combine all data columns into signal matrix
        data_matrix = df[data_columns].values
        
        logger.info(f"Ingested data: {len(time_vector)} points, {len(data_columns)} signals")
        return time_vector, data_matrix

class DifferentialEquationGenerator:
    """Generates differential equations from any time series data"""
    
    def __init__(self):
        self.equations = {}
        self.coefficients = {}
    
    def generate_first_order_de(self, t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate first-order differential equation from time series"""
        
        # Calculate derivatives using multiple methods
        derivatives = self._calculate_derivatives(t, y)
        
        # Fit differential equation model: dy/dt = f(y, t)
        de_model = self._fit_differential_model(t, y, derivatives)
        
        # Generate symbolic representation
        symbolic_eq = self._create_symbolic_equation(de_model)
        
        return {
            'derivatives': derivatives,
            'model': de_model,
            'symbolic': symbolic_eq,
            'coefficients': de_model.get('coefficients', {}),
            'equation_type': de_model.get('type', 'general'),
            'goodness_of_fit': de_model.get('r_squared', 0)
        }
    
    def _calculate_derivatives(self, t: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate derivatives using multiple numerical methods"""
        
        derivatives = {}
        
        # Method 1: Central difference
        dt = np.mean(np.diff(t))
        dydt_central = np.gradient(y, dt)
        derivatives['central_diff'] = dydt_central
        
        # Method 2: Savitzky-Golay derivative
        try:
            if len(y) >= 5:
                dydt_savgol = signal.savgol_filter(y, min(len(y)//4*2+1, 51), 3, deriv=1, delta=dt)
                derivatives['savgol'] = dydt_savgol
        except:
            derivatives['savgol'] = dydt_central
        
        # Method 3: Smooth spline derivative
        try:
            from scipy import interpolate
            spline = interpolate.UnivariateSpline(t, y, s=0)
            dydt_spline = spline.derivative()(t)
            derivatives['spline'] = dydt_spline
        except:
            derivatives['spline'] = dydt_central
        
        # Choose best derivative (least noisy)
        noise_levels = {name: np.std(np.diff(deriv)) for name, deriv in derivatives.items()}
        best_method = min(noise_levels.keys(), key=lambda k: noise_levels[k])
        derivatives['best'] = derivatives[best_method]
        derivatives['method_used'] = best_method
        
        return derivatives
    
    def _fit_differential_model(self, t: np.ndarray, y: np.ndarray, derivatives: Dict) -> Dict:
        """Fit differential equation model to data"""
        
        dydt = derivatives['best']
        
        # Try different DE model types
        models = []
        
        # Model 1: Linear - dy/dt = a*y + b
        try:
            A = np.vstack([y, np.ones(len(y))]).T
            coeffs_linear, residuals_linear = np.linalg.lstsq(A, dydt, rcond=None)[:2]
            r_squared_linear = 1 - residuals_linear[0] / np.sum((dydt - np.mean(dydt))**2)
            
            models.append({
                'type': 'linear',
                'coefficients': {'a': coeffs_linear[0], 'b': coeffs_linear[1]},
                'r_squared': r_squared_linear,
                'equation': lambda t, y: coeffs_linear[0]*y + coeffs_linear[1]
            })
        except:
            pass
        
        # Model 2: Oscillatory - dy/dt = -w²*y + damping
        try:
            # Fit d²y/dt² + 2*zeta*w*dy/dt + w²*y = 0
            d2ydt2 = np.gradient(dydt, np.mean(np.diff(t)))
            
            # Create design matrix for [d²y/dt², dy/dt, y, 1]
            A = np.vstack([d2ydt2, dydt, y, np.ones(len(y))]).T
            target = np.zeros(len(y))  # Homogeneous equation
            
            # Use regularized least squares
            alpha = 1e-6
            ATA = A.T @ A + alpha * np.eye(A.shape[1])
            ATb = A.T @ target
            coeffs_osc = solve(ATA, ATb)
            
            # Extract physical parameters
            w_squared = -coeffs_osc[2] / coeffs_osc[0] if coeffs_osc[0] != 0 else 1
            damping = -coeffs_osc[1] / (2 * coeffs_osc[0]) if coeffs_osc[0] != 0 else 0.1
            
            if w_squared > 0:
                omega = np.sqrt(w_squared)
                predicted = np.exp(-damping*t) * (np.cos(omega*t) + np.sin(omega*t))
                r_squared_osc = 1 - np.sum((y - predicted)**2) / np.sum((y - np.mean(y))**2)
                
                models.append({
                    'type': 'oscillatory',
                    'coefficients': {'omega': omega, 'damping': damping},
                    'r_squared': r_squared_osc,
                    'equation': lambda t, y: -damping*dydt - omega**2*y
                })
        except:
            pass
        
        # Model 3: Nonlinear - dy/dt = a*y² + b*y + c
        try:
            A = np.vstack([y**2, y, np.ones(len(y))]).T
            coeffs_nonlin, residuals_nonlin = np.linalg.lstsq(A, dydt, rcond=None)[:2]
            r_squared_nonlin = 1 - residuals_nonlin[0] / np.sum((dydt - np.mean(dydt))**2)
            
            models.append({
                'type': 'nonlinear',
                'coefficients': {'a': coeffs_nonlin[0], 'b': coeffs_nonlin[1], 'c': coeffs_nonlin[2]},
                'r_squared': r_squared_nonlin,
                'equation': lambda t, y: coeffs_nonlin[0]*y**2 + coeffs_nonlin[1]*y + coeffs_nonlin[2]
            })
        except:
            pass
        
        # Choose best model
        if models:
            best_model = max(models, key=lambda m: m.get('r_squared', 0))
            logger.info(f"Best DE model: {best_model['type']} (R² = {best_model['r_squared']:.4f})")
            return best_model
        else:
            # Fallback to simple linear model
            return {
                'type': 'simple_linear',
                'coefficients': {'slope': np.mean(dydt)},
                'r_squared': 0.0,
                'equation': lambda t, y: np.mean(dydt)
            }
    
    def _create_symbolic_equation(self, model: Dict) -> str:
        """Create symbolic representation of differential equation"""
        
        if model['type'] == 'linear':
            a, b = model['coefficients']['a'], model['coefficients']['b']
            return f"dy/dt = {a:.4f}*y + {b:.4f}"
        
        elif model['type'] == 'oscillatory':
            omega, damping = model['coefficients']['omega'], model['coefficients']['damping']
            return f"d²y/dt² + {2*damping:.4f}*dy/dt + {omega**2:.4f}*y = 0"
        
        elif model['type'] == 'nonlinear':
            a, b, c = model['coefficients']['a'], model['coefficients']['b'], model['coefficients']['c']
            return f"dy/dt = {a:.4f}*y² + {b:.4f}*y + {c:.4f}"
        
        else:
            return "dy/dt = constant"

class LaplaceTransformEngine:
    """Transforms differential equations to frequency domain"""
    
    def __init__(self):
        self.transform_cache = {}
    
    def transform_to_laplace(self, de_result: Dict, t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Transform differential equation to Laplace domain"""
        
        # Extract differential equation
        eq_type = de_result.get('type', 'linear')
        coefficients = de_result.get('coefficients', {})
        
        # Compute Laplace transform symbolically and numerically
        symbolic_laplace = self._symbolic_laplace_transform(eq_type, coefficients)
        numerical_laplace = self._numerical_laplace_transform(t, y)
        
        return {
            'symbolic_transform': symbolic_laplace,
            'numerical_transform': numerical_laplace,
            'transfer_function': self._create_transfer_function(eq_type, coefficients),
            'poles_zeros': self._analyze_poles_zeros(eq_type, coefficients),
            'frequency_response': self._compute_frequency_response(t, y)
        }
    
    def _symbolic_laplace_transform(self, eq_type: str, coeffs: Dict) -> Dict:
        """Compute symbolic Laplace transform"""
        
        s = sp.Symbol('s')
        Y = sp.Symbol('Y')
        
        if eq_type == 'linear':
            # dy/dt = a*y + b  =>  s*Y(s) - y(0) = a*Y(s) + b/s
            # Y(s) = (y(0) + b/s) / (s - a)
            a, b = coeffs.get('a', 0), coeffs.get('b', 0)
            transfer_func = 1 / (s - a)
            forced_response = b / (s * (s - a))
            
        elif eq_type == 'oscillatory':
            # d²y/dt² + 2*zeta*w*dy/dt + w²*y = 0
            # s²*Y(s) + 2*zeta*w*s*Y(s) + w²*Y(s) = initial conditions
            omega = coeffs.get('omega', 1)
            damping = coeffs.get('damping', 0.1)
            transfer_func = 1 / (s**2 + 2*damping*omega*s + omega**2)
            forced_response = transfer_func
            
        elif eq_type == 'nonlinear':
            # Nonlinear - approximate as linear around operating point
            b = coeffs.get('b', 1)
            transfer_func = 1 / (s - b)
            forced_response = transfer_func
            
        else:
            transfer_func = 1 / s
            forced_response = 1 / s
        
        return {
            'transfer_function': str(transfer_func),
            'forced_response': str(forced_response),
            'equation_type': eq_type,
            'symbolic_s_domain': f"({transfer_func})"
        }
    
    def _numerical_laplace_transform(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Compute numerical Laplace transform"""
        
        # Use FFT-based approximation to Laplace transform
        dt = np.mean(np.diff(t))
        n = len(y)
        
        # Frequency domain via FFT
        Y_fft = fftpack.fft(y)
        frequencies = fftpack.fftfreq(n, dt)
        
        # Convert to s-domain (s = jω)
        s_values = 1j * 2 * np.pi * frequencies
        
        # Magnitude and phase
        magnitude = np.abs(Y_fft)
        phase = np.angle(Y_fft)
        
        return {
            's_values': s_values,
            'magnitude': magnitude,
            'phase': phase,
            'frequencies': frequencies,
            'sampling_rate': 1/dt
        }
    
    def _create_transfer_function(self, eq_type: str, coeffs: Dict) -> Callable:
        """Create transfer function for frequency analysis"""
        
        if eq_type == 'oscillatory':
            omega = coeffs.get('omega', 1)
            damping = coeffs.get('damping', 0.1)
            
            def H(s):
                return 1 / (s**2 + 2*damping*omega*s + omega**2)
            return H
            
        elif eq_type == 'linear':
            a = coeffs.get('a', -1)
            
            def H(s):
                return 1 / (s - a)
            return H
            
        else:
            def H(s):
                return 1 / s
            return H
    
    def _analyze_poles_zeros(self, eq_type: str, coeffs: Dict) -> Dict:
        """Analyze poles and zeros of the system"""
        
        poles = []
        zeros = []
        
        if eq_type == 'oscillatory':
            omega = coeffs.get('omega', 1)
            damping = coeffs.get('damping', 0.1)
            
            # Characteristic equation: s² + 2*ζ*ω*s + ω² = 0
            discriminant = (damping*omega)**2 - omega**2
            
            if discriminant >= 0:
                # Overdamped
                poles = [-damping*omega + np.sqrt(discriminant), 
                        -damping*omega - np.sqrt(discriminant)]
            else:
                # Underdamped
                real_part = -damping*omega
                imag_part = omega*np.sqrt(1 - damping**2)
                poles = [complex(real_part, imag_part), complex(real_part, -imag_part)]
                
        elif eq_type == 'linear':
            a = coeffs.get('a', -1)
            poles = [a]
            
        return {
            'poles': poles,
            'zeros': zeros,
            'system_stability': all(np.real(p) < 0 for p in poles) if poles else True
        }
    
    def _compute_frequency_response(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Compute frequency response characteristics"""
        
        dt = np.mean(np.diff(t))
        fs = 1 / dt
        
        # Compute power spectral density
        frequencies, psd = signal.periodogram(y, fs)
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(psd, height=np.max(psd)*0.1)[0]
        dominant_freqs = frequencies[peak_indices]
        dominant_powers = psd[peak_indices]
        
        return {
            'frequencies': frequencies,
            'power_spectral_density': psd,
            'dominant_frequencies': dominant_freqs,
            'dominant_powers': dominant_powers,
            'total_power': np.sum(psd),
            'bandwidth': frequencies[np.where(psd > np.max(psd)/2)[0][-1]] if len(np.where(psd > np.max(psd)/2)[0]) > 0 else fs/2
        }

class SEntropyAnalyzer:
    """Tridimensional S-Entropy analysis with interchangeable differential forms"""
    
    def __init__(self):
        self.entropy_cache = {}
        self.differential_forms = {
            DifferentialForm.TIME: self._compute_time_entropy,
            DifferentialForm.INFO: self._compute_info_entropy,
            DifferentialForm.ENTROPY: self._compute_entropy_entropy
        }
    
    def analyze_s_entropy(self, t: np.ndarray, y: np.ndarray, 
                         differential_form: DifferentialForm = DifferentialForm.TIME) -> Dict:
        """Compute S-entropy in specified differential form"""
        
        logger.info(f"Computing S-entropy in {differential_form.value} form")
        
        # Base entropy calculations
        base_entropy = self._compute_base_entropy(y)
        
        # Differential form specific entropy
        form_entropy = self.differential_forms[differential_form](t, y)
        
        # Tridimensional entropy matrix
        tridimensional_entropy = self._compute_tridimensional_entropy(t, y)
        
        # Interchangeable form analysis
        interchangeable_analysis = self._analyze_interchangeable_forms(t, y)
        
        return {
            'base_entropy': base_entropy,
            'differential_form': differential_form.value,
            'form_specific_entropy': form_entropy,
            'tridimensional_matrix': tridimensional_entropy,
            'interchangeable_analysis': interchangeable_analysis,
            'entropy_dimension': 3,
            'total_s_entropy': self._compute_total_s_entropy(base_entropy, form_entropy, tridimensional_entropy)
        }
    
    def _compute_base_entropy(self, y: np.ndarray) -> Dict:
        """Compute base entropy measures"""
        
        # Shannon entropy
        hist, _ = np.histogram(y, bins=50)
        hist = hist[hist > 0]  # Remove zeros
        shannon_entropy = -np.sum((hist/np.sum(hist)) * np.log2(hist/np.sum(hist)))
        
        # Approximate entropy
        def _maxdist(xi, xj, N):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _approximate_entropy(U, m, r):
            N = len(U)
            patterns = np.array([U[i:i+m] for i in range(N-m+1)])
            
            C = np.zeros(N-m+1)
            for i in range(N-m+1):
                template_i = patterns[i]
                matches = sum([1 for j in range(N-m+1) 
                              if _maxdist(template_i, patterns[j], m) <= r])
                C[i] = matches / (N-m+1)
            
            phi_m = np.mean(np.log(C))
            
            patterns_m1 = np.array([U[i:i+m+1] for i in range(N-m)])
            C_m1 = np.zeros(N-m)
            for i in range(N-m):
                template_i = patterns_m1[i]
                matches = sum([1 for j in range(N-m) 
                              if _maxdist(template_i, patterns_m1[j], m+1) <= r])
                C_m1[i] = matches / (N-m)
            
            phi_m1 = np.mean(np.log(C_m1))
            return phi_m - phi_m1
        
        try:
            r = 0.2 * np.std(y)
            approx_entropy = _approximate_entropy(y, 2, r)
        except:
            approx_entropy = shannon_entropy
        
        return {
            'shannon_entropy': shannon_entropy,
            'approximate_entropy': approx_entropy,
            'variance_entropy': np.log2(2*np.pi*np.e*np.var(y))/2  # Differential entropy for Gaussian
        }
    
    def _compute_time_entropy(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Compute entropy in time differential form (dx/dt)"""
        
        # Time to solution analysis
        dt = np.diff(t)
        dy = np.diff(y)
        
        # Time-based entropy rate
        time_entropy_rate = []
        window_size = min(50, len(dt)//10)
        
        for i in range(len(dt) - window_size):
            window = dy[i:i+window_size]
            hist, _ = np.histogram(window, bins=10)
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy = -np.sum((hist/np.sum(hist)) * np.log2(hist/np.sum(hist)))
                time_entropy_rate.append(entropy)
        
        return {
            'time_entropy_rate': np.array(time_entropy_rate),
            'mean_time_entropy': np.mean(time_entropy_rate) if time_entropy_rate else 0,
            'time_complexity': np.std(dt),
            'temporal_regularity': 1 / (1 + np.std(dt)/np.mean(dt)) if np.mean(dt) > 0 else 0
        }
    
    def _compute_info_entropy(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Compute entropy in information differential form (dx/dinfo)"""
        
        # Information-based entropy
        # Compute mutual information between consecutive points
        def mutual_information(x, y, bins=20):
            hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
            hist_x, _ = np.histogram(x, bins=bins)
            hist_y, _ = np.histogram(y, bins=bins)
            
            # Normalize
            hist_xy = hist_xy / np.sum(hist_xy)
            hist_x = hist_x / np.sum(hist_x)
            hist_y = hist_y / np.sum(hist_y)
            
            mi = 0
            for i in range(bins):
                for j in range(bins):
                    if hist_xy[i,j] > 0:
                        mi += hist_xy[i,j] * np.log2(hist_xy[i,j] / (hist_x[i] * hist_y[j]))
            
            return mi
        
        if len(y) > 1:
            mi = mutual_information(y[:-1], y[1:])
            
            # Information gain rate
            info_gain = []
            window_size = min(30, len(y)//10)
            for i in range(len(y) - window_size):
                window = y[i:i+window_size]
                gain = np.var(window[-window_size//2:]) / (np.var(window[:window_size//2]) + 1e-10)
                info_gain.append(gain)
        else:
            mi = 0
            info_gain = [0]
        
        return {
            'mutual_information': mi,
            'info_gain_rate': np.array(info_gain),
            'mean_info_gain': np.mean(info_gain),
            'information_complexity': np.std(info_gain)
        }
    
    def _compute_entropy_entropy(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Compute entropy in entropy differential form (dx/dentropy)"""
        
        # Entropy of entropy - meta-entropy analysis
        window_size = min(20, len(y)//5)
        entropy_series = []
        
        for i in range(len(y) - window_size):
            window = y[i:i+window_size]
            hist, _ = np.histogram(window, bins=10)
            hist = hist[hist > 0]
            if len(hist) > 0:
                entropy = -np.sum((hist/np.sum(hist)) * np.log2(hist/np.sum(hist)))
                entropy_series.append(entropy)
        
        if entropy_series:
            # Entropy of the entropy series itself
            entropy_hist, _ = np.histogram(entropy_series, bins=10)
            entropy_hist = entropy_hist[entropy_hist > 0]
            
            if len(entropy_hist) > 0:
                meta_entropy = -np.sum((entropy_hist/np.sum(entropy_hist)) * 
                                     np.log2(entropy_hist/np.sum(entropy_hist)))
            else:
                meta_entropy = 0
                
            # Entropy derivative
            entropy_derivative = np.gradient(entropy_series)
        else:
            meta_entropy = 0
            entropy_derivative = np.array([0])
        
        return {
            'entropy_series': np.array(entropy_series),
            'meta_entropy': meta_entropy,
            'entropy_derivative': entropy_derivative,
            'entropy_complexity': np.std(entropy_series) if entropy_series else 0
        }
    
    def _compute_tridimensional_entropy(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Compute tridimensional entropy matrix"""
        
        # Compute entropy in all three forms
        time_entropy = self._compute_time_entropy(t, y)
        info_entropy = self._compute_info_entropy(t, y)
        entropy_entropy = self._compute_entropy_entropy(t, y)
        
        # Create 3D entropy matrix
        entropy_matrix = np.array([
            [time_entropy['mean_time_entropy'], 
             time_entropy['time_complexity'],
             time_entropy['temporal_regularity']],
            [info_entropy['mutual_information'],
             info_entropy['mean_info_gain'],
             info_entropy['information_complexity']],
            [entropy_entropy['meta_entropy'],
             entropy_entropy['entropy_complexity'],
             np.mean(entropy_entropy['entropy_derivative'])]
        ])
        
        # Compute matrix properties
        eigenvals, eigenvecs = np.linalg.eig(entropy_matrix)
        determinant = np.linalg.det(entropy_matrix)
        trace = np.trace(entropy_matrix)
        
        return {
            'entropy_matrix': entropy_matrix,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'determinant': determinant,
            'trace': trace,
            'matrix_rank': np.linalg.matrix_rank(entropy_matrix),
            'condition_number': np.linalg.cond(entropy_matrix)
        }
    
    def _analyze_interchangeable_forms(self, t: np.ndarray, y: np.ndarray) -> Dict:
        """Analyze interchangeability between differential forms"""
        
        # Compute all three forms
        forms = {}
        for form in DifferentialForm:
            forms[form.value] = self.differential_forms[form](t, y)
        
        # Cross-correlations between forms
        correlations = {}
        
        # Extract comparable metrics from each form
        time_metric = forms['dt']['mean_time_entropy']
        info_metric = forms['dinfo']['mutual_information']
        entropy_metric = forms['dentropy']['meta_entropy']
        
        metrics = np.array([time_metric, info_metric, entropy_metric])
        
        # Correlation matrix
        if len(metrics) > 1 and np.std(metrics) > 0:
            # Create synthetic series for correlation analysis
            time_series = np.array(forms['dt']['time_entropy_rate'])
            info_series = np.array(forms['dinfo']['info_gain_rate'])
            entropy_series = np.array(forms['dentropy']['entropy_series'])
            
            # Ensure same length for correlation
            min_len = min(len(time_series), len(info_series), len(entropy_series))
            if min_len > 5:
                time_series = time_series[:min_len]
                info_series = info_series[:min_len]
                entropy_series = entropy_series[:min_len]
                
                corr_matrix = np.corrcoef([time_series, info_series, entropy_series])
            else:
                corr_matrix = np.eye(3)
        else:
            corr_matrix = np.eye(3)
        
        return {
            'form_correlations': corr_matrix,
            'interchangeability_score': np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])),
            'dominant_form': max(forms.keys(), key=lambda k: np.mean(list(forms[k].values())[:3])),
            'form_stability': 1 - np.std([time_metric, info_metric, entropy_metric]) / (np.mean([time_metric, info_metric, entropy_metric]) + 1e-10)
        }
    
    def _compute_total_s_entropy(self, base: Dict, form: Dict, tridim: Dict) -> float:
        """Compute total S-entropy combining all forms"""
        
        # Weight the different entropy contributions
        base_contrib = base['shannon_entropy'] * 0.3
        form_contrib = np.mean(list(form.values())[:3]) * 0.4  # Take first 3 numeric values
        tridim_contrib = abs(tridim['determinant']) * 0.3
        
        return base_contrib + form_contrib + tridim_contrib

def main():
    """Test the universal transformation framework"""
    
    # Generate test data
    t = np.linspace(0, 10, 1000)
    y = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t) + 0.1*np.random.normal(size=len(t))
    
    data = pd.DataFrame({'time': t, 'signal': y})
    
    logger.info("Testing Universal Oscillatory Transformation Framework...")
    
    # 1. Data ingestion
    ingestion = UniversalDataIngestion()
    time_vec, data_matrix = ingestion.ingest_data(data, time_column='time')
    logger.info(f"Data ingested: {len(time_vec)} points")
    
    # 2. Generate differential equation
    de_gen = DifferentialEquationGenerator()
    de_result = de_gen.generate_first_order_de(time_vec, data_matrix[:, 0])
    logger.info(f"Generated DE: {de_result['symbolic']}")
    
    # 3. Laplace transform
    laplace_engine = LaplaceTransformEngine()
    laplace_result = laplace_engine.transform_to_laplace(de_result, time_vec, data_matrix[:, 0])
    logger.info(f"Laplace transform computed")
    
    # 4. S-entropy analysis
    s_entropy = SEntropyAnalyzer()
    entropy_result = s_entropy.analyze_s_entropy(time_vec, data_matrix[:, 0], DifferentialForm.TIME)
    logger.info(f"S-entropy: {entropy_result['total_s_entropy']:.4f}")
    
    print(f"\nResults:")
    print(f"Differential Equation: {de_result['symbolic']}")
    print(f"Total S-Entropy: {entropy_result['total_s_entropy']:.4f}")
    print(f"System Stability: {laplace_result['poles_zeros']['system_stability']}")

if __name__ == "__main__":
    main()
