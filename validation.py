#validation.py
#To ensure your simulation aligns with industrial benchmarks and physical realities, focus on these essential validation steps:
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json

class SmartFactoryValidator:
    """Validator class for Smart Factory Beamforming Scenario"""
    
    def __init__(self, config, result_dir: str):
        """
        Initialize the validator
        
        Args:
            config: Configuration object
            result_dir: Directory to save validation results
        """
        self.config = config
        self.result_dir = result_dir
        self.validation_results = {}
        
        # Define validation thresholds
        self.thresholds = {
            'ris_gain_db': 3.0,  # Minimum expected RIS gain in dB
            'los_ratio': 0.3,    # Expected LOS ratio
            'delay_spread_ns': 100.0,  # Expected delay spread in ns
            'doppler_hz': 78.0,  # Expected Doppler shift at 3 km/h
            'beam_accuracy': 0.9, # Minimum beam alignment accuracy
            'energy_reduction': 0.5  # Minimum beam training reduction
        }

    def validate_ris_effectiveness(self, channel_response: Dict) -> Dict:
        """
        Validate RIS effectiveness through SNR gain
        
        Args:
            channel_response: Dictionary containing channel responses
            
        Returns:
            Dictionary containing validation results
        """
        h_with_ris = channel_response['h_with_ris']
        h_without_ris = channel_response['h_without_ris']
        
        # Calculate SNR gain in dB
        snr_gain = 10 * tf.math.log(
            tf.reduce_mean(tf.abs(h_with_ris)**2) / 
            tf.reduce_mean(tf.abs(h_without_ris)**2)
        ) / tf.math.log(10.0)
        
        result = {
            'metric': 'RIS SNR Gain',
            'value': float(snr_gain),
            'unit': 'dB',
            'threshold': self.thresholds['ris_gain_db'],
            'passed': float(snr_gain) >= self.thresholds['ris_gain_db']
        }
        
        self.validation_results['ris_effectiveness'] = result
        return result

    def validate_los_ratio(self, channel_response: Dict) -> Dict:
        """
        Validate LOS/NLOS ratio
        
        Args:
            channel_response: Dictionary containing channel responses
            
        Returns:
            Dictionary containing validation results
        """
        los_ratio = tf.reduce_mean(tf.cast(channel_response['los_condition'], tf.float32))
        
        result = {
            'metric': 'LOS Ratio',
            'value': float(los_ratio),
            'unit': '%',
            'threshold': self.thresholds['los_ratio'],
            'passed': abs(float(los_ratio) - self.thresholds['los_ratio']) <= 0.1
        }
        
        self.validation_results['los_ratio'] = result
        return result

    def validate_delay_spread(self, channel_response: Dict) -> Dict:
        """
        Validate RMS delay spread
        
        Args:
            channel_response: Dictionary containing channel responses
            
        Returns:
            Dictionary containing validation results
        """
        tau = channel_response['tau'].numpy().flatten()
        valid_tau = tau[~np.isnan(tau)]
        rms_delay = np.sqrt(np.mean(valid_tau**2))
        
        result = {
            'metric': 'RMS Delay Spread',
            'value': float(rms_delay * 1e9),
            'unit': 'ns',
            'threshold': self.thresholds['delay_spread_ns'],
            'passed': abs(float(rms_delay * 1e9) - self.thresholds['delay_spread_ns']) <= 20
        }
        
        self.validation_results['delay_spread'] = result
        return result

    def validate_mobility_impact(self, channel_response: Dict) -> Dict:
        """
        Validate AGV mobility impact through Doppler shift
        
        Args:
            channel_response: Dictionary containing channel responses
            
        Returns:
            Dictionary containing validation results
        """
        h_with_ris = channel_response['h_with_ris']
        doppler_shift = tf.reduce_mean(
            tf.angle(h_with_ris[:, 1:] * tf.math.conj(h_with_ris[:, :-1]))
        )
        
        result = {
            'metric': 'Doppler Shift',
            'value': float(doppler_shift),
            'unit': 'Hz',
            'threshold': self.thresholds['doppler_hz'],
            'passed': abs(float(doppler_shift) - self.thresholds['doppler_hz']) <= 10
        }
        
        self.validation_results['mobility_impact'] = result
        return result

    def validate_beamforming(self, predicted_beams: np.ndarray, optimal_beams: np.ndarray) -> Dict:
        """
        Validate beamforming accuracy
        
        Args:
            predicted_beams: Array of predicted beam indices
            optimal_beams: Array of optimal beam indices
            
        Returns:
            Dictionary containing validation results
        """
        accuracy = np.mean(predicted_beams == optimal_beams)
        
        result = {
            'metric': 'Beam Alignment Accuracy',
            'value': float(accuracy),
            'unit': '%',
            'threshold': self.thresholds['beam_accuracy'],
            'passed': accuracy >= self.thresholds['beam_accuracy']
        }
        
        self.validation_results['beamforming'] = result
        return result

    def validate_xai_plausibility(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Validate XAI plausibility through SHAP values
        
        Args:
            shap_values: Array of SHAP values
            feature_names: List of feature names
            
        Returns:
            Dictionary containing validation results
        """
        top_features = np.argsort(shap_values)[-3:]
        important_features = ['RIS Phase', 'Shelf Distance', 'AGV Position']
        
        # Check if top features include important ones
        has_important = any(feature_names[i] in important_features for i in top_features)
        
        result = {
            'metric': 'XAI Plausibility',
            'value': has_important,
            'unit': None,
            'top_features': [feature_names[i] for i in top_features],
            'passed': has_important
        }
        
        self.validation_results['xai_plausibility'] = result
        return result

    def validate_energy_efficiency(self, baseline_scans: int, xai_scans: int) -> Dict:
        """
        Validate energy efficiency through beam training reduction
        
        Args:
            baseline_scans: Number of baseline beam scans
            xai_scans: Number of XAI-guided beam scans
            
        Returns:
            Dictionary containing validation results
        """
        reduction = (baseline_scans - xai_scans) / baseline_scans
        
        result = {
            'metric': 'Beam Training Reduction',
            'value': float(reduction),
            'unit': '%',
            'threshold': self.thresholds['energy_reduction'],
            'passed': reduction >= self.thresholds['energy_reduction']
        }
        
        self.validation_results['energy_efficiency'] = result
        return result

    def run_full_validation(self, channel_response: Dict, 
                        predicted_beams: np.ndarray, 
                        optimal_beams: np.ndarray,
                        shap_values: np.ndarray,
                        feature_names: List[str],
                        baseline_scans: int,
                        xai_scans: int) -> Dict:
        """
        Run all validation checks
        
        Args:
            channel_response: Dictionary containing channel responses
            predicted_beams: Array of predicted beam indices
            optimal_beams: Array of optimal beam indices
            shap_values: Array of SHAP values
            feature_names: List of feature names
            baseline_scans: Number of baseline beam scans
            xai_scans: Number of XAI-guided beam scans
            
        Returns:
            Dictionary containing all validation results
        """
        self.validate_ris_effectiveness(channel_response)
        self.validate_los_ratio(channel_response)
        self.validate_delay_spread(channel_response)
        self.validate_mobility_impact(channel_response)
        self.validate_beamforming(predicted_beams, optimal_beams)
        self.validate_xai_plausibility(shap_values, feature_names)
        self.validate_energy_efficiency(baseline_scans, xai_scans)
        
        # Save validation results
        self._save_validation_results()
        
        return self.validation_results

    def _save_validation_results(self):
        """Save validation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.result_dir, f'validation_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=4)
        
        # Generate validation report plot
        self._plot_validation_results()

    def _plot_validation_results(self):
        """Generate validation results visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics = []
        values = []
        passed = []
        
        for key, result in self.validation_results.items():
            if isinstance(result['value'], (int, float)):
                metrics.append(result['metric'])
                values.append(result['value'])
                passed.append(result['passed'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics, values)
        
        # Color bars based on pass/fail
        for bar, pass_status in zip(bars, passed):
            bar.set_color('green' if pass_status else 'red')
        
        plt.xticks(rotation=45, ha='right')
        plt.title('Validation Results Overview')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.result_dir, f'validation_results_{timestamp}.png')
        plt.savefig(plot_file)
        plt.close()

def create_validator(config, result_dir: str) -> SmartFactoryValidator:
    """
    Factory function to create a validator instance
    
    Args:
        config: Configuration object
        result_dir: Directory to save validation results
        
    Returns:
        SmartFactoryValidator instance
    """
    return SmartFactoryValidator(config, result_dir)

#how to use" # In main.py
#from validation import create_validator

#def main():
    # Your existing initialization code...
    
    # Create validator
    #validator = create_validator(config, result_dir)
    
    # After generating channel response and other data...
    #validation_results = validator.run_full_validation(
        #channel_response=channel_response,
        #predicted_beams=predicted_beams,
        #optimal_beams=optimal_beams,
        #shap_values=shap_values,
        #feature_names=feature_names,
        #baseline_scans=baseline_scans,
        #xai_scans=xai_scans
    #)
    
    #print("Validation complete. Results saved in:", result_dir)