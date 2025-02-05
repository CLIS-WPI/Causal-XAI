import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import logging
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChannelValidator:
    """Validator class for Smart Factory channel simulation"""
    
    def __init__(self, config):
        """
        Initialize validator with configuration parameters
        
        Parameters:
        -----------
        config : SmartFactoryConfig
            Configuration object containing validation thresholds
        """
        self.config = config
        self.validation_results = {}
        
        # Define validation thresholds
        self.thresholds = {
            'min_ris_gain_db': 3.0,  # Minimum expected RIS gain in dB
            'expected_los_ratio': 0.3,  # Expected LOS ratio in factory
            'expected_delay_spread_ns': 100,  # Expected RMS delay spread in ns
            'expected_doppler_hz': 78,  # Expected Doppler shift at 28GHz
            'min_beam_accuracy': 0.9,  # Minimum beam alignment accuracy
            'min_energy_reduction': 0.5  # Minimum energy efficiency improvement
        }

    def validate_ris_effectiveness(self, channel_response: Dict[str, tf.Tensor]) -> Tuple[bool, float]:
        """
        Validate RIS effectiveness through SNR gain
        
        Returns:
        --------
        bool : Validation pass/fail
        float : Calculated SNR gain
        """
        try:
            h_with_ris = channel_response['h_with_ris']
            h_without_ris = channel_response['h_without_ris']
            
            # Calculate SNR gain in dB
            snr_gain = 10 * tf.math.log(
                tf.reduce_mean(tf.abs(h_with_ris)**2) / 
                tf.reduce_mean(tf.abs(h_without_ris)**2)
            ) / tf.math.log(10.0)
            
            passes = float(snr_gain) >= self.thresholds['min_ris_gain_db']
            
            logger.info(f"RIS SNR Gain: {float(snr_gain):.1f} dB (Required: ≥{self.thresholds['min_ris_gain_db']} dB)")
            return passes, float(snr_gain)
            
        except Exception as e:
            logger.error(f"RIS effectiveness validation failed: {str(e)}")
            return False, 0.0

    def validate_los_ratio(self, channel_response: Dict[str, Any]) -> Tuple[bool, float]:
        """Validate LOS/NLOS ratio"""
        try:
            los_ratio = tf.reduce_mean(tf.cast(channel_response['los_condition'], tf.float32))
            
            # Allow for ±10% deviation from expected ratio
            passes = abs(float(los_ratio) - self.thresholds['expected_los_ratio']) <= 0.1
            
            logger.info(f"LOS Ratio: {float(los_ratio):.1%} (Expected: {self.thresholds['expected_los_ratio']:.1%})")
            return passes, float(los_ratio)
            
        except Exception as e:
            logger.error(f"LOS ratio validation failed: {str(e)}")
            return False, 0.0

    def validate_delay_spread(self, channel_response: Dict[str, Any]) -> Tuple[bool, float]:
        """Validate RMS delay spread"""
        try:
            tau = channel_response['tau'].numpy().flatten()
            valid_tau = tau[~np.isnan(tau)]
            rms_delay = np.sqrt(np.mean(valid_tau**2)) * 1e9  # Convert to ns
            
            # Allow for ±20% deviation from expected delay spread
            passes = abs(rms_delay - self.thresholds['expected_delay_spread_ns']) <= 20
            
            logger.info(f"RMS Delay Spread: {rms_delay:.1f} ns (Expected: {self.thresholds['expected_delay_spread_ns']} ns)")
            return passes, rms_delay
            
        except Exception as e:
            logger.error(f"Delay spread validation failed: {str(e)}")
            return False, 0.0

    def validate_doppler_shift(self, channel_response: Dict[str, tf.Tensor]) -> Tuple[bool, float]:
        """Validate Doppler shift consistency"""
        try:
            h = channel_response['h_with_ris']
            doppler_shift = tf.reduce_mean(
                tf.angle(h[:, 1:] * tf.math.conj(h[:, :-1]))
            ) / (2 * np.pi * self.config.sampling_time)
            
            # Allow for ±10Hz deviation from expected Doppler
            passes = abs(float(doppler_shift) - self.thresholds['expected_doppler_hz']) <= 10
            
            logger.info(f"Doppler Shift: {float(doppler_shift):.1f} Hz (Expected: {self.thresholds['expected_doppler_hz']} Hz)")
            return passes, float(doppler_shift)
            
        except Exception as e:
            logger.error(f"Doppler validation failed: {str(e)}")
            return False, 0.0

    def validate_beam_accuracy(self, predicted_beams: np.ndarray, optimal_beams: np.ndarray) -> Tuple[bool, float]:
        """Validate beamforming accuracy"""
        try:
            accuracy = np.mean(predicted_beams == optimal_beams)
            passes = accuracy >= self.thresholds['min_beam_accuracy']
            
            logger.info(f"Beam Accuracy: {accuracy:.1%} (Required: ≥{self.thresholds['min_beam_accuracy']:.1%})")
            return passes, accuracy
            
        except Exception as e:
            logger.error(f"Beam accuracy validation failed: {str(e)}")
            return False, 0.0

    def validate_energy_efficiency(self, baseline_scans: int, xai_scans: int) -> Tuple[bool, float]:
        """Validate energy efficiency improvement"""
        try:
            reduction = (baseline_scans - xai_scans) / baseline_scans
            passes = reduction >= self.thresholds['min_energy_reduction']
            
            logger.info(f"Energy Efficiency Improvement: {reduction:.1%} (Required: ≥{self.thresholds['min_energy_reduction']:.1%})")
            return passes, reduction
            
        except Exception as e:
            logger.error(f"Energy efficiency validation failed: {str(e)}")
            return False, 0.0

    def run_full_validation(self, channel_response: Dict[str, Any], 
                        predicted_beams: np.ndarray, 
                        optimal_beams: np.ndarray,
                        baseline_scans: int,
                        xai_scans: int) -> Dict[str, Any]:
        """
        Run all validation checks and generate report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Run all validations
        self.validation_results = {
            'ris_effectiveness': self.validate_ris_effectiveness(channel_response),
            'los_ratio': self.validate_los_ratio(channel_response),
            'delay_spread': self.validate_delay_spread(channel_response),
            'doppler_shift': self.validate_doppler_shift(channel_response),
            'beam_accuracy': self.validate_beam_accuracy(predicted_beams, optimal_beams),
            'energy_efficiency': self.validate_energy_efficiency(baseline_scans, xai_scans)
        }
        
        # Generate validation report
        self.generate_validation_report(timestamp)
        
        return self.validation_results

    def generate_validation_report(self, timestamp: str) -> None:
        """Generate and save validation report"""
        report_dir = os.path.join(os.getcwd(), 'validation_reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f'validation_report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write("Smart Factory Channel Validation Report\n")
            f.write("=====================================\n\n")
            
            for metric, (passes, value) in self.validation_results.items():
                status = "PASS" if passes else "FAIL"
                f.write(f"{metric}: {status} (Value: {value:.3f})\n")
            
            f.write(f"\nValidation completed at: {timestamp}")
        
        logger.info(f"Validation report saved to: {report_path}")