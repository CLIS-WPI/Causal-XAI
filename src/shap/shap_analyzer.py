#src/shap/shape_analyzer.py
import tensorflow as tf
import shap
import numpy as np

class ShapAnalyzer:
    def __init__(self, config):
        self.config = config
        self.background_data = None
        
    def prepare_background_data(self, channel_data, num_samples=None):
            """Prepare background data for SHAP analysis"""
            if num_samples is None:
                num_samples = 10  # Default minimum samples
                
            # Extract relevant features from channel data
            features = {
                'h_with_ris': channel_data['h_with_ris'],
                'h_without_ris': channel_data['h_without_ris'],
                'los_condition': channel_data['los_condition'],
                'agv_positions': channel_data['agv_positions']
            }
            
            return features
        
    def analyze_channel_response(self, channel_response):
        """Analyze channel response using SHAP"""
        if self.background_data is None:
            raise ValueError("Background data not prepared. Call prepare_background_data first.")
            
        # Create explainer
        explainer = shap.DeepExplainer(self._get_model(), self.background_data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(channel_response)
        
        return {
            'shap_values': shap_values,
            'feature_importance': self._calculate_feature_importance(shap_values),
            'interaction_values': self._calculate_interaction_values(shap_values)
        }
        
    def _get_model(self):
        """Get simplified model for SHAP analysis"""
        def model(x):
            # Extract features
            channel = x[..., :self.config.num_subcarriers]
            los = x[..., -1:]
            
            # Simple channel quality metric
            quality = tf.abs(channel) * tf.cast(los, tf.float32)
            return quality
            
        return model
        
    def _calculate_feature_importance(self, shap_values):
        """Calculate feature importance from SHAP values"""
        return tf.reduce_mean(tf.abs(shap_values), axis=0)
        
    def _calculate_interaction_values(self, shap_values):
        """Calculate interaction values between features"""
        interactions = {}
        for i in range(shap_values.shape[-1]):
            for j in range(i+1, shap_values.shape[-1]):
                interaction = tf.reduce_mean(
                    tf.abs(shap_values[..., i] * shap_values[..., j])
                )
                interactions[f"feature_{i}_{j}"] = interaction
        return interactions