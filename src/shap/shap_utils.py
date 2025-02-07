import tensorflow as tf
import numpy as np

def preprocess_channel_data(channel_response, config):
    """Preprocess channel data for SHAP analysis"""
    features = []
    
    if config.shap['features']['channel_response']:
        features.append(tf.abs(channel_response['h']))
        
    if config.shap['features']['los_condition']:
        features.append(tf.cast(channel_response['los_condition'], tf.float32))
        
    if config.shap['features']['agv_position']:
        features.append(tf.reshape(channel_response['agv_positions'], [-1]))
        
    if config.shap['features']['ris_state'] and 'ris_state' in channel_response:
        features.append(tf.reshape(channel_response['ris_state'], [-1]))
    
    return tf.concat(features, axis=-1)

def get_feature_names(config):
    """Get feature names for SHAP analysis"""
    feature_names = []
    
    if config.shap['features']['channel_response']:
        feature_names.extend([f'h_{i}' for i in range(config.num_subcarriers)])
        
    if config.shap['features']['los_condition']:
        feature_names.append('los_condition')
        
    if config.shap['features']['agv_position']:
        feature_names.extend(['agv_x', 'agv_y', 'agv_z'])
        
    if config.shap['features']['ris_state']:
        feature_names.extend([f'ris_{i}' for i in range(np.prod(config.ris_elements))])
        
    return feature_names

def normalize_shap_values(shap_values):
    """Normalize SHAP values for better visualization"""
    return shap_values / np.abs(shap_values).max()

def aggregate_shap_values(shap_values, feature_groups):
    """Aggregate SHAP values by feature groups"""
    aggregated = {}
    for group_name, indices in feature_groups.items():
        aggregated[group_name] = tf.reduce_sum(shap_values[..., indices], axis=-1)
    return aggregated