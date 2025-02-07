import tensorflow as tf
import numpy as np

def preprocess_channel_data(channel_response, config):
    """Preprocess channel data for SHAP analysis"""
    # First determine the common shape to use
    channel_shape = channel_response['h'].shape
    feature_length = tf.size(channel_response['h'])  # Total size of channel response
    
    features = []
    
    if config.shap['features']['channel_response']:
        # Reshape channel response to be flat
        channel = tf.reshape(tf.abs(channel_response['h']), [feature_length])
        features.append(channel)
        
    if config.shap['features']['los_condition']:
        # Expand LOS condition to match feature length
        los = tf.cast(channel_response['los_condition'], tf.float32)
        los = tf.broadcast_to(los, [feature_length])
        features.append(los)
        
    if config.shap['features']['agv_position']:
        # Expand AGV positions to match feature length
        positions = tf.reshape(channel_response['agv_positions'], [-1])
        positions = tf.tile(positions, [feature_length // tf.size(positions) + 1])[:feature_length]
        features.append(positions)
        
    if config.shap['features']['ris_state'] and 'ris_state' in channel_response:
        # Expand RIS state to match feature length
        ris_state = tf.reshape(channel_response['ris_state'], [-1])
        ris_state = tf.tile(ris_state, [feature_length // tf.size(ris_state) + 1])[:feature_length]
        features.append(ris_state)
    
    # Concatenate all features
    return tf.concat(features, axis=0)

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