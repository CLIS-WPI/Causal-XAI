#src/shap/shap_utils.py
import matplotlib.pyplot as plt
import shap
import numpy as np

class ShapVisualizer:
    def __init__(self, config):
        self.config = config
        self.fig_size = config.shap['visualization']['figure_size']
        
    def plot_feature_importance(self, shap_values, feature_names=None):
        """Plot feature importance using SHAP values"""
        plt.figure(figsize=self.fig_size)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(shap_values.shape[-1])]
            
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=self.config.shap['visualization']['max_display'],
            plot_type=self.config.shap['visualization']['plot_type']
        )
        
    def plot_interaction_values(self, interaction_values, feature_names=None):
        """Plot feature interactions"""
        plt.figure(figsize=self.fig_size)
        
        # Create interaction matrix
        n_features = len(feature_names) if feature_names else int(np.sqrt(len(interaction_values)))
        interaction_matrix = np.zeros((n_features, n_features))
        
        # Fill interaction matrix
        for key, value in interaction_values.items():
            i, j = map(int, key.split('_')[1:])
            interaction_matrix[i, j] = value
            interaction_matrix[j, i] = value
            
        plt.imshow(interaction_matrix)
        plt.colorbar(label='Interaction Strength')
        
        if feature_names:
            plt.xticks(range(n_features), feature_names, rotation=45)
            plt.yticks(range(n_features), feature_names)
            
        plt.title('Feature Interactions')
        plt.tight_layout()
        
    def plot_shap_waterfall(self, shap_values, feature_names=None):
        """Plot SHAP waterfall plot"""
        plt.figure(figsize=self.fig_size)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(shap_values.shape[-1])]
            
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values,
                feature_names=feature_names
            )
        )