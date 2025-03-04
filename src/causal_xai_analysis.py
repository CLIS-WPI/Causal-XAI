# src/causal_xai_analysis.py
import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import shap
from pathlib import Path
import logging
from sklearn.linear_model import LogisticRegression  # Added at the top for consistency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='causal_xai_analysis.log')
logger = logging.getLogger(__name__)

class CausalXaiAnalysis:
    def __init__(self, sim_file_path, perf_file_path):
        """Initialize with paths to simulation and performance files."""
        self.sim_file = Path(sim_file_path)
        self.perf_file = Path(perf_file_path)
        self.step_results = {}
        logger.info(f"Initialized with sim_file: {self.sim_file}, perf_file: {self.perf_file}")

    def validate_hdf5(self, file_path):
        """Validate the structure and values of an HDF5 file."""
        logger.info(f"Validating file: {file_path}")
        try:
            with h5py.File(file_path, 'r') as f:
                logger.info(f"Top-level keys: {list(f.keys())}")
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        subkeys = list(f[key].keys())
                        logger.info(f"Group {key}: {subkeys}")
                        for subkey in subkeys:
                            subgroup = f[key][subkey]
                            if isinstance(subgroup, h5py.Group):
                                logger.info(f"  Subgroup {subkey}: {list(subgroup.keys())}")
                                for subsubkey in subgroup.keys():
                                    data = subgroup[subsubkey][()]
                                    logger.info(f"    {subsubkey}: shape={data.shape}, dtype={data.dtype}")
                                    if data.size > 0:
                                        logger.info(f"    {subsubkey}: min={np.min(data)}, max={np.max(data)}")
                                    else:
                                        logger.warning(f"    {subsubkey} is empty (size=0)")
                                    if data.size > 0 and (np.any(np.isnan(data)) or np.any(np.isinf(data))):
                                        logger.warning(f"    {subsubkey} contains NaN or Inf values!")
                            else:
                                data = subgroup[()]
                                logger.info(f"  {subkey}: shape={data.shape}, dtype={data.dtype}")
                                if data.size > 0:
                                    logger.info(f"  {subkey}: min={np.min(data)}, max={np.max(data)}")
                                else:
                                    logger.warning(f"  {subkey} is empty (size=0)")
                                if data.size > 0 and (np.any(np.isnan(data)) or np.any(np.isinf(data))):
                                    logger.warning(f"  {subkey} contains NaN or Inf values!")
                    else:
                        data = f[key][()]
                        logger.info(f"Dataset {key}: shape={data.shape}, dtype={data.dtype}")
                        if data.size > 0:
                            logger.info(f"Dataset {key}: min={np.min(data)}, max={np.max(data)}")
                        else:
                            logger.warning(f"Dataset {key} is empty (size=0)")
            return True
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return False

    def load_all_steps(self):
        """Load and analyze all steps from simulation_data.h5."""
        if not self.sim_file.exists():
            logger.error(f"Simulation file not found: {self.sim_file}")
            return

        with h5py.File(self.sim_file, 'r') as f:
            steps = [key for key in f.keys() if key.startswith('step_')]
            for step in steps:
                step_num = int(step.split('_')[1])
                logger.info(f"Processing step {step_num}")
                self.analyze_single_step(f, step_num)

        if not self.step_results:
            logger.warning("No steps were successfully processed")

    def analyze_single_step(self, h5_file, step_num):
        """Analyze a single step from the HDF5 file."""
        try:
            step_group = h5_file[f'step_{step_num}']
            required = ['mobility_data/agv_positions', 'mobility_data/los_conditions', 'beam_data/snr_db', 'beam_data/beam_directions']
            for r in required:
                if r not in step_group:
                    logger.warning(f"Missing {r} in step {step_num}")
                    return

            agv_positions = step_group['mobility_data/agv_positions'][()]
            los_conditions = step_group['mobility_data/los_conditions'][()]
            snr = step_group['beam_data/snr_db'][()]
            beam_directions = step_group['beam_data/beam_directions'][()]
            switch_reason = step_group['beam_data'].attrs.get('switch_reason', 'None')

            if step_num == 0:
                B = np.zeros(len(agv_positions))
            else:
                prev_step = h5_file[f'step_{step_num-1}']
                prev_beams = prev_step['beam_data/beam_directions'][()]
                B = np.any(beam_directions != prev_beams, axis=1).astype(np.float32)

            O = np.array(los_conditions == 0, dtype=np.float32)
            M = np.array(snr, dtype=np.float32)
            P = np.linalg.norm(agv_positions[:, :2], axis=1)
            S = np.array([1 if 'switch' in switch_reason.lower() else 0] * len(agv_positions), dtype=np.float32)

            min_length = min(len(O), len(M), len(B), len(P), len(S))
            O, M, B, P, S = O[:min_length], M[:min_length], B[:min_length], P[:min_length], S[:min_length]

            self.step_results[step_num] = {
                'O': O, 'M': M, 'B': B, 'P': P, 'S': S,
                'effects': self.calculate_effects(O, M, B, P, S, step_num),  # Pass step_num here
                'beam_directions': beam_directions[:min_length, 0]  # Store only azimuth (1D)
            }
            logger.debug(f"Step {step_num} analyzed successfully")

        except Exception as e:
            logger.error(f"Error in step {step_num}: {e}")

    def calculate_effects(self, O, M, B, P, S, step_num):    
            """Calculate causal effects with scikit-learn LogisticRegression and fallback models to handle perfect separation."""
            try:
                data = pd.DataFrame({'O': O, 'M': M, 'B': B, 'P': P, 'S': S})
                # Log B distribution with more detail, including total samples and class balance
                b_unique, b_counts = np.unique(B, return_counts=True) if B.size > 0 else ([0], [0])
                b_dist = dict(zip(b_unique, b_counts))
                min_samples_per_class = 1  # Reduced minimum samples required per class for LogisticRegression
                logger.debug(f"Step {step_num} - B distribution: {b_dist}, total samples: {len(B)}")
                X1 = sm.add_constant(data[['O']])
                X2 = sm.add_constant(data[['O', 'P', 'S']])
                X3 = sm.add_constant(data[['O', 'M', 'P', 'S']])

                # Debug shapes of X2 and X3, and input data
                logger.debug(f"Step {step_num} - X2 shape: {X2.shape}, X3 shape: {X3.shape}")
                logger.debug(f"Step {step_num} - Data shapes: O={len(O)}, M={len(M)}, B={len(B)}, P={len(P)}, S={len(S)}")
                logger.debug(f"Step {step_num} - B unique values: {b_unique}, counts: {b_counts}")

                # Use OLS for O -> M (continuous outcome)
                model1 = sm.OLS(data['M'], X1).fit()  # O -> M

                # Handle single-class or imbalanced data in B
                if len(b_unique) < 2 or any(count < min_samples_per_class for count in b_counts):
                    logger.warning(f"Step {step_num} - B has insufficient class variation ({b_dist}), using OLS as fallback")
                    # Attempt to aggregate with previous/next step if available and data is minimal
                    if step_num > 0 and step_num < 9 and step_num in self.step_results and len(B) < 4:  # Aggregate if total samples < 4
                        prev_b = self.step_results[step_num - 1]['B'] if step_num - 1 in self.step_results else None
                        next_b = self.step_results[step_num + 1]['B'] if step_num + 1 in self.step_results else None
                        if prev_b is not None and len(np.unique(prev_b)) >= 2 and all(count >= min_samples_per_class for count in np.unique(prev_b, return_counts=True)[1]):
                            logger.info(f"Step {step_num} - Aggregating B with previous step")
                            B = np.concatenate([prev_b, B])
                            data['B'] = B
                            X2 = sm.add_constant(data[['O', 'P', 'S']])
                            X3 = sm.add_constant(data[['O', 'M', 'P', 'S']])
                        elif next_b is not None and len(np.unique(next_b)) >= 2 and all(count >= min_samples_per_class for count in np.unique(next_b, return_counts=True)[1]):
                            logger.info(f"Step {step_num} - Aggregating B with next step")
                            B = np.concatenate([B, next_b])
                            data['B'] = B
                            X2 = sm.add_constant(data[['O', 'P', 'S']])
                            X3 = sm.add_constant(data[['O', 'M', 'P', 'S']])
                    model2 = sm.OLS(data['B'], X2).fit()
                    model3 = sm.OLS(data['B'], X3).fit()
                else:
                    try:
                        # Try L2 regularization first with adjusted C
                        model2 = LogisticRegression(penalty='l2', C=0.5, max_iter=2000, random_state=42)  # Reduced C, increased iterations
                        model2.fit(X2, data['B'])
                        # Manually create params Series, ensuring length matches columns
                        model2_params = np.insert(model2.coef_[0], 0, model2.intercept_)
                        if len(model2_params) != len(X2.columns):
                            logger.warning(f"Step {step_num} - Mismatch in model2 params length: {len(model2_params)} vs columns {len(X2.columns)}")
                            # Try L1 regularization with saga solver for potential sparsity
                            model2 = LogisticRegression(penalty='l1', C=0.5, max_iter=2000, random_state=42, solver='saga')
                            model2.fit(X2, data['B'])
                            model2_params = np.insert(model2.coef_[0], 0, model2.intercept_)
                            if len(model2_params) != len(X2.columns):
                                logger.warning(f"Step {step_num} - L1 also failed for model2 (saga), trying RandomForest as fallback")
                                from sklearn.ensemble import RandomForestClassifier
                                model2 = RandomForestClassifier(n_estimators=100, random_state=42)
                                model2.fit(X2, data['B'])
                                # Estimate coefficients for compatibility (simplified approximation, ensuring correct shape)
                                model2_params = np.zeros(len(X2.columns))
                                model2_params[0] = model2.predict_proba(X2)[:, 1].mean() if X2.shape[0] > 0 else 0.0  # Handle empty data
                                if X2.shape[1] > 1:  # Ensure features exist
                                    model2_params[1:] = model2.feature_importances_
                                else:
                                    model2_params[1:] = 0.0
                                model2.params = pd.Series(model2_params, index=X2.columns)
                            else:
                                model2.params = pd.Series(model2_params, index=X2.columns)
                        else:
                            model2.params = pd.Series(model2_params, index=X2.columns)

                        # Model 3: O -> M -> B
                        model3 = LogisticRegression(penalty='l2', C=0.5, max_iter=2000, random_state=42)
                        model3.fit(X3, data['B'])
                        model3_params = np.insert(model3.coef_[0], 0, model3.intercept_)
                        if len(model3_params) != len(X3.columns):
                            logger.warning(f"Step {step_num} - Mismatch in model3 params length: {len(model3_params)} vs columns {len(X3.columns)}")
                            model3 = LogisticRegression(penalty='l1', C=0.5, max_iter=2000, random_state=42, solver='saga')
                            model3.fit(X3, data['B'])
                            model3_params = np.insert(model3.coef_[0], 0, model3.intercept_)
                            if len(model3_params) != len(X3.columns):
                                logger.warning(f"Step {step_num} - L1 also failed for model3 (saga), trying RandomForest as fallback")
                                from sklearn.ensemble import RandomForestClassifier
                                model3 = RandomForestClassifier(n_estimators=100, random_state=42)
                                model3.fit(X3, data['B'])
                                model3_params = np.zeros(len(X3.columns))
                                model3_params[0] = model3.predict_proba(X3)[:, 1].mean() if X3.shape[0] > 0 else 0.0
                                if X3.shape[1] > 1:
                                    model3_params[1:] = model3.feature_importances_
                                else:
                                    model3_params[1:] = 0.0
                                model3.params = pd.Series(model3_params, index=X3.columns)
                            else:
                                model3.params = pd.Series(model3_params, index=X3.columns)
                        else:
                            model3.params = pd.Series(model3_params, index=X3.columns)
                    except Exception as e:
                        logger.warning(f"LogisticRegression failed for models, using OLS as fallback: {e}")
                        model2 = sm.OLS(data['B'], X2).fit()
                        model3 = sm.OLS(data['B'], X3).fit()

                effects = {
                    'total_effect': float(model2.params['O']),
                    'direct_effect': float(model3.params['O']),
                    'indirect_effect': float(model1.params['O'] * model3.params['M']),
                    'position_effect': float(model3.params['P']),
                    'switch_intent_effect': float(model3.params['S'])
                }
                return effects
            except Exception as e:
                logger.warning(f"Effect calculation failed: {e}")
                return {k: 0.0 for k in ['total_effect', 'direct_effect', 'indirect_effect', 'position_effect', 'switch_intent_effect']}

    def plot_effects_over_steps(self):
        """Plot causal effects over steps with annotations for significant steps."""
        if not self.step_results:
            logger.warning("No results to plot")
            return

        steps = sorted(self.step_results.keys())
        effects = {k: [self.step_results[s]['effects'][k] for s in steps] for k in self.step_results[0]['effects']}

        plt.figure(figsize=(15, 8))
        for label, values in effects.items():
            plt.plot(steps, values, label=label.replace('_', ' ').title(), linewidth=2)
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Effect Size', fontsize=12)
        plt.legend(fontsize=10)
        plt.title('Causal Effects Over Steps', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Annotate significant steps (e.g., steps 4 and 6 with large effects)
        for step in [4, 6]:
            plt.annotate(f'Step {step}', (step, effects['total_effect'][step]), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        plt.savefig(self.sim_file.parent / 'effects_over_steps.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Effects plot saved to {self.sim_file.parent / 'effects_over_steps.png'}")

    def xai_analysis(self):
        """Perform XAI analysis using SHAP for beam directions, including Switch Intent."""
        if not self.step_results:
            logger.warning("No data for XAI analysis")
            return

        all_data = pd.DataFrame({
            'O': np.concatenate([self.step_results[s]['O'] for s in self.step_results]),
            'M': np.concatenate([self.step_results[s]['M'] for s in self.step_results]),
            'B': np.concatenate([self.step_results[s]['B'] for s in self.step_results]),
            'P': np.concatenate([self.step_results[s]['P'] for s in self.step_results]),
            'S': np.concatenate([self.step_results[s]['S'] for s in self.step_results]),
            'beam_directions': np.concatenate([self.step_results[s]['beam_directions'] for s in self.step_results])
        })

        X = all_data[['O', 'M', 'P', 'S']]  # Include S in features
        y = all_data['beam_directions']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Create SHAP summary plot with custom formatting (bar plot)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("Feature Importance in Beam Direction Prediction", fontsize=14)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.savefig(self.sim_file.parent / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP bar plot saved to {self.sim_file.parent / 'shap_summary.png'}")

        # Create SHAP scatter plot to show distribution of impacts
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="dot", show=False)
        plt.title("Feature Impact Distribution in Beam Direction Prediction", fontsize=14)
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.savefig(self.sim_file.parent / 'shap_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP scatter plot saved to {self.sim_file.parent / 'shap_scatter.png'}")

    def build_causal_graph(self):
        """Build and visualize the causal graph with edge weights."""
        G = nx.DiGraph()
        G.add_nodes_from(['O', 'M', 'B', 'P', 'S'])
        G.add_edges_from([
            ('O', 'M'), ('O', 'B'),
            ('M', 'B'),
            ('P', 'M'), ('P', 'B'),
            ('S', 'M'), ('S', 'B')
        ])
        
        # Calculate average effect magnitudes for edge weights (simplified)
        avg_effects = {}
        if self.step_results:
            steps = list(self.step_results.keys())
            effects = [self.step_results[s]['effects'] for s in steps]
            avg_effects['O->M'] = np.mean([e['indirect_effect'] for e in effects if e['indirect_effect'] != 0.0] or [0.0])
            avg_effects['O->B'] = np.mean([e['direct_effect'] for e in effects if e['direct_effect'] != 0.0] or [0.0])
            avg_effects['M->B'] = np.mean([e['indirect_effect'] for e in effects if e['indirect_effect'] != 0.0] or [0.0])
            avg_effects['P->M'] = np.mean([e['position_effect'] for e in effects if e['position_effect'] != 0.0] or [0.0])
            avg_effects['P->B'] = np.mean([e['position_effect'] for e in effects if e['position_effect'] != 0.0] or [0.0])
            avg_effects['S->M'] = np.mean([e['switch_intent_effect'] for e in effects if e['switch_intent_effect'] != 0.0] or [0.0])
            avg_effects['S->B'] = np.mean([e['switch_intent_effect'] for e in effects if e['switch_intent_effect'] != 0.0] or [0.0])

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', arrows=True)

        # Add edge labels (weights)
        edge_labels = {}
        for u, v in G.edges():
            edge_key = f"{u}->{v}"
            weight = abs(avg_effects.get(edge_key, 0.0))
            edge_labels[(u, v)] = f"{weight:.2f}" if weight > 0 else ""
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Causal DAG for Beamforming with Effect Weights", fontsize=14)
        plt.savefig(self.sim_file.parent / 'causal_graph.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Causal graph saved to {self.sim_file.parent / 'causal_graph.png'}")
        return G

if __name__ == "__main__":
    try:
        sim_file_path = "/home/tanglab/Desktop/Causal-XAI/results/simulation_data.h5"
        perf_file_path = "/home/tanglab/Desktop/Causal-XAI/results/performance_metrics.h5"

        analysis = CausalXaiAnalysis(sim_file_path, perf_file_path)

        if analysis.validate_hdf5(sim_file_path) and analysis.validate_hdf5(perf_file_path):
            analysis.load_all_steps()

            if analysis.step_results:
                analysis.plot_effects_over_steps()
                analysis.xai_analysis()
                analysis.build_causal_graph()

                logger.info("\nSummary of Causal Effects Across Steps:")
                for step, results in analysis.step_results.items():
                    effects = results['effects']
                    logger.info(f"\nStep {step}:")
                    for k, v in effects.items():
                        logger.info(f"{k.replace('_', ' ').title()}: {v:.3f}")
        else:
            logger.error("Validation failed, stopping analysis")

    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)