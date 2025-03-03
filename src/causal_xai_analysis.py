#src/causal_xai_analysis.py
import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
import shap
from pathlib import Path
import logging

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
                'effects': self.calculate_effects(O, M, B, P, S),
                'beam_directions': beam_directions[:min_length, 0]  # Store only azimuth (1D)
            }
            logger.debug(f"Step {step_num} analyzed successfully")

        except Exception as e:
            logger.error(f"Error in step {step_num}: {e}")

    def calculate_effects(self, O, M, B, P, S):
        """Calculate causal effects with fallback for singular matrices."""
        try:
            data = pd.DataFrame({'O': O, 'M': M, 'B': B, 'P': P, 'S': S})
            X1 = sm.add_constant(data[['O']])
            X2 = sm.add_constant(data[['O', 'P', 'S']])
            X3 = sm.add_constant(data[['O', 'M', 'P', 'S']])

            model1 = sm.OLS(data['M'], X1).fit()  # O -> M
            # Use OLS as fallback if Logit fails due to perfect separation
            try:
                model2 = sm.Logit(data['B'], X2).fit(disp=0, maxiter=100)  # O -> B
            except:
                logger.warning("Logit failed for model2, using OLS as fallback")
                model2 = sm.OLS(data['B'], X2).fit()
            try:
                model3 = sm.Logit(data['B'], X3).fit(disp=0, maxiter=100)  # O -> M -> B
            except:
                logger.warning("Logit failed for model3, using OLS as fallback")
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
        """Plot causal effects over steps."""
        if not self.step_results:
            logger.warning("No results to plot")
            return

        steps = sorted(self.step_results.keys())
        effects = {k: [self.step_results[s]['effects'][k] for s in steps] for k in self.step_results[0]['effects']}

        plt.figure(figsize=(15, 8))
        for label, values in effects.items():
            plt.plot(steps, values, label=label.replace('_', ' ').title())
        plt.xlabel('Step Number')
        plt.ylabel('Effect Size')
        plt.legend()
        plt.title('Causal Effects Over Steps')
        plt.grid(True)
        plt.savefig(self.sim_file.parent / 'effects_over_steps.png')
        plt.close()
        logger.info(f"Effects plot saved to {self.sim_file.parent / 'effects_over_steps.png'}")

    def xai_analysis(self):
        """Perform XAI analysis using SHAP for beam directions."""
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

        X = all_data[['O', 'M', 'P']]
        y = all_data['beam_directions']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(self.sim_file.parent / 'shap_summary.png')
        plt.close()
        logger.info(f"SHAP plot saved to {self.sim_file.parent / 'shap_summary.png'}")

    def build_causal_graph(self):
        """Build and visualize the causal graph."""
        G = nx.DiGraph()
        G.add_nodes_from(['O', 'M', 'B', 'P', 'S'])
        G.add_edges_from([
            ('O', 'M'), ('O', 'B'),
            ('M', 'B'),
            ('P', 'M'), ('P', 'B'),
            ('S', 'M'), ('S', 'B')
        ])
        
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', arrows=True)
        plt.title("Causal DAG for Beamforming")
        plt.savefig(self.sim_file.parent / 'causal_graph.png')
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