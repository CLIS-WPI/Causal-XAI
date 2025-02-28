import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.utils import resample
from pathlib import Path
import logging

# Configure logging to match the project's style
logger = logging.getLogger(__name__)

class CausalAnalysis:
    def __init__(self, results_folder):
        """Initialize CausalAnalysis with the results folder."""
        self.results_folder = Path(results_folder)
        self.step_results = {}
        logger.info(f"CausalAnalysis initialized with results folder: {self.results_folder}")

    def load_all_steps(self):
        """Load and analyze all steps from simulation_data.h5."""
        sim_file = self.results_folder / "simulation_data.h5"
        if not sim_file.exists():
            logger.error(f"Simulation data file not found: {sim_file}")
            return

        with h5py.File(sim_file, 'r') as f:
            steps = [key for key in f.keys() if key.startswith('step_')]
            for step in steps:
                step_num = int(step.split('_')[1])
                logger.info(f"Analyzing step {step_num}")
                self.analyze_single_step(f, step_num)
        
        if not self.step_results:
            logger.warning("No steps were successfully processed")

    def analyze_single_step(self, h5_file, step_num):
        """Analyze a single step from the HDF5 file."""
        try:
            step_group = h5_file[f'step_{step_num}']
            csi_data = step_group['csi_data']
            mobility_data = step_group['mobility_data']
            beam_data = step_group['beam_data']

            # Extract required metrics with fallbacks
            agv_positions = mobility_data['agv_positions'][()]
            los_conditions = mobility_data['los_conditions'][()] if 'los_conditions' in mobility_data else np.zeros(len(agv_positions), dtype=np.int32)
            snr = beam_data['snr_db'][()] if 'snr_db' in beam_data else np.zeros(len(agv_positions))
            beam_directions = beam_data['beam_directions'][()] if 'beam_directions' in beam_data else np.zeros((len(agv_positions), 2))
            switch_reason = beam_data.attrs.get('switch_reason', 'None')

            # Compute beam changes (B) based on beam_directions history
            if step_num == 0:
                B = np.zeros(len(agv_positions))  # No previous step for first step
            else:
                prev_step_group = h5_file[f'step_{step_num-1}']
                prev_beam_directions = prev_step_group['beam_data']['beam_directions'][()]
                B = np.any(beam_directions != prev_beam_directions, axis=1).astype(np.float32)

            # Convert data to appropriate format
            O = np.array(los_conditions == 0, dtype=np.float32)  # 1 when NLoS
            M = np.array(snr, dtype=np.float32)  # SNR as mediator

            # Ensure compatible shapes
            min_length = min(len(O), len(M), len(B))
            O = O[:min_length]
            M = M[:min_length]
            B = B[:min_length]

            # Include additional variables for richer analysis
            P = np.linalg.norm(agv_positions[:, :2], axis=1)[:min_length]  # Distance from origin as position proxy
            S = np.array([1 if 'switch' in switch_reason.lower() else 0] * min_length, dtype=np.float32)  # Switch intent

            # Store results
            self.step_results[step_num] = {
                'O': O,  # Blockage (treatment)
                'M': M,  # SNR (mediator)
                'B': B,  # Beam switch (outcome)
                'P': P,  # Position (covariate)
                'S': S,  # Switch reason (covariate)
                'effects': self.calculate_effects(O, M, B, P, S)
            }
            logger.debug(f"Step {step_num} analyzed successfully")

        except KeyError as e:
            logger.warning(f"Missing data in step {step_num}: {e}")
        except Exception as e:
            logger.error(f"Error processing step {step_num}: {e}")

    def calculate_effects(self, O, M, B, P, S):
        """Calculate causal effects including covariates."""
        try:
            # Create DataFrame with all variables
            data = pd.DataFrame({'O': O, 'M': M, 'B': B, 'P': P, 'S': S})
            
            # Mediation analysis with covariates
            X1 = sm.add_constant(data[['O']])  # Blockage -> SNR
            X2 = sm.add_constant(data[['O', 'P', 'S']])  # Blockage + covariates -> Beam Switch
            X3 = sm.add_constant(data[['O', 'M', 'P', 'S']])  # Full model
            
            model1 = sm.OLS(data['M'], X1).fit()  # O -> M
            model2 = sm.Logit(data['B'], X2).fit(disp=0)  # O -> B (direct + covariates)
            model3 = sm.Logit(data['B'], X3).fit(disp=0)  # O -> B via M (full model)
            
            effects = {
                'total_effect': float(model2.params['O']),
                'direct_effect': float(model3.params['O']),
                'indirect_effect': float(model1.params['O'] * model3.params['M']),
                'position_effect': float(model3.params['P']),
                'switch_intent_effect': float(model3.params['S'])
            }
            logger.debug(f"Causal effects calculated: {effects}")
            return effects
            
        except Exception as e:
            logger.warning(f"Error in effect calculation: {e}")
            return {
                'total_effect': 0.0,
                'direct_effect': 0.0,
                'indirect_effect': 0.0,
                'position_effect': 0.0,
                'switch_intent_effect': 0.0
            }

    def plot_effects_over_steps(self):
        """Plot how effects change over steps."""
        if not self.step_results:
            logger.warning("No results to plot")
            return

        steps = sorted(self.step_results.keys())
        total_effects = [self.step_results[s]['effects']['total_effect'] for s in steps]
        direct_effects = [self.step_results[s]['effects']['direct_effect'] for s in steps]
        indirect_effects = [self.step_results[s]['effects']['indirect_effect'] for s in steps]
        position_effects = [self.step_results[s]['effects']['position_effect'] for s in steps]
        switch_effects = [self.step_results[s]['effects']['switch_intent_effect'] for s in steps]

        plt.figure(figsize=(15, 8))
        plt.plot(steps, total_effects, 'b-', label='Total Effect (O → B)')
        plt.plot(steps, direct_effects, 'r-', label='Direct Effect (O → B)')
        plt.plot(steps, indirect_effects, 'g-', label='Indirect Effect (O → M → B)')
        plt.plot(steps, position_effects, 'c-', label='Position Effect (P → B)')
        plt.plot(steps, switch_effects, 'm-', label='Switch Intent Effect (S → B)')
        plt.xlabel('Step Number')
        plt.ylabel('Effect Size')
        plt.legend()
        plt.title('Causal Effects Over Steps in Beamforming Simulation')
        plt.grid(True)
        plt.savefig(self.results_folder / 'effects_over_steps.png')
        plt.close()
        logger.info(f"Effects plot saved to {self.results_folder / 'effects_over_steps.png'}")

if __name__ == "__main__":
    try:
        # Configure logging to match project style
        logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='causal_analysis.log')

        # Initialize analysis
        analysis = CausalAnalysis('results/')
        
        # Analyze all steps
        analysis.load_all_steps()
        
        # Plot effects if any data was processed
        if analysis.step_results:
            analysis.plot_effects_over_steps()
            
            # Print summary of successful analyses
            logger.info("\nSummary of Causal Effects Across Steps:")
            for step, results in analysis.step_results.items():
                effects = results['effects']
                logger.info(f"\nStep {step}:")
                logger.info(f"Total Effect: {effects['total_effect']:.3f}")
                logger.info(f"Direct Effect: {effects['direct_effect']:.3f}")
                logger.info(f"Indirect Effect: {effects['indirect_effect']:.3f}")
                logger.info(f"Position Effect: {effects['position_effect']:.3f}")
                logger.info(f"Switch Intent Effect: {effects['switch_intent_effect']:.3f}")
        else:
            logger.warning("No data was successfully processed")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)