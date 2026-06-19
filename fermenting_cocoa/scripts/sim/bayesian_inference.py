import json
import time
import numpy as np
import matplotlib.pyplot as plt
import pints
import pints.plot
import os
import matplotlib.pyplot as plt


# Import your custom build/run scripts
from fermenting_cocoa.scripts import run_model_pH_citric
from fermenting_cocoa.scripts import build_model_pH_citric

# 2. Turn LaTeX OFF absolutely LAST, so nothing can overwrite it
plt.rcParams['text.usetex'] = False


# ==========================================
# 1. Load Setup & Build pydae Model
# ==========================================
trial = "initial"

def calculate_Cat(pH, K_w, Cit, M_cit, K_a1_cit, K_a2_cit, K_a3_cit):
    H = pow(10, -pH)
    term1 = K_w / H - H
    term2a = K_a1_cit * H ** 2 + 2 * K_a1_cit * K_a2_cit * H + 3 * K_a1_cit * K_a2_cit * K_a3_cit
    term2b = H ** 3 + K_a1_cit * H ** 2 + K_a1_cit * K_a2_cit * H + K_a1_cit * K_a2_cit * K_a3_cit
    Cat = term1 + (Cit / M_cit) * term2a / term2b
    return Cat

def flatten_json(data):
    items = {}
    for k, v in data.items():
        if isinstance(v, dict):
            items.update(flatten_json(v))
        else:
            items[k] = v
    return items

# Load JSON
with open(f"resources/{trial}/pH_T_O2_citric/params.json") as param_file:
    params_json = json.load(param_file)

params = flatten_json(params_json)
initial_conditions = params_json["initial_conditions"]
scales = params_json["scales"]

# Scale initial conditions
initial_conditions_nd = {k: initial_conditions[k] / scales[f"{k}_sc"] for k in initial_conditions.keys()}

# Calculate initial Cations
Cat_0 = calculate_Cat(params['pH_initial'], params['K_w'], initial_conditions['Cit'], params['M_Cit'],
                      params['K_a1_Cit'], params['K_a2_Cit'], params['K_a3_Cit'])
params['Cat'] = Cat_0

# Build the pydae model once
model = build_model_pH_citric(params)

# ==========================================
# 2. Define the PINTS Forward Model
# ==========================================
class CocoaFermentationModel(pints.ForwardModel):
    def __init__(self, base_params, initial_conditions_nd, scales, model_builder):
        self.base_params = base_params.copy()
        self.initial_conditions_nd = initial_conditions_nd
        self.scales = scales
        self.model = model_builder
        
        # The 14 parameters extracted from Table 2
        self.param_names = [
            'K_O2_EtOH', 'K_O2_Ac', 'K_O2_LA', 'A_max', 
            'Y_Q_Glc', 'Y_Q_Fru', 'Y_Q_EtOH', 'Y_Q_LA', 'Q_L', 
            'b_E0', 'b_E1', 'b_AC0', 'b_AC1', 'b_LA'
        ]
        
    def n_parameters(self):
        return len(self.param_names)

    def n_outputs(self):
        return 2

    def simulate(self, parameters, times):
        # Create a dummy return array of NaNs/zeros in case the solver crashes
        dummy_return = np.zeros((len(times), 2))
        
        for i, name in enumerate(self.param_names):
            self.base_params[name] = parameters[i]
            
        try:
            # 1. Try to initialize the model with the proposed parameters
            init_success = self.model.ini(self.base_params, xy_0=self.initial_conditions_nd)
            
            # If pydae returns a status indicating failure, or if it doesn't converge
            if init_success == False:
                return dummy_return
                
            # 2. Run the model forward in time
            self.model.run(times[-1], {})
            self.model.post()
            
            # 3. Extract and scale outputs
            T_sim = np.interp(times, self.model.Time, self.model.get_values("T")) * self.scales["T_sc"]
            pH_sim = np.interp(times, self.model.Time, self.model.get_values("pH")) * self.scales["pH_sc"]
            
            # Check for invalid numerical outputs (Inf or NaN) from the solver
            if np.any(np.isnan(T_sim)) or np.any(np.isnan(pH_sim)):
                return dummy_return
                
            return np.column_stack((T_sim, pH_sim))
            
        except Exception:
            # If the C-code wrapper under pydae throws a hard exception, catch it safely
            return dummy_return
# function must return exactly what you have measured in the real world

# ==========================================
# 3. Load Experimental Data
# ==========================================
# TODO: Replace with your actual experimental data loading logic
experimental_times = np.linspace(0, 168, 20)  
experimental_T = np.random.normal(45, 2, 20)  
experimental_pH = np.random.normal(4.5, 0.5, 20) 


# Generate 20 completely random numbers that average around 45, with a little bit of noise (standard deviation of 2) for T.
# That is entirely fake, mathematically generated "dummy" data for testing purposes. Replace with your actual experimental data loading logic.
# If we have real data, replace the above with something like:
#import pandas as pd

# Load your real industry/lab data
#my_data = pd.read_csv("my_fermentation_trial_data.csv")

# Extract the specific columns
#experimental_times = my_data['Time_hours'].values
#experimental_T = my_data['Temperature_C'].values
#experimental_pH = my_data['pH_level'].values

# Bundle them for PINTS
#experimental_values = np.column_stack((experimental_T, experimental_pH))#

experimental_values = np.column_stack((experimental_T, experimental_pH))

# ==========================================
# 4. Setup PINTS Problem & Likelihood
# ==========================================
forward_model = CocoaFermentationModel(params, initial_conditions_nd, scales, model)
problem = pints.MultiOutputProblem(forward_model, experimental_times, experimental_values)
log_likelihood = pints.GaussianLogLikelihood(problem)

# Nominal values for your 14 physics/biology parameters
nominal_physics_values = np.array([
    0.005, 0.005, 0.005, 1.0, 
    0.1, 0.1, 1.0, 0.1, 0.005, 
    0.001, 10.0, 0.005, 10.0, 0.01
])

# Nominal guesses for the 2 noise parameters (sigma_Temp, sigma_pH)
# (In our dummy data we used a standard deviation of 2.0 for T and 0.5 for pH)
nominal_noise_values = np.array([2.0, 0.5])

# Combine them so PINTS has exactly 16 parameters to work with!
full_nominal_values = np.concatenate((nominal_physics_values, nominal_noise_values))

# Setting generic bounds: 0.1x to 10x the nominal value for all 16 parameters
# Tighter bounds: Allow parameters to vary by only ±5% of nominal values
lower_bounds = full_nominal_values * 0.95
upper_bounds = full_nominal_values * 1.05
log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)

log_posterior = pints.LogPosterior(log_likelihood, log_prior)

# ==========================================
# 5. Run MCMC
# ==========================================
# Generate 3 starting guesses within ±1% of nominal values
np.random.seed(42)
x0 = [full_nominal_values * np.random.uniform(0.99, 1.01, size=16) for _ in range(3)]

# Set the total number of iterations here (e.g., 50 for testing, 5000 for final)
mcmc_iterations = 50  

print(f"Starting MCMC for {mcmc_iterations} iterations...")
mcmc = pints.MCMCController(log_posterior, 3, x0)
mcmc.set_max_iterations(mcmc_iterations)
mcmc.set_log_to_screen(True)

chains = mcmc.run()

# ==========================================
# 6. Diagnostics & Visualization
# ==========================================
print("\nMCMC complete. Generating and saving plots...")

# Create a directory to store the results specifically for this trial
output_dir = f"resources/{trial}/pH_T_O2_citric/mcmc_results"
os.makedirs(output_dir, exist_ok=True)

# Define all 16 parameter names for labeling
param_names_all = forward_model.param_names + ['sigma_Temp', 'sigma_pH']

# ---------------------------------------------------------
# Plot A: Full Trace Plot (All iterations)
# ---------------------------------------------------------
fig_trace, axes_trace = pints.plot.trace(chains, parameter_names=param_names_all)
fig_trace.savefig(os.path.join(output_dir, '01_full_trace.png'), bbox_inches='tight')
plt.close(fig_trace)

# ---------------------------------------------------------
# Plot B: Zoomed Trace Plot (Discarding first 50% as burn-in)
# ---------------------------------------------------------
burn_in = int(mcmc_iterations / 2)
fig_zoomed, axes_zoomed = pints.plot.trace(chains, parameter_names=param_names_all)

for i, ax in enumerate(axes_zoomed[:, 1]):
    # Zoom x-axis to the second half of the run
    ax.set_xlim(burn_in, mcmc_iterations)
    
    # Calculate the min and max values for THIS parameter post-burn-in
    y_min = np.min(chains[:, burn_in:, i])
    y_max = np.max(chains[:, burn_in:, i])
    
    # Add 10% visual padding
    padding = (y_max - y_min) * 0.1
    if padding == 0: padding = 1e-6 # Safety catch in case parameter flatlines
    ax.set_ylim(y_min - padding, y_max + padding)

fig_zoomed.savefig(os.path.join(output_dir, '02_zoomed_trace.png'), bbox_inches='tight')
plt.close(fig_zoomed)

# ---------------------------------------------------------
# Plot C: Pairwise Marginals (Post burn-in)
# ---------------------------------------------------------
print("Generating pairwise marginals (this may take a minute for a 16x16 grid)...")
chains_burned_in = chains[:, burn_in:, :]

# Flatten all 3 chains together to get the best aggregate distribution
flat_chains = chains_burned_in.reshape(-1, 16)

# Generate the massive grid plot
fig_pairwise, axes_pairwise = pints.plot.pairwise(flat_chains, kde=True, parameter_names=param_names_all)
fig_pairwise.savefig(os.path.join(output_dir, '03_pairwise_marginals.png'), bbox_inches='tight')
plt.close(fig_pairwise)

# ---------------------------------------------------------
# Print Final Inferred Parameters
# ---------------------------------------------------------
inferred_means = np.mean(flat_chains, axis=0)

print("\n==========================================")
print("   FINAL INFERENCE RESULTS (Post Burn-in)   ")
print("==========================================")
for i, name in enumerate(param_names_all):
    print(f"{name:>12}: {inferred_means[i]:.5g}")

print(f"\nAll files successfully saved in: {output_dir}")