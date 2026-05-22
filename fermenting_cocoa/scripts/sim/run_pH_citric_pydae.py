import json
import numpy as np
import matplotlib.pyplot as plt

from fermenting_cocoa.scripts import run_model_pH_citric

plt.rcParams['text.usetex'] = True

trial = "initial"

# ==========================================
# Execution Setup
# ==========================================

# KEY PARAMETERS TO VARY FOR FITTING (order of importance):
# Q_L, Cat, Y_Q_EtOH, k_AAB, A_max, b_LA, b_AC0, b_AC1, b_E0, b_E1, mu_max_LAB_Cit
# (note that currently, Cat is determined by the initial conditions, but should be
# a fitting parameter in reality)
# IDEAS FOR PRIORS:
# 0.0001 < Q_L < 0.1, 0.01 < Cat < 0.1, 0.00096 < k_AAB < 0.0096
# New citric acid parameters that may need fitting:
# K_Cit_LAB, Y_Cit_LAB, Y_LA_LAB_Cit, Y_Ac_LAB_Cit
# Other potential parameters for varying
# mu_max_AAB_EtOH, Y_EtOH_Y_LA, Y_LA_Y, Y_Ac_AAB, t_aer


def calculate_Cat(pH, K_w, Cit, M_cit, K_a1_cit, K_a2_cit, K_a3_cit):
    """
    Calculates the concentration of cations (K+, Na+, etc.) at the beginning of the simulation
    given the initial pH and initial citric acid concentration in the solution.
    """
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


param_file = open(f"resources/{trial}/pH_T_O2_citric/params.json")
params_json = json.load(param_file)
params = flatten_json(params_json)

initial_conditions = params_json["initial_conditions"]
initial_conditions["H"] = 10 ** (-initial_conditions["pH"])
short_labels = list(initial_conditions.keys())[1:]

# Cation concentration in pulp (not including H+). We treat this as a constant.
Cat_0 = calculate_Cat(params['pH_initial'], params['K_w'], initial_conditions['Cit'], params['M_Cit'],
                      params['K_a1_Cit'], params['K_a2_Cit'], params['K_a3_Cit'])
# Cat_0 = 0.05
params['Cat'] = Cat_0

t_end = 168

# Run model
model = run_model_pH_citric(params, initial_conditions, t_end)

# Plotting the Results
nrows, ncols = 4, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# fig.suptitle('Cocoa bean fermentation')

labels = ['Glucose', 'Fructose', 'Citric Acid', 'Ethanol', 'Lactic Acid', 'Acetic Acid',
          'Yeast', 'LAB', 'AAB', 'O2', 'Temperature', 'pH']
colors = ['blue', 'orange', 'darkgoldenrod', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'cyan', 'black', 'darkviolet']

times = np.linspace(0, t_end, t_end + 1)
q_hourly = lambda symbol: np.interp(times, model.Time, model.get_values(symbol))

# Calculating ambient temperature
T_e_range = params['T_e_max'] - params['T_e_min']
T_e = T_e_range / 2 * np.cos(np.pi * times / 12) + (params['T_e_max'] - T_e_range / 2)

for i in range(11):
    ax = axs[i//ncols, i%ncols]
    ax.set_title(labels[i])
    ax.set_xlabel('Time [h]')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))

    if labels[i] == 'Temperature':
        ax.plot(times, q_hourly(short_labels[i]), color=colors[i], label='Pulp')
        ax.set_ylabel('°C')
        ax.plot(times, T_e, color=colors[i], label='Ambient', linestyle='dotted', lw=0.5)
        ax.legend()
    else:
        ax.plot(times, q_hourly(short_labels[i]), color=colors[i])
        ax.set_ylabel('mg g(pulp)\\textsuperscript{-1}')

ax_pH = axs[nrows-1, ncols-1]
ax_pH.plot(times, q_hourly("pH"), color=colors[-1])
ax_pH.set_title(labels[-1])
ax_pH.set_xlabel('Time [h]')

fig.savefig(f'resources/{trial}/pH_T_O2_citric/time_traces_pydae.png', bbox_inches='tight', dpi=400)
