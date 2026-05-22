import json
import numpy as np
import matplotlib.pyplot as plt

from fermenting_cocoa.scripts import run_model_pH

import time
plt.rcParams['text.usetex'] = True

trial = "initial"


# ==========================================
# Execution Setup
# ==========================================


def calculate_Cit(pH, K_w, M_cit, K_a1_cit, K_a2_cit, K_a3_cit):
    """
    Calculates the concentration of citric acid at the beginning of the simulation
    given the initial pH of the solution.
    """
    H = pow(10, -pH)
    term1 = H - K_w / H
    term2 = H ** 3 + K_a1_cit * H ** 2 + K_a1_cit * K_a2_cit * H + K_a1_cit * K_a2_cit * K_a3_cit
    denominator = K_a1_cit * H ** 2 + 2 * K_a1_cit * K_a2_cit * H + 3 * K_a1_cit * K_a2_cit * K_a3_cit
    Cit = term1 * term2 / denominator * M_cit
    return Cit


def flatten_json(data):
    items = {}
    for k, v in data.items():
        if isinstance(v, dict):
            items.update(flatten_json(v))
        else:
            items[k] = v
    return items


param_file = open(f"resources/{trial}/pH_T_O2/params.json")
params_json = json.load(param_file)
params = flatten_json(params_json)

# Citric acid concentration (calculated from initial conditions)
Cit0 = calculate_Cit(params['pH'], params['K_w'], params['M_Cit'], params['K_a1_Cit'],
                     params['K_a2_Cit'], params['K_a3_Cit'])
params["Cit_0"] = Cit0

initial_conditions = params_json["initial_conditions"]
initial_conditions["H"] = 10 ** (-initial_conditions["pH"])
short_labels = list(initial_conditions.keys())[1:]

t_end = 168

model = run_model_pH(params, initial_conditions, t_end)

# Plotting the Results
nrows, ncols = 4, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# fig.suptitle('Cocoa bean fermentation')

labels = ['Glucose', 'Fructose', 'Ethanol', 'Lactic Acid', 'Acetic Acid',
          'Yeast', 'LAB', 'AAB', 'O2', 'Temperature', 'pH', 'Citric Acid']
colors = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'cyan', 'black', 'darkviolet', 'darkgoldenrod']

times = np.linspace(0, t_end, t_end + 1)
q_hourly = lambda symbol: np.interp(times, model.Time, model.get_values(symbol))

# Calculating ambient temperature
T_e_range = params['T_e_max'] - params['T_e_min']
T_e = T_e_range / 2 * np.cos(np.pi * times / 12) + (params['T_e_max'] - T_e_range / 2)

for i in range(10):
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

ax_pH = axs[nrows-1, ncols-2]
ax_pH.plot(times, q_hourly("pH"), color=colors[-2])
ax_pH.set_title(labels[-2])
ax_pH.set_xlabel('Time [h]')

ax_Cit = axs[nrows-1, ncols-1]
ax_Cit.plot(times, Cit0 * np.ones(t_end + 1), color=colors[-1])
ax_Cit.set_title(labels[-1])
ax_Cit.set_xlabel('Time [h]')
ax_Cit.set_ylabel('mg g(pulp)\\textsuperscript{-1}')

fig.savefig(f'resources/{trial}/pH_T_O2/time_traces_pydae.png', bbox_inches='tight', dpi=400)
