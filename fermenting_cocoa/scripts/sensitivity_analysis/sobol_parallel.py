import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from fermenting_cocoa.scripts import build_model_pH_citric
from fermenting_cocoa.scripts import run_model_pH_citric


plt.rcParams['text.usetex'] = True
trial = "initial"
res_dir = f"../../resources/{trial}/pH_T_O2_citric"


def flatten_json(data):
    items = {}
    for k, v in data.items():
        if isinstance(v, dict):
            items.update(flatten_json(v))
        else:
            items[k] = v
    return items


# Get default parameter values
param_file = open(f"{res_dir}/params.json")
params_json = json.load(param_file)
params = flatten_json(params_json)

initial_conditions = params_json["initial_conditions"]

scales = params_json["scales"]
short_labels = ["Glc", "Fru", "Cit", "EtOH", "LA", "Ac", "Y", "LAB", "AAB", "O2", "T", "pH"]

# Scaling initial conditions
initial_conditions_nd = {k: initial_conditions[k] / scales[f"{k}_sc"]
                         for k in initial_conditions.keys()}

# Setting up the model
t_end = 168 # 7 days = 168 hours
verbose = False

# Plotting
labels = ['Glucose', 'Fructose', 'Citric Acid', 'Ethanol', 'Lactic Acid', 'Acetic Acid',
          'Yeast', 'LAB', 'AAB', 'O2', 'Temperature', 'pH']
colours = ['blue', 'orange', 'darkgoldenrod', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'cyan', 'black', 'darkviolet']
times = np.linspace(0, t_end, t_end + 1)


def ambient_T(_times: np.array, T_e_min: float, T_e_max: float):
    """Calculates the ambient temperature profile given minimum
    and maximum values.
    """
    T_e_range = T_e_max - T_e_min
    return T_e_range / 2 * np.cos(np.pi * _times / 12) + (params['T_e_max'] - T_e_range / 2)


T_e = ambient_T(times, params['T_e_min'], params['T_e_max'])


def extract_data_from_model(_times, _model, _short_labels, _scales):
    """Creates a pandas dataframe from the _model data.
    """
    # The below is a helper function for converting pydae's output (with lots of timesteps) into an hourly array
    q_hourly = lambda symbol: np.interp(_times, _model.Time, _model.get_values(symbol)) * _scales[f"{symbol}_sc"]
    df = pd.DataFrame(columns=_short_labels)
    for label in _short_labels:
        df[label] = q_hourly(label)
    return df


def plot_all_profiles(_times, _df, _colours, _short_labels, _labels, _scales, _T_e,
                      _fig=None, _axs=None, _linestyle="solid"):
    """Plots the profiles for a given set of data and timepoints.
    """
    nrows, ncols = 4, 3
    if _fig is None and _axs is None:
        _fig, _axs = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for i in range(11):
        ax = _axs[i//ncols, i%ncols]
        ax.set_title(_labels[i])
        ax.set_xlabel('Time [h]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))

        if labels[i] == 'Temperature':
            ax.plot(_times, _df[_short_labels[i]], color=_colours[i], label='Pulp', linestyle=_linestyle)
            ax.set_ylabel('°C')
            ax.plot(_times, _T_e, color=_colours[i], label='Ambient', linestyle='dotted', lw=0.5)
            ax.legend()
        else:
            ax.plot(_times, _df[_short_labels[i]], color=_colours[i], linestyle=_linestyle)
            ax.set_ylabel('mg g(pulp)\\textsuperscript{-1}')

    ax_pH = _axs[nrows-1, ncols-1]
    ax_pH.plot(_times, _df["pH"], color=_colours[-1], linestyle=_linestyle)
    ax_pH.set_title(labels[-1])
    ax_pH.set_xlabel('Time [h]')
    return _fig, _axs


# Running model to generate noisy data
model = build_model_pH_citric(params)
model = run_model_pH_citric(model, params, initial_conditions_nd, t_end)
default_params_df = extract_data_from_model(times, model, short_labels, scales)

n_time_rec = 30
times_rec = np.linspace(0, t_end, n_time_rec, dtype=int)

# Reading in the noisy data
noisy_df = pd.read_csv(f"{res_dir}/noisy_data.csv", index_col=0)

# Creating the problem and specifying the error measure


class Problem:
    """A class containing both the model and the recorded data.
    """

    def __init__(self, _model, _initial_conditions_nd, _t_end, _times_rec, _short_labels, _scales, _data_df, _param_info, _fixed_params):
        self._model = _model
        self._model_out = None
        self._initial_conditions_nd = _initial_conditions_nd
        self._t_end = _t_end
        self._times_rec = _times_rec
        self._short_labels = _short_labels
        self._scales = _scales
        self._data_df = _data_df
        self._param_info = _param_info # dict keyed by param name with value "lin" if not log-transformed and "log" if log-transformed
        self._fixed_params = _fixed_params # dict for all the fixed parameter values of the model

    def sse(self, _param_arr, Dt=1e-2):
        """Calculates the sum of squared errors between the model simulated with specific parameters and the recorded (or
        synthetic) data. The _param_arr argument is a numpy array of parameters in the same order as in the _param_info dict.
        Some parameters may be log-transformed so we must transform them back in the DAE system.
        """
        _names = list(self._param_info.keys())
        _var_params = {_names[i]: _param_arr[i] if self._param_info[_names[i]] == "lin" else 10 ** _param_arr[i]
                       for i in range(len(_param_arr))}
        _params = {**self._fixed_params, **_var_params}
        self._model_out = run_model_pH_citric(self._model, _params, self._initial_conditions_nd, self._t_end, Dt)
        _model_df = extract_data_from_model(self._times_rec, self._model_out, self._short_labels, self._scales)
        sse = 0
        for short_label in self._short_labels:
            if short_label in self._data_df:
                scale = self._scales[f"{short_label}_sc"]
                error = np.sum((_model_df[short_label] - self._data_df[short_label]) ** 2) / (scale ** 2)
                sse += error
        return sse


# Selecting the parameters we wish to fix and vary
fixed_param_keys = ["T_min_Y", "T_opt_Y", "T_max_Y", "T_min_LAB", "T_opt_LAB", "T_max_LAB", "T_min_AAB", "T_opt_AAB", "T_max_AAB",
                    "T_e_min", "T_e_max", 
                    "C_air", 
                    "Delta_H_EtOH", "Delta_H_Ac", "R",
                    "M_EtOH", "M_LA", "M_Ac", "M_Cit",
                    "pH_initial", "K_w", "K_a1_Cit", "K_a2_Cit", "K_a3_Cit", "K_a_Ac", "K_a_LA", "Cat",
                    "pH_min_Y", "pH_opt_Y", "pH_max_Y", "pH_min_LAB", "pH_opt_LAB", "pH_max_LAB", "pH_min_AAB", "pH_opt_AAB", "pH_max_AAB",
                    "tau", "Glc", "Fru", "Cit", "EtOH", "LA", "Ac", "Y", "LAB", "AAB", "O2", "T", "pH", # initial conditions
                    "tau_sc", "Glc_sc", "Fru_sc", "Cit_sc", "EtOH_sc", "LA_sc", "Ac_sc", "Y_sc", "LAB_sc", "AAB_sc", "O2_sc", "T_sc", "pH_sc",
                    "alpha_solver"] # from pydae

# Reading in Morris data and adding new fixed parameters
sort_dict = lambda _dict: {k: v for k, v in sorted(_dict.items(), key=lambda item: item[1], reverse=True)}

morris_df_all = pd.read_csv(f"{res_dir}/morris/rankings.csv")
k = 15
n_morris = int((len(morris_df_all.columns) - 1) / 2)

count = sum([(morris_df_all[f"mu_star_{j}"] <= k).astype(int)
             for j in range(n_morris)])
count_dict = {list(morris_df_all["name"])[i]: list(count)[i] for i in range(len(count))}
count_dict = sort_dict(count_dict)
for i in range(len(count_dict)):
    name = list(count_dict.keys())[i]
    print(f"{i + 1}. {name}: {count_dict[name]}")
count_df = pd.DataFrame({"name": count_dict.keys(), "count": count_dict.values()})
count_df.to_csv(f"{res_dir}/morris/counts.csv")
sobol_labels = list(count_dict.keys())[:15]
if "k_AAB" not in sobol_labels:
    sobol_labels[14] = "k_AAB" # Have reason to believe that this parameter is very important, so we would like to keep it in
print(sobol_labels)

# Here we start the setup for the Sobol analysis

# Adding more fixed parameters from Morris to the dict
fixed_params_morris = {fixed_param: params[fixed_param] for fixed_param in fixed_param_keys}
fixed_param_keys = list(fixed_params_morris.keys()) + [label for label in morris_df_all["name"] if label not in sobol_labels] 
fixed_params_sobol = {fixed_param: params[fixed_param] for fixed_param in fixed_param_keys}
var_params_sobol = {param: params[param] for param in params if param not in fixed_param_keys}

# We log-transform the problem to help with convergence
param_info_sobol = {param_name: "log" for param_name in var_params_sobol}
problem_sobol = Problem(model, initial_conditions_nd, t_end, times_rec, short_labels, scales, noisy_df, param_info_sobol, fixed_params_sobol)
log_var_params = {param: np.log10(params[param]) for param in var_params_sobol}
param_arr = np.array(list(log_var_params.values()))

# Parameter bounds
bounds = {"mu_max_Y_Glc": [-1, -0.1], "mu_max_Y_Fru": [-2, -0.1], "mu_max_Y_LA": [-2, -0.1], "mu_max_LAB_Glc": [-3, -0.5],
          "mu_max_LAB_Fru": [-2, -0.4], "mu_max_AAB_EtOH": [-1, -0.1], "mu_max_AAB_LA": [-3, -1.2], "mu_max_AAB_Ac": [-3, -0.2],
          "K_Glc_Y": [-4, 1.9], "K_Fru_Y": [0.9, 2.1], "K_LA_Y": [0.5, 1.9], "K_Glc_LAB": [-0.5, 2.1], "K_Fru_LAB": [0.8, 1.9],
          "K_EtOH_AAB": [0.6, 1.5], "K_LA_AAB": [2, 3.7], "K_Ac_AAB": [-1, 1.4],
          "k_Y": [-1.8, -1.2], "k_LAB": [-2.6, -1.8], "k_AAB": [-3.5, -1.7],
          "Y_Glc_Y": [1, 1.9], "Y_Glc_LAB": [0.5, 1.9], "Y_Fru_Y": [1, 1.9], "Y_Fru_LAB": [1, 2], "Y_EtOH_Y_Glc": [-0.5, 1.3], "Y_EtOH_Y_Fru": [-0.5, 1.2],
          "Y_EtOH_LAB_Glc": [0, 1.4], "Y_EtOH_LAB_Fru": [-0.5, 1.4], "Y_EtOH_AAB": [2, 3.5], "Y_LA_LAB_Glc": [0, 1.2], "Y_LA_LAB_Fru": [0.2, 1.2],
          "Y_LA_AAB": [0, 3.7], "Y_Ac_LAB_Glc": [-0.5, 0.6], "Y_Ac_LAB_Fru": [-0.5, 1], "Y_EtOH_Y_LA": [-0.5, 1.3], "Y_Ac_AAB_EtOH": [1, 2.4],
          "Y_Ac_AAB_LA": [2, 3.5], "Y_Ac_Y_Glc": [-0.5, 0.3], "Y_Ac_Y_Fru": [-0.5, 0.4], "Y_LA_Y": [0, 1], "Y_Ac_AAB": [2, 3.5],
          # Below here are new parameters. Therefore, the bounds are more educated guesses
          "Q_L": [-4, 0], "Y_Q_Glc": [-2, 0], "Y_Q_Fru": [-2, 0], "Y_Q_EtOH": [-1, 1], "Y_Q_LA": [-2, 0],
          "K_O2_EtOH": [-4, -2.1], "K_O2_LA": [-4, -2.1], "K_O2_Ac": [-4, -2.1], "A_max": [-1, 1], "t_aer": [1.4, 1.9],
          "b_LA": [-3, -1], "b_E0": [-4, -2], "b_E1": [0, 2], "b_AC0": [-4, -2], "b_AC1": [0, 2],
          "mu_max_LAB_Cit": [-2, -0.1], "K_Cit_LAB": [0.3, 0.9], "Y_Cit_LAB": [0, 1.9], "Y_LA_LAB_Cit": [0, 1.2], "Y_Ac_LAB_Cit": [-0.5, 0.8]}

# Now we can setup the Sobol analysis

meta_info = {
    "num_vars": 15,
    "names": sobol_labels,
    "bounds": [bounds[label] for label in sobol_labels]
}

from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze

# Here we run Sobol for 2^15 iterations
N_sobol = 2**15
param_values = sample(meta_info, N=N_sobol, calc_second_order=True)


# Below is the parallelised code
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

PROBLEM = None
MODEL = None

def init_worker(cf):
    """
    Build the solver/model inside each subprocess.
    problem_config should contain only plain Python objects
    (dicts, lists, floats, ints, strings).
    """
    global MODEL
    global PROBLEM
    MODEL = build_model_pH_citric(params)
    PROBLEM = Problem(MODEL, cf["initial_conditions_nd"], cf["t_end"], cf["times_rec"], cf["short_labels"], cf["scales"], 
                      cf["noisy_df"], cf["param_info_sobol"], cf["fixed_params_sobol"])

def eval_sse(args):
    i, params = args
    global PROBLEM

    try:
        y = PROBLEM.sse(params)
    except RuntimeError:
        print(f"Convergence error at It {i}, increasing resolution...")
        y = PROBLEM.sse(params, Dt=1e-3)

    return i, y

if __name__ == "__main__":
    start = time.perf_counter()
    n_its = len(param_values)
    Y_list = [None] * n_its

    problem_config = {
        "params": params,
        "initial_conditions_nd": initial_conditions_nd,
        "t_end": t_end,
        "times_rec": times_rec,
        "short_labels": short_labels,
        "scales": scales,
        "noisy_df": noisy_df,
        "param_info_sobol": param_info_sobol,
        "fixed_params_sobol": fixed_params_sobol
    }

    n_workers = 16

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(problem_config,),
    ) as ex:
        futures = [ex.submit(eval_sse, (i, param_values[i])) for i in range(n_its)]

        for j, fut in enumerate(as_completed(futures), 1):
            i, y = fut.result()
            Y_list[i] = y

            if j % 1000 == 0:
                elapsed = time.perf_counter() - start
                hours, remainder = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(int(remainder), 60)
                print(f"It {j} ~ {hours:02}:{minutes:02}:{seconds:02}")

    Y = np.array(Y_list)

# Below is the analysis
Si = analyze(meta_info, Y, calc_second_order=True, conf_level=0.95, print_to_console=False)

# Collecting the results into various dataframes and dictionaries
S1_dict = {sobol_labels[i]: Si["S1"][i] for i in range(meta_info["num_vars"])}
S1_conf_dict = {sobol_labels[i]: Si["S1_conf"][i] for i in range(meta_info["num_vars"])}
ST_dict = {sobol_labels[i]: Si["ST"][i] for i in range(meta_info["num_vars"])}
ST_conf_dict = {sobol_labels[i]: Si["ST_conf"][i] for i in range(meta_info["num_vars"])}

# Second order indices for later analysis
S2_arr = Si["S2"]
S2_conf_arr = Si["S2_conf"]
S2_df = pd.DataFrame(S2_arr, index=sobol_labels, columns=sobol_labels)
S2_df.to_csv(f"{res_dir}/sobol/S2.csv")
S2_conf_df = pd.DataFrame(S2_conf_arr / S2_arr, index=sobol_labels, columns=sobol_labels)
S2_conf_df.to_csv(f"{res_dir}/sobol/S2_conf_normalised.csv")

S1_sorted = sort_dict(S1_dict)
ST_sorted = sort_dict(ST_dict)
sorted_sobol_labels = ST_sorted.keys()

str_x = lambda x: (np.array([np.format_float_positional(x[i], precision=3, unique=False, fractional=False, trim='k')
                            for i in range(len(x))]) if isinstance(x, np.ndarray) 
                   else np.format_float_positional(x, precision=3, unique=False, fractional=False, trim='k'))

S1_sorted_by_ST = {label: S1_sorted[label] for label in sorted_sobol_labels}
ST_conf_sorted_by_ST = {label: ST_conf_dict[label] for label in sorted_sobol_labels}
ST_arr, S1_arr, ST_conf_arr = np.array(list(ST_sorted.values())), np.array(list(S1_sorted_by_ST.values())), np.array(list(ST_conf_sorted_by_ST.values()))
sobol_df = pd.DataFrame({"name": sorted_sobol_labels, "S1": str_x(S1_arr), "ST": str_x(ST_arr), "S1_conf": str(S1_conf_arr), "ST_conf": str(ST_conf_arr),
                         "ST - S1": str_x(ST_arr - S1_arr), "ST_conf / ST": str_x(ST_conf_arr / ST_arr)})

sobol_df.to_csv(f"{res_dir}/sobol/rankings.csv")

# Printing results
print("="*80)
print("S1 ranking:")
print("-"*80)
for i in range(len(S1_dict)):
    name = list(S1_sorted.keys())[i]
    print(f"{i + 1}. {name}: {str_x(S1_sorted[name])}")
print("="*80)
print("ST ranking:")
print("-"*80)
for i in range(len(ST_dict)):
    name = list(ST_sorted.keys())[i]
    print(f"{i + 1}. {name}: {str_x(ST_sorted[name])}")
print("="*80)

# Now we create some diagnostic plots

# Here we plot the S1 indices against the ST ones
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.scatter(S1_arr, ST_arr, label="Sobol data")
ax.plot(S1_arr, S1_arr, label="$S_T=S_1$", color="black", linestyle="dashed")
for i, name in enumerate(sorted_sobol_labels):
    ax.annotate(name, (S1_arr[i], ST_arr[i]))
ax.legend()
ax.set_xlabel("$S_1$")
ax.set_ylabel("$S_T$")
fig.savefig(f"{res_dir}/sobol/S1_ST.png", bbox_inches="tight", dpi=400)

# Now we plot the bar charts for each parameter
S1_conf_sorted_by_S1 = [S1_conf_dict[name] for name in S1_sorted.keys()]
ST_conf_sorted_by_ST = [ST_conf_dict[name] for name in ST_sorted.keys()]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax_S1, ax_ST = axs[0], axs[1]
ax_S1.barh(list(S1_sorted.keys()), list(S1_sorted.values()), xerr=S1_conf_sorted_by_S1, align='center')
ax_S1.yaxis.set_inverted(True)
ax_S1.set_xlabel("$S_1$")
ax_ST.barh(list(ST_sorted.keys()), list(ST_sorted.values()), xerr=ST_conf_sorted_by_ST, align='center')
ax_ST.yaxis.set_inverted(True)
ax_ST.set_xlabel("$S_T$")
fig.savefig(f"{res_dir}/sobol/bar.png", bbox_inches="tight", dpi=400)

# Contingency table-like plot for the S2 indices
plt.figure(figsize=(8, 6))
sn.heatmap(S2_df, annot=True)
plt.savefig(f"{res_dir}/sobol/S2.png")
plt.figure(figsize=(8, 6))
sn.heatmap(S2_conf_df, annot=True)
plt.savefig(f"{res_dir}/sobol/S2_conf_normalised.png")
