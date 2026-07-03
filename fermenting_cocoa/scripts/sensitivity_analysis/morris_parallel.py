import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Creating the noisy data
n_time_rec = 30
times_rec = np.linspace(0, t_end, n_time_rec, dtype=int)

noise_scale = 0.01
std_devs = {short_label: scales[f"{short_label}_sc"] * noise_scale for short_label in short_labels}
noise = {short_label: np.random.normal(0, std_devs[short_label], n_time_rec) for short_label in short_labels}
noisy_df = pd.DataFrame(columns=short_labels)
for short_label in short_labels:
    noisy_df[short_label] = default_params_df[short_label].to_numpy()[times_rec] + noise[short_label]

# Saving the noisy data in a .csv file
noisy_df.to_csv(f"{res_dir}/noisy_data.csv")
    
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

fixed_params_morris = {fixed_param: params[fixed_param] for fixed_param in fixed_param_keys}
var_params_morris = {param: params[param] for param in params if param not in fixed_param_keys}

# We log-transform the problem to help with convergence
param_info_morris = {param_name: "log" for param_name in var_params_morris}
problem_morris = Problem(model, initial_conditions_nd, t_end, times_rec, short_labels, scales, noisy_df, param_info_morris, fixed_params_morris)
log_var_params = {param: np.log10(params[param]) for param in var_params_morris}
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

# Now we can setup the Morris screening
sort_dict = lambda _dict: {k: v for k, v in sorted(_dict.items(), key=lambda item: item[1], reverse=True)}

meta_info = {
    "num_vars": 60,
    "names": list(param_info_morris.keys()),
    "bounds": [bound_list for bound_list in bounds.values()]
}

from SALib.sample.morris import sample
from SALib.analyze.morris import analyze

# Here we run Morris a number of times and compare rankings at the end
n_morris = 8
N = 100
param_values_all = []
for j in range(n_morris):
    param_values = sample(meta_info, N=N, num_levels=6, local_optimization=True)
    param_values_all.append(param_values)


# Below is the parallelised code
Y_all = []

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
                      cf["noisy_df"], cf["param_info_morris"], cf["fixed_params_morris"])


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
    for k in range(n_morris):
        param_values = param_values_all[k]
        start = time.perf_counter()
        n_its = len(param_values)

        problem_config = {
            "params": params,
            "initial_conditions_nd": initial_conditions_nd,
            "t_end": t_end,
            "times_rec": times_rec,
            "short_labels": short_labels,
            "scales": scales,
            "noisy_df": noisy_df,
            "param_info_morris": param_info_morris,
            "fixed_params_morris": fixed_params_morris
        }

        n_workers = 16

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(problem_config,),
        ) as ex:
            futures = [ex.submit(eval_sse, (i, param_values[i])) for i in range(n_its)]

            Y_list = [None] * n_its
            for j, fut in enumerate(as_completed(futures), 1):
                i, y = fut.result()
                Y_list[i] = y

                if j % 100 == 0:
                    elapsed = time.perf_counter() - start
                    hours, remainder = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(int(remainder), 60)
                    print(f"Run {k} It {j} ~ {hours:02}:{minutes:02}:{seconds:02}")

        Y = np.array(Y_list)
        Y_all.append(Y)

# Now we perform the analysis and record data to a .csv file
Si_list = []
for j in range(n_morris):
    Y = Y_all[j]
    param_values = param_values_all[j]
    Si = analyze(meta_info, param_values, Y, conf_level=0.95, print_to_console=False)
    Si_list.append(Si)

morris_df_all = pd.DataFrame(columns=["name"] + [f"mu_star_{j}" for j in range(n_morris)] + [f"sigma_{j}" for j in range(n_morris)])

for j in range(n_morris):
    Si = Si_list[j]
    mu_star, sigma = Si["mu_star"], Si["sigma"]
    mu_star_dict = {meta_info["names"][i]: mu_star[i] for i in range(len(mu_star))}
    sigma_dict = {meta_info["names"][i]: sigma[i] for i in range(len(sigma))}

    mu_star_sorted = sort_dict(mu_star_dict)
    sigma_sorted = sort_dict(sigma_dict)
    labels_morris = list(mu_star_sorted.keys())
    if j == 0:
        morris_df_all["name"] = labels_morris
    print("mu_star")
    mu_star_ranking = [0] * len(labels_morris)
    for i in range(len(mu_star_sorted)):
        label = list(mu_star_sorted.keys())[i]
        label_pos = list(morris_df_all["name"]).index(label)
        mu_star_ranking[label_pos] = i + 1
        print(f"{i + 1}. {label}: {mu_star_sorted[label]}")
    print("-"*80)
    print("sigma")
    sigma_ranking = [0] * len(labels_morris)
    for i in range(len(sigma_sorted)):
        label = list(sigma_sorted.keys())[i]
        label_pos = list(morris_df_all["name"]).index(label)
        sigma_ranking[label_pos] = i + 1
        print(f"{i + 1}. {label}: {sigma_sorted[label]}")

    sigma_sorted_by_mu_star = {label: sigma_sorted[label] for label in labels_morris}
    morris_df = pd.DataFrame({"name": labels_morris,
                              "mu_star": list(mu_star_sorted.values()),
                              "sigma": list(sigma_sorted_by_mu_star.values())})
    morris_df.to_csv(f"{res_dir}/morris/results_{j}.csv", index=False)
    morris_df_all[f"mu_star_{j}"] = mu_star_ranking
    morris_df_all[f"sigma_{j}"] = sigma_ranking
    print("="*80)
morris_df_all.to_csv(f"{res_dir}/morris/rankings.csv", index=False)

# Plotting

from SALib.plotting.morris import horizontal_bar_plot, covariance_plot

for j in range(n_morris):
    Si = Si_list[j]
    fig, ax = plt.subplots(1, 1, figsize=(6, 20))
    p = horizontal_bar_plot(ax, Si)
    # ax.set_xscale("log")
    fig.savefig(f"{res_dir}/morris/bar_{j}.png", dpi=400, bbox_inches="tight")

for j in range(n_morris):
    Si = Si_list[j]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    p = covariance_plot(ax, Si)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    mu_star = np.linspace(0, 8000, 10)
    gradient = 6.5
    sigma_obs = mu_star * gradient
    ax.plot(mu_star, sigma_obs, color="darkviolet", label=r"$\sigma / \mu^{*}=$" + f" ${gradient}$")
    ax.legend()
    fig.savefig(f"{res_dir}/morris/covariance_{j}.png", dpi=400, bbox_inches="tight")
