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
trial = "camu2007"
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
                      _fig=None, _axs=None, _linestyle="solid", _style="line"):
    """Plots the profiles for a given set of data and timepoints.
    """
    nrows, ncols = 4, 3
    if _fig is None and _axs is None:
        _fig, _axs = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for i in range(11):
        if _short_labels[i] not in _df:
            continue
        ax = _axs[i//ncols, i%ncols]
        ax.set_title(_labels[i])
        ax.set_xlabel('Time [h]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))

        if labels[i] == 'Temperature':
            if _style == "scatter":
                ax.scatter(_times, _df[_short_labels[i]], color=_colours[i], label='Pulp', marker='x')
            else:
                ax.plot(_times, _df[_short_labels[i]], color=_colours[i], label='Pulp', linestyle=_linestyle)
            ax.set_ylabel('°C')
            ax.plot(_times, _T_e, color=_colours[i], label='Ambient', linestyle='dotted', lw=0.5)
            ax.legend()
        else:
            if _style == "scatter":
                ax.scatter(_times, _df[_short_labels[i]], color=_colours[i], marker='x')
            else:
                ax.plot(_times, _df[_short_labels[i]], color=_colours[i], linestyle=_linestyle)
            ax.set_ylabel('mg g(pulp)\\textsuperscript{-1}')

    ax_pH = _axs[nrows-1, ncols-1]
    if _style == "scatter":
        ax_pH.scatter(_times, _df["pH"], color=_colours[-1], marker='x')
    else:
        ax_pH.plot(_times, _df["pH"], color=_colours[-1], linestyle=_linestyle)
    ax_pH.set_title(labels[-1])
    ax_pH.set_xlabel('Time [h]')
    return _fig, _axs


# Running model to generate noisy data
model = build_model_pH_citric(params)
model = run_model_pH_citric(model, params, initial_conditions_nd, t_end)
default_params_df = extract_data_from_model(times, model, short_labels, scales)

# # Creating the noisy data
# n_time_rec = 30
# times_rec = np.linspace(0, t_end, n_time_rec, dtype=int)

# noise_scale = 0.01
# std_devs = {short_label: scales[f"{short_label}_sc"] * noise_scale for short_label in short_labels}
# noise = {short_label: np.random.normal(0, std_devs[short_label], n_time_rec) for short_label in short_labels}
# noisy_df = pd.DataFrame(columns=short_labels)
# for short_label in short_labels:
#     noisy_df[short_label] = default_params_df[short_label].to_numpy()[times_rec] + noise[short_label]

# # Saving the noisy data in a .csv file
# noisy_df.to_csv(f"{res_dir}/noisy_data.csv")

# Reading in the experimental data
data_name = "camu_2007_data"
experimental_data_df = pd.read_csv(f"{res_dir}/experimental_data/{data_name}.csv")
times_rec = experimental_data_df["Time"].to_numpy()

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
        self._mins = {label: np.min(_data_df[label]) for label in _short_labels if label in _data_df}
        self._maxes = {label: np.max(_data_df[label]) for label in _short_labels if label in _data_df}
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
                q_range = self._maxes[short_label] - self._mins[short_label]
                error = np.sum((_model_df[short_label] - self._data_df[short_label]) ** 2) / (q_range ** 2)
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
problem_morris = Problem(model, initial_conditions_nd, t_end, times_rec, short_labels, scales, experimental_data_df, param_info_morris, fixed_params_morris)
log_var_params = {param: np.log10(params[param]) for param in var_params_morris}
param_arr = np.array(list(log_var_params.values()))

# Parameter bounds
bounds_file = open(f"{res_dir}/bounds.json")
bounds_tiered = json.load(bounds_file)
bounds = flatten_json(bounds_tiered)

# Now we can setup the Morris screening
sort_dict = lambda _dict: {k: v for k, v in sorted(_dict.items(), key=lambda item: item[1], reverse=True)}

meta_info = {
    "num_vars": 63,
    "names": list(param_info_morris.keys()),
    "bounds": [bounds[name] for name in param_info_morris.keys()]
}

from SALib.sample.morris import sample
from SALib.analyze.morris import analyze

# Here we run Morris a number of times and compare rankings at the end
n_morris = 16
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
                      cf["data_df"], cf["param_info_morris"], cf["fixed_params_morris"])


def eval_sse(args):
    i, params = args
    global PROBLEM

    try:
        y = PROBLEM.sse(params)
    except RuntimeError:
        print(f"Convergence error at It {i}, increasing resolution...")
        try:
            y = PROBLEM.sse(params, Dt=1e-3)
        except RuntimeError:
            y = 1e4

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
            "data_df": experimental_data_df,
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

# Finding the best results
num_best_its = 5
for k in range(num_best_its):
    fig, axs = plot_all_profiles(times_rec, experimental_data_df, colours, short_labels, labels, scales, experimental_data_df["T_ext"],
                                 _style="scatter")
    j = np.argmin(Y_all)
    n_its = len(Y_all[0])
    print(problem_morris.sse(param_values_all[j//n_its][j%n_its], Dt=1e-3))
    output_df = extract_data_from_model(times, problem_morris._model_out, short_labels, scales)
    fig, axs = plot_all_profiles(times, output_df, colours, short_labels, labels, scales, T_e, _fig=fig, _axs=axs)
    fig.savefig(f"{res_dir}/morris/morris_traces_{k}.png")
    Y_all[j//n_its][j%n_its] = 1e6

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
