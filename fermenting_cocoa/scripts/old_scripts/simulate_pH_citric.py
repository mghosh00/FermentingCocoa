import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
plt.rcParams['text.usetex'] = True


def calculate_mu_T(T, mu_opt, T_min, T_opt, T_max):
    """
    Calculates the temperature-adjusted specific growth rate using the
    cardinal model.
    """
    if T < T_min or T > T_max:
        return 0.0

    numerator = (T - T_max) * ((T - T_min) ** 2)

    term1 = (T_opt - T_min) * (T - T_opt)
    term2 = (T_opt - T_max) * (T_opt + T_min - 2 * T)
    denominator = (T_opt - T_min) * (term1 - term2)

    if denominator == 0:
        return 0.0

    return mu_opt * (numerator / denominator)


def calculate_mu_pH(pH, pH_min, pH_opt, pH_max):
    """
    Calculates the effect pH has on the reaction rate using the cardinal
    model.
    """
    if pH < pH_min or pH > pH_max:
        return 0.0

    numerator = (pH - pH_min) * (pH - pH_max)
    term1 = (pH - pH_min) * (pH - pH_max)
    term2 = (pH - pH_opt) ** 2
    denominator = term1 - term2

    if denominator == 0:
        return 0.0
    return max(0.0, numerator / denominator) # Prevent negative scaling


def calculate_Cat(pH, K_w, Cit, M_cit, K_a1_cit, K_a2_cit, K_a3_cit):
    """
    Calculates the concentration of cations (K+, Na+, etc.) at the beginning of the simulation
    given the initial pH and initial citric acid concentration in the solution.
    """
    H = 10 ** (-pH)
    term1 = K_w / H - H
    term2a = K_a1_cit * H ** 2 + 2 * K_a1_cit * K_a2_cit * H + 3 * K_a1_cit * K_a2_cit * K_a3_cit
    term2b = H ** 3 + K_a1_cit * H ** 2 + K_a1_cit * K_a2_cit * H + K_a1_cit * K_a2_cit * K_a3_cit
    Cat = term1 + (Cit / M_cit) * term2a / term2b
    return Cat


def solve_H(x, *args):
    H = float(x[0])
    # Prevent negative H guesses from breaking the math
    H = max(H, 1e-14)

    K_w, Cit, M_Cit, K_a1_cit, K_a2_cit, K_a3_cit = args[0], args[1], args[2], args[3], args[4], args[5]
    Ac, M_Ac, K_a_Ac = args[6], args[7], args[8]
    LA, M_LA, K_a_LA = args[9], args[10], args[11]
    Cat = args[12] # Cation concentration. This is a fitting parameter determined at the beginning of the simulation.

    term1 = K_w / H - H - Cat
    term2 = ((Cit / M_Cit) * (K_a1_cit * H ** 2 + 2 * K_a1_cit * K_a2_cit * H + 3 * K_a1_cit * K_a2_cit * K_a3_cit) /
             (H ** 3 + K_a1_cit * H ** 2 + K_a1_cit * K_a2_cit * H + K_a1_cit * K_a2_cit * K_a3_cit))

    term3 = (Ac / M_Ac) * K_a_Ac / (H + K_a_Ac)
    term4 = (LA / M_LA) * K_a_LA / (H + K_a_LA)

    eq = term1 + term2 + term3 + term4
    return np.array([eq], dtype=float)


def full_fermentation_derivatives(t, states, params):
    """
    Computes the derivatives for the FULL cocoa bean fermentation model,
    including M1-M5, Oxygen dynamics and Temperature dynamics.
    """
    Glc, Fru, Cit, EtOH, LA, Ac, Y, LAB, AAB, Oxygen, T = states

    # FIXED: Better initial guess for fsolve based on starting pH
    initial_guess = 10**(-params['pH_initial'])

    H = fsolve(solve_H, np.array([initial_guess]), args=(params['K_w'],
                                                         Cit, params['M_Cit'], params['K_a1_Cit'], params['K_a2_Cit'], params['K_a3_Cit'],
                                                         Ac, params['M_Ac'], params['K_a_Ac'],
                                                         LA, params['M_LA'], params['K_a_LA'],
                                                         params['Cat']))[0]

    # Protect against math domain error if solver drops H below 0
    H = max(H, 1e-14)
    pH = -np.log10(H)

    Y_pop = Y
    LAB_pop = LAB
    AAB_pop = AAB

    # --- Temperature Adjustments Grouped by Microbe ---
    # Yeast (Y)
    T_min_Y, T_opt_Y, T_max_Y = params['T_min_Y'], params['T_opt_Y'], params['T_max_Y']
    pH_min_Y, pH_opt_Y, pH_max_Y = params['pH_min_Y'], params['pH_opt_Y'], params['pH_max_Y']
    mu_pH_Y = calculate_mu_pH(pH, pH_min_Y, pH_opt_Y, pH_max_Y)
    mu_Y_Glc = calculate_mu_T(T, params['mu_max_Y_Glc'], T_min_Y, T_opt_Y, T_max_Y) * mu_pH_Y
    mu_Y_Fru = calculate_mu_T(T, params['mu_max_Y_Fru'], T_min_Y, T_opt_Y, T_max_Y) * mu_pH_Y
    mu_Y_LA  = calculate_mu_T(T, params['mu_max_Y_LA'], T_min_Y, T_opt_Y, T_max_Y) * mu_pH_Y

    # Lactic Acid Bacteria (LAB)
    T_min_LAB, T_opt_LAB, T_max_LAB = params['T_min_LAB'], params['T_opt_LAB'], params['T_max_LAB']
    pH_min_LAB, pH_opt_LAB, pH_max_LAB = params['pH_min_LAB'], params['pH_opt_LAB'], params['pH_max_LAB']
    mu_pH_LAB = calculate_mu_pH(pH, pH_min_LAB, pH_opt_LAB, pH_max_LAB)
    mu_LAB_Glc = calculate_mu_T(T, params['mu_max_LAB_Glc'], T_min_LAB, T_opt_LAB, T_max_LAB) * mu_pH_LAB
    mu_LAB_Fru = calculate_mu_T(T, params['mu_max_LAB_Fru'], T_min_LAB, T_opt_LAB, T_max_LAB) * mu_pH_LAB
    mu_LAB_Cit = calculate_mu_T(T, params['mu_max_LAB_Cit'], T_min_LAB, T_opt_LAB, T_max_LAB) * mu_pH_LAB

    # Acetic Acid Bacteria (AAB)
    T_min_AAB, T_opt_AAB, T_max_AAB = params['T_min_AAB'], params['T_opt_AAB'], params['T_max_AAB']
    pH_min_AAB, pH_opt_AAB, pH_max_AAB = params['pH_min_AAB'], params['pH_opt_AAB'], params['pH_max_AAB']
    mu_pH_AAB = calculate_mu_pH(pH, pH_min_AAB, pH_opt_AAB, pH_max_AAB)
    mu_AAB_EtOH = calculate_mu_T(T, params['mu_max_AAB_EtOH'], T_min_AAB, T_opt_AAB, T_max_AAB) * mu_pH_AAB
    mu_AAB_LA   = calculate_mu_T(T, params['mu_max_AAB_LA'], T_min_AAB, T_opt_AAB, T_max_AAB) * mu_pH_AAB
    mu_AAB_Ac   = calculate_mu_T(T, params['mu_max_AAB_Ac'], T_min_AAB, T_opt_AAB, T_max_AAB) * mu_pH_AAB

    # 1. Calculate Growth Rates
    v1 = (mu_Y_Glc * Glc / (Glc + params['K_Glc_Y'])) * Y_pop
    v2 = (mu_Y_Fru * Fru / (Fru + params['K_Fru_Y'])) * Y_pop
    v10 = (mu_Y_LA * LA / (LA + params['K_LA_Y'])) * Y_pop

    v3 = (mu_LAB_Glc * Glc / (Glc + params['K_Glc_LAB'])) * LAB_pop
    v9 = (mu_LAB_Fru * Fru / (Fru + params['K_Fru_LAB'])) * LAB_pop

    # NEW PATHWAY: Citric acid fermented by LAB and converted into acetic acid and lactic acid
    # This leads to increase in pH, thereby activating AAB and temperature increase
    v12 = (mu_LAB_Cit * Cit / (Cit + params['K_Cit_LAB'])) * LAB_pop

    v4 = (mu_AAB_EtOH * EtOH / (EtOH + params['K_EtOH_AAB'])) * (Oxygen / (Oxygen + params['K_Oxygen_EtOH'])) * AAB_pop

    v5_denom = LA + params['K_LA_AAB'] * AAB_pop
    v5 = (mu_AAB_LA * LA / v5_denom) * (Oxygen / (Oxygen + params['K_Oxygen_LA'])) * AAB_pop if v5_denom > 0 else 0

    v11 = (mu_AAB_Ac * Ac / (Ac + params['K_Ac_AAB'])) * (Oxygen / (Oxygen + params['K_Oxygen_Ac'])) * AAB_pop

    # 2. Calculate Mortality & Decay Rates
    mu_Y_EtOH = calculate_mu_T(T, params['k_Y'], T_min_Y, T_opt_Y, T_max_Y) * mu_pH_Y
    mu_LAB_LA = calculate_mu_T(T, params['k_LAB'], T_min_LAB, T_opt_LAB, T_max_LAB) * mu_pH_LAB
    mu_AAB_Ac = calculate_mu_T(T, params['k_AAB'], T_min_AAB, T_opt_AAB, T_max_AAB) * mu_pH_AAB

    v6 = Y_pop * EtOH * mu_Y_EtOH
    v7 = LAB_pop * LA * mu_LAB_LA
    v8 = AAB_pop * (Ac**2) * mu_AAB_Ac

    T_Kelvin = T + 273.15

    rate_d1 = params['b_E0'] + params['b_E1'] * np.exp(-params['Delta_H_EtOH'] / (params['R'] * T_Kelvin))
    d1 = rate_d1 * EtOH

    d2 = params['b_LA'] * LA

    rate_d3 = params['b_AC0'] + params['b_AC1'] * np.exp(-params['Delta_H_Ac'] / (params['R'] * T_Kelvin))
    d3 = rate_d3 * Ac

    # 3. Calculate Differentials (ODEs)
    dGlc = - params['Y_Glc_Y'] * v1 - params['Y_Glc_LAB'] * v3
    dFru = - params['Y_Fru_Y'] * v2 - params['Y_Fru_LAB'] * v9
    dCit = - params['Y_Cit_LAB'] * v12
    dEtOH = (params['Y_EtOH_Y_Glc'] * v1 + params['Y_EtOH_Y_Fru'] * v2 +
             params['Y_EtOH_LAB_Glc'] * v3 + params['Y_EtOH_LAB_Fru'] * v9 +
             params['Y_EtOH_Y_LA'] * v10 - params['Y_EtOH_AAB'] * v4 - d1)
    dLA = (params['Y_LA_LAB_Glc'] * v3 + params['Y_LA_LAB_Fru'] * v9 +
           params['Y_LA_LAB_Cit'] * v12 - params['Y_LA_AAB'] * v5 - params['Y_LA_Y'] * v10 - d2)
    dAc = (params['Y_Ac_LAB_Glc'] * v3 + params['Y_Ac_LAB_Fru'] * v9 +
           params['Y_Ac_AAB_EtOH'] * v4 + params['Y_Ac_AAB_LA'] * v5 +
           params['Y_Ac_Y_Glc'] * v1 + params['Y_Ac_Y_Fru'] * v2 +
           params['Y_Ac_LAB_Cit'] * v12 - params['Y_Ac_AAB'] * v11 - d3)
    dY = v1 + v2 + v10 - v6
    dLAB = v3 + v9 + v12 - v7
    dAAB = v4 + v5 + v11 - v8

    dOxygen = (params['A_max'] / (1 + np.exp(-(t - params['t_aer'])))) * (params['C_air'] - Oxygen) - v4 - v5 - v11

    T_e_range = params['T_e_max'] - params['T_e_min']
    T_e = T_e_range / 2 * np.cos(np.pi * t / 12) + (params['T_e_max'] - T_e_range / 2)

    dTemperature = (
            params['Y_Q_Glc'] * (params['Y_Glc_Y'] * v1 + params['Y_Glc_LAB'] * v3) +
            params['Y_Q_Fru'] * (params['Y_Fru_Y'] * v2) +
            params['Y_Q_EtOH'] * (params['Y_EtOH_AAB'] * v4) +
            params['Y_Q_LA'] * (params['Y_LA_AAB'] * v5) -
            params['Q_L'] * (T - T_e)
    )

    return [dGlc, dFru, dCit, dEtOH, dLA, dAc, dY, dLAB, dAAB, dOxygen, dTemperature]

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

baseline_params = {

    # Max growth rates
    'mu_max_Y_Glc': 0.406, 'mu_max_Y_Fru': 0.233, 'mu_max_Y_LA': 0.5,
    'mu_max_LAB_Glc': 0.137, 'mu_max_LAB_Fru': 0.193,
    'mu_max_AAB_EtOH': 0.466, 'mu_max_AAB_LA': 0.016, 'mu_max_AAB_Ac': 0.5,

    # Saturation constants
    'K_Glc_Y': 38.569, 'K_Fru_Y': 39.902, 'K_Glc_LAB': 34.266,
    'K_EtOH_AAB': 15.995, 'K_LA_AAB': 2568.5,
    'K_Fru_LAB': 38.586, 'K_LA_Y': 14.74, 'K_Ac_AAB': 3.84,

    # Mortality base constants
    'k_Y': 0.0455, 'k_LAB': 0.007, 'k_AAB': 0.0096,

    # Yield coefficients
    'Y_Glc_Y': 33.305, 'Y_Glc_LAB': 46.411, 'Y_Fru_Y': 39.129,
    'Y_EtOH_Y_Glc': 4.236, 'Y_EtOH_Y_Fru': 6.12, 'Y_EtOH_AAB': 1769.652,
    'Y_LA_LAB_Glc': 9.021, 'Y_LA_AAB': 2012.646,
    'Y_Ac_LAB_Glc': 3.277, 'Y_Ac_AAB_EtOH': 68.368, 'Y_Ac_AAB_LA': 1170.869,
    'Y_Fru_LAB': 51.41, 'Y_EtOH_LAB_Glc': 14.762, 'Y_EtOH_LAB_Fru': 9.598,
    'Y_EtOH_Y_LA': 34.08, 'Y_LA_LAB_Fru': 9.673, 'Y_LA_Y': 27.78,
    'Y_Ac_LAB_Fru': 1.809, 'Y_Ac_Y_Glc': 1.038, 'Y_Ac_Y_Fru': 1.635, 'Y_Ac_AAB': 793.44
}

T_O2_pH_params = {
    # Temperature Parameters grouped by Microbe for the cardinal model
    'T_min_Y': 2.8,   'T_opt_Y': 32.3,   'T_max_Y': 45.4,
    'T_min_LAB': 12, 'T_opt_LAB': 37.1, 'T_max_LAB': 52,
    'T_min_AAB': 5, 'T_opt_AAB': 27.5, 'T_max_AAB': 42,

    # Environmental temperature parameters. Q_L determines how much the internal temperature
    # is affected by the ambient temperature
    'T_e_min': 19.0,
    'T_e_max': 36.0,
    'Q_L': 0.005,

    # Heat Yield Coefficients
    'Y_Q_Glc': 0.1, 'Y_Q_Fru': 0.1, 'Y_Q_EtOH': 1, 'Y_Q_LA': 0.1,

    # Oxygen Parameters (t_aer is mid-transition phase from anaerobic to aerobic)
    'K_Oxygen_EtOH': 5e-3, 'K_Oxygen_LA': 5e-3, 'K_Oxygen_Ac': 5e-3,
    'A_max': 1.0, 't_aer': 60.0, 'C_air': 0.00826,

    # Decay parameters
    'b_LA': 0.01,
    'b_E0': 0.001,
    'b_E1': 10,
    'b_AC0': 0.005,
    'b_AC1': 10,

    # Enthalpies of evaporation and gas constant
    'Delta_H_EtOH': 38975.22,
    'Delta_H_Ac': 24140.1,
    'R': 8.314,

    # Molar masses
    'M_EtOH': 46.07, 'M_LA': 90.08, 'M_Ac': 60.05, 'M_Cit': 192.12,

    # Dissociation constants and pH
    'pH_initial': 3.5,
    'K_w': 1e-14,
    'K_a1_Cit': 7.4e-4, 'K_a2_Cit': 1.7e-5, 'K_a3_Cit': 4.0e-7,
    'K_a_Ac': 1.75e-5, 'K_a_LA': 1.38e-4,

    # pH for cardinal model
    'pH_min_Y': 2.5,   'pH_opt_Y': 4,   'pH_max_Y': 8.5,
    'pH_min_LAB': 3.2, 'pH_opt_LAB': 5.85, 'pH_max_LAB': 9.2,
    'pH_min_AAB': 3.2, 'pH_opt_AAB': 5.9, 'pH_max_AAB': 7.0,
}

Cit_params = {

    # Max growth rate
    'mu_max_LAB_Cit': 2,

    # Saturation constant
    'K_Cit_LAB': 4.5,

    # Yield coefficients
    'Y_Cit_LAB': 1.0, 'Y_LA_LAB_Cit': 1.0, 'Y_Ac_LAB_Cit': 1.0
}

params = {**baseline_params, **T_O2_pH_params, **Cit_params}

# Initial conditions [Glc, Fru, Cit, EtOH, LA, Ac, Y, LAB, AAB, Oxygen, T]
Cit_0 = 9.0
initial_conditions = [52.0, 58.0, Cit_0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.001, 0.005, 27]

# Cation concentration in pulp (not including H+). We treat this as a constant.
Cat_0 = calculate_Cat(params['pH_initial'], params['K_w'], Cit_0, params['M_Cit'],
                      params['K_a1_Cit'], params['K_a2_Cit'], params['K_a3_Cit'])
# Cat_0 = 0.05
params['Cat'] = Cat_0

t_span = (0, 168)
t_eval = np.linspace(t_span[0], t_span[1], 1000)


solution = solve_ivp(
    fun=full_fermentation_derivatives,
    t_span=t_span,
    y0=initial_conditions,
    t_eval=t_eval,
    args=(params,),
    method='RK45'
)

pH_list = []
LA = solution.y[4]
Ac = solution.y[5]
Cit = solution.y[2]

# FIXED: Improved guess for plotting loop
initial_guess = 10**(-params['pH_initial'])

# Finding the pH
for t in range(len(LA)):
    H = fsolve(solve_H, np.array([initial_guess]), args=(params['K_w'],
                                                         Cit[t], params['M_Cit'], params['K_a1_Cit'], params['K_a2_Cit'], params['K_a3_Cit'],
                                                         Ac[t], params['M_Ac'], params['K_a_Ac'],
                                                         LA[t], params['M_LA'], params['K_a_LA'],
                                                         params['Cat']))[0]
    H = max(H, 1e-14)
    pH = -np.log10(H)
    pH_list.append(pH)

# Calculating ambient temperature
T_e_range = params['T_e_max'] - params['T_e_min']
T_e = T_e_range / 2 * np.cos(np.pi * solution.t / 12) + (params['T_e_max'] - T_e_range / 2)

# Plotting the Results
nrows, ncols = 4, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(10, 12), sharex=True)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# fig.suptitle('Cocoa bean fermentation')

labels = ['Glucose', 'Fructose', 'Citric Acid', 'Ethanol', 'Lactic Acid', 'Acetic Acid',
          'Yeast', 'LAB', 'AAB', 'Oxygen', 'Temperature', 'pH']
colors = ['blue', 'orange', 'darkgoldenrod', 'green', 'red', 'purple',
          'brown', 'pink', 'gray', 'cyan', 'black', 'darkviolet']

for i in range(11):
    ax = axs[i//ncols, i%ncols]
    ax.set_title(labels[i])
    ax.set_xlabel('Time [h]')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 5))

    if labels[i] == 'Temperature':
        ax.plot(solution.t, solution.y[i], color=colors[i], label='Pulp')
        ax.set_ylabel('°C')
        ax.plot(solution.t, T_e, color=colors[i], label='Ambient', linestyle='dotted', lw=0.5)
        ax.legend()
    else:
        ax.plot(solution.t, solution.y[i], color=colors[i])
        ax.set_ylabel('mg g(pulp)\\textsuperscript{-1}')

ax_pH = axs[nrows-1, ncols-1]
ax_pH.plot(solution.t, np.array(pH_list), color=colors[-1])
ax_pH.set_title(labels[-1])
ax_pH.set_xlabel('Time [h]')

fig.savefig('resources/initial/pH_T_O2_citric.png', bbox_inches='tight', dpi=400)
