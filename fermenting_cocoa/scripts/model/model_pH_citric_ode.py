import json
import numpy as np
import scipy.integrate as si
from numba import njit, types
from numba.typed import Dict
import matplotlib.pyplot as plt
import io
import contextlib
plt.rcParams['text.usetex'] = True

trial = "initial"


@njit(nopython=True)
def calculate_mu_T(T, mu_opt, T_min, T_opt, T_max):
    """
    Calculates the temperature-adjusted specific growth rate using the
    cardinal model.
    """

    numerator = (T - T_max) * ((T - T_min) ** 2)

    term1 = (T_opt - T_min) * (T - T_opt)
    term2 = (T_opt - T_max) * (T_opt + T_min - 2 * T)
    denominator = (T_opt - T_min) * (term1 - term2)

    mu_T = 0.0 if T < T_min else 0.0 if T > T_max else mu_opt * numerator / denominator

    return mu_T


@njit(nopython=True)
def calculate_mu_pH(pH, pH_min, pH_opt, pH_max):
    """
    Calculates the effect pH has on the reaction rate using the cardinal
    model.
    """
    numerator = (pH - pH_min) * (pH - pH_max)
    term1 = (pH - pH_min) * (pH - pH_max)
    term2 = (pH - pH_opt) ** 2
    denominator = term1 - term2

    mu_pH = 0.0 if pH < pH_min else 0.0 if pH > pH_max else numerator / denominator
    return mu_pH


@njit(nopython=True)
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


@njit(nopython=True)
def pH_RHS(_pH, _Cit, _Ac, _LA, params):

    _H = pow(10, -_pH)

    K_w, _Cat = params["K_w"], params["Cat"]
    M_Cit, K_a1_cit, K_a2_cit, K_a3_cit = params["M_Cit"], params["K_a1_Cit"], params["K_a2_Cit"], params["K_a3_Cit"]
    M_Ac, K_a_Ac = params["M_Ac"], params["K_a_Ac"]
    M_LA, K_a_LA = params["M_LA"], params["K_a_LA"]

    term1 = K_w / _H - _H - _Cat
    term2 = (_Cit / M_Cit * (K_a1_cit * _H ** 2 + 2 * K_a1_cit * K_a2_cit * _H + 3 * K_a1_cit * K_a2_cit * K_a3_cit) /
             (_H ** 3 + K_a1_cit * _H ** 2 + K_a1_cit * K_a2_cit * _H + K_a1_cit * K_a2_cit * K_a3_cit))

    term3 = (_Ac / M_Ac) * K_a_Ac / (_H + K_a_Ac)
    term4 = (_LA / M_LA) * K_a_LA / (_H + K_a_LA)

    eq = term1 + term2 + term3 + term4
    return eq


@njit(nopython=True)
def dCat_dt(params):
    """The derivative of the concentration of cations as a function of time.
    """
    return 0.0


@njit(nopython=True)
def H_numerator(_H, _dCit, _dAc, _dLA, _params):
    Cit_sc, Ac_sc, LA_sc = _params["Cit_sc"], _params["Ac_sc"], _params["LA_sc"]
    M_Cit, K_a1_cit, K_a2_cit, K_a3_cit = _params["M_Cit"], _params["K_a1_Cit"], _params["K_a2_Cit"], _params["K_a3_Cit"]
    M_Ac, K_a_Ac = _params["M_Ac"], _params["K_a_Ac"]
    M_LA, K_a_LA = _params["M_LA"], _params["K_a_LA"]
    term1 = -dCat_dt(_params)
    term2 = (_dCit * Cit_sc / M_Cit * (K_a1_cit * _H ** 2 + 2 * K_a1_cit * K_a2_cit * _H + 3 * K_a1_cit * K_a2_cit * K_a3_cit) /
             (_H ** 3 + K_a1_cit * _H ** 2 + K_a1_cit * K_a2_cit * _H + K_a1_cit * K_a2_cit * K_a3_cit))
    term3 = _dAc * Ac_sc / M_Ac * K_a_Ac / (_H + K_a_Ac)
    term4 = _dLA * LA_sc / M_LA * K_a_LA / (_H + K_a_LA)
    return term1 + term2 + term3 + term4


@njit(nopython=True)
def H_denominator(_H, _Cit, _Ac, _LA, _params):
    K_w = _params["K_w"]
    M_Cit, K_a1_cit, K_a2_cit, K_a3_cit = _params["M_Cit"], _params["K_a1_Cit"], _params["K_a2_Cit"], _params["K_a3_Cit"]
    M_Ac, K_a_Ac = _params["M_Ac"], _params["K_a_Ac"]
    M_LA, K_a_LA = _params["M_LA"], _params["K_a_LA"]
    term1 = 1 + K_w / _H ** 2
    term2 = (_Cit / M_Cit * K_a1_cit *
             (_H ** 4 + 4 * K_a2_cit * _H ** 3 + (K_a1_cit + 9 * K_a3_cit) * K_a2_cit * _H ** 2 +
              4 * K_a1_cit * K_a2_cit * K_a3_cit * _H + K_a1_cit * K_a2_cit ** 2 * K_a3_cit) /
             (_H ** 3 + K_a1_cit * _H ** 2 + K_a1_cit * K_a2_cit * _H + K_a1_cit * K_a2_cit * K_a3_cit) ** 2)
    term3 = _Ac * K_a_Ac / M_Ac / (_H + K_a_Ac) ** 2
    term4 = _LA * K_a_LA / M_LA / (_H + K_a_LA) ** 2
    return term1 + term2 + term3 + term4


@njit(nopython=True)
def system(t, states, params):
    """
    Computes the derivatives for the FULL cocoa bean fermentation model,
    including M1-M5, O2 dynamics and Temperature dynamics.
    """

    # Nondimensional states
    Glc_nd, Fru_nd, Cit_nd, EtOH_nd, LA_nd, Ac_nd, Y_nd, LAB_nd, AAB_nd, O2_nd, T_nd, pH_nd = states

    # Dimensional states (to be inserted into equations)
    Glc, Fru = Glc_nd * params["Glc_sc"], Fru_nd * params["Fru_sc"]
    Cit, EtOH, LA = Cit_nd * params["Cit_sc"], EtOH_nd * params["EtOH_sc"], LA_nd * params["LA_sc"]
    Ac, Y, LAB = Ac_nd * params["Ac_sc"], Y_nd * params["Y_sc"], LAB_nd * params["LAB_sc"]
    AAB, O2, T, pH = AAB_nd * params["AAB_sc"], O2_nd * params["O2_sc"], T_nd * params["T_sc"], pH_nd * params["pH_sc"]
    H = pow(10, -pH)

    # To prevent concentrations going below zero
    Glc = max(Glc, 0.0)
    O2 = max(O2, 0.0)

    # Setting up differential equations

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
    v1 = (mu_Y_Glc * Glc / (Glc + params['K_Glc_Y'])) * Y
    v2 = (mu_Y_Fru * Fru / (Fru + params['K_Fru_Y'])) * Y
    v10 = (mu_Y_LA * LA / (LA + params['K_LA_Y'])) * Y

    v3 = (mu_LAB_Glc * Glc / (Glc + params['K_Glc_LAB'])) * LAB
    v9 = (mu_LAB_Fru * Fru / (Fru + params['K_Fru_LAB'])) * LAB

    # NEW PATHWAY: Citric acid fermented by LAB and converted into acetic acid and lactic acid
    # This leads to increase in pH, thereby activating AAB and temperature increase
    v12 = (mu_LAB_Cit * Cit / (Cit + params['K_Cit_LAB'])) * LAB

    v4 = (mu_AAB_EtOH * EtOH / (EtOH + params['K_EtOH_AAB'])) * (O2 / (O2 + params['K_O2_EtOH'])) * AAB

    v5_denom = LA + params['K_LA_AAB'] * AAB
    v5 = (mu_AAB_LA * LA / v5_denom) * (O2 / (O2 + params['K_O2_LA'])) * AAB

    v11 = (mu_AAB_Ac * Ac / (Ac + params['K_Ac_AAB'])) * (O2 / (O2 + params['K_O2_Ac'])) * AAB

    # 2. Calculate Mortality & Decay Rates
    T_Kelvin = T + 273.15

    k_Y = params['k_Y'] * np.exp(-params['E_a_k_Y'] / (params['R'] * T_Kelvin))
    k_LAB = params['k_LAB'] * np.exp(-params['E_a_k_LAB'] / (params['R'] * T_Kelvin))
    k_AAB = params['k_AAB'] * np.exp(-params['E_a_k_AAB'] / (params['R'] * T_Kelvin))

    v6 = Y * EtOH * k_Y
    v7 = LAB * LA * k_LAB
    v8 = AAB * (Ac**2) * k_AAB

    rate_d1 = params['b_E0'] + params['b_E1'] * np.exp(-params['Delta_H_EtOH'] / (params['R'] * T_Kelvin))
    d1 = rate_d1 * EtOH

    d2 = params['b_LA'] * LA

    rate_d3 = params['b_AC0'] + params['b_AC1'] * np.exp(-params['Delta_H_Ac'] / (params['R'] * T_Kelvin))
    d3 = rate_d3 * Ac

    # ODEs
    dGlc = (- params['Y_Glc_Y'] * v1 - params['Y_Glc_LAB'] * v3) / params['Glc_sc']
    dFru = (- params['Y_Fru_Y'] * v2 - params['Y_Fru_LAB'] * v9) / params['Fru_sc']
    dCit = (- params['Y_Cit_LAB'] * v12) / params['Cit_sc']
    dEtOH = (params['Y_EtOH_Y_Glc'] * v1 + params['Y_EtOH_Y_Fru'] * v2 +
             params['Y_EtOH_LAB_Glc'] * v3 + params['Y_EtOH_LAB_Fru'] * v9 +
             params['Y_EtOH_Y_LA'] * v10 - params['Y_EtOH_AAB'] * v4 - d1) / params['EtOH_sc']
    dLA = (params['Y_LA_LAB_Glc'] * v3 + params['Y_LA_LAB_Fru'] * v9 +
           params['Y_LA_LAB_Cit'] * v12 - params['Y_LA_AAB'] * v5 - params['Y_LA_Y'] * v10 - d2) / params['LA_sc']
    dAc = (params['Y_Ac_LAB_Glc'] * v3 + params['Y_Ac_LAB_Fru'] * v9 +
           params['Y_Ac_AAB_EtOH'] * v4 + params['Y_Ac_AAB_LA'] * v5 +
           params['Y_Ac_Y_Glc'] * v1 + params['Y_Ac_Y_Fru'] * v2 +
           params['Y_Ac_LAB_Cit'] * v12 - params['Y_Ac_AAB'] * v11 - d3) / params['Ac_sc']
    dY = (v1 + v2 + v10 - v6) / params['Y_sc']
    dLAB = (v3 + v9 + v12 - v7) / params['LAB_sc']
    dAAB = (v4 + v5 + v11 - v8) / params['AAB_sc']

    dO2 = ((params['A_max'] / (1 + np.exp(-(t - params['t_aer'])))) * (params['C_air'] - O2)
           - v4 - v5 - v11) / params['O2_sc']

    T_e_range = params['T_e_max'] - params['T_e_min']
    T_e = T_e_range / 2 * np.cos(np.pi * t / 12) + (params['T_e_max'] - T_e_range / 2)

    dT = (
                 params['Y_Q_Glc'] * (params['Y_Glc_Y'] * v1 + params['Y_Glc_LAB'] * v3) +
                 params['Y_Q_Fru'] * (params['Y_Fru_Y'] * v2) +
                 params['Y_Q_EtOH'] * (params['Y_EtOH_AAB'] * v4) +
                 params['Y_Q_LA'] * (params['Y_LA_AAB'] * v5) -
                 params['Q_L'] * (T - T_e)
         ) / params['T_sc']

    dH = (H_numerator(H, dCit, dAc, dLA, params) / H_denominator(H, Cit, Ac, LA, params))
    dpH = - dH / (H * np.log(10)) / params['pH_sc']

    return [dGlc, dFru, dCit, dEtOH, dLA, dAc, dY, dLAB, dAAB, dO2, dT, dpH]


def run_model_ODE(params, initial_conditions, t_end, Dt=1e-2):
    """Runs the model.

    :param params: Parameters on a standard scale.
    :param initial_conditions: The initial conditions for the system.
    :param t_end: The final timepoint.
    :param Dt: The timestep the solver uses.
    :return: The underlying model.
    """
    t_span = (0, t_end)
    t_eval = np.arange(t_span[0], t_span[1], Dt)
    params_numba = Dict.empty(key_type=types.string, value_type=types.float64)
    for k, v in params.items():
        params_numba[k] = v
    sol = si.solve_ivp(fun=system,
                       t_span=t_span,
                       t_eval=t_eval,
                       y0=initial_conditions,
                       args=(params_numba,),
                       method='RK45',
                       rtol=1e-4,
                       atol=1e-6)
    return sol
