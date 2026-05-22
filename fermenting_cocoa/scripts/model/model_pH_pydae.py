import json
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from pydae.core import Builder, Model

import time
plt.rcParams['text.usetex'] = True

trial = "initial"


def calculate_mu_T(T, mu_opt, T_min, T_opt, T_max):
    """
    Calculates the temperature-adjusted specific growth rate using the
    cardinal model.
    """

    numerator = (T - T_max) * ((T - T_min) ** 2)

    term1 = (T_opt - T_min) * (T - T_opt)
    term2 = (T_opt - T_max) * (T_opt + T_min - 2 * T)
    denominator = (T_opt - T_min) * (term1 - term2)

    mu_T = sym.Piecewise(
        (0.0, T < T_min),
        (0.0, T > T_max),
        (mu_opt * numerator / denominator, True)
    )

    return mu_T


def calculate_mu_pH(pH, pH_min, pH_opt, pH_max):
    """
    Calculates the effect pH has on the reaction rate using the cardinal
    model.
    """
    numerator = (pH - pH_min) * (pH - pH_max)
    term1 = (pH - pH_min) * (pH - pH_max)
    term2 = (pH - pH_opt) ** 2
    denominator = term1 - term2

    mu_pH = sym.Piecewise(
        (0.0, pH < pH_min),
        (0.0, pH > pH_max),
        (numerator / denominator, True)
    )

    return mu_pH


def H_RHS(_H, _Ac, _LA, params):

    K_w, _Cit = params["K_w"], params["Cit_0"]
    M_Cit, K_a1_cit, K_a2_cit, K_a3_cit = params["M_Cit"], params["K_a1_Cit"], params["K_a2_Cit"], params["K_a3_Cit"]
    M_Ac, K_a_Ac = params["M_Ac"], params["K_a_Ac"]
    M_LA, K_a_LA = params["M_LA"], params["K_a_LA"]

    term1 = K_w / _H - _H
    term2 = (_Cit / M_Cit * (K_a1_cit * _H ** 2 + 2 * K_a1_cit * K_a2_cit * _H + 3 * K_a1_cit * K_a2_cit * K_a3_cit) /
             (_H ** 3 + K_a1_cit * _H ** 2 + K_a1_cit * K_a2_cit * _H + K_a1_cit * K_a2_cit * K_a3_cit))

    term3 = (_Ac / M_Ac) * K_a_Ac / (_H + K_a_Ac)
    term4 = (_LA / M_LA) * K_a_LA / (_H + K_a_LA)

    eq = term1 + term2 + term3 + term4
    return eq


def setup_system(params):
    """
    Computes the derivatives for the FULL cocoa bean fermentation model,
    including M1-M5, O2 dynamics and Temperature dynamics.
    """

    # States (note we include a dummy time variable tau)
    tau, Glc, Fru, EtOH, LA, Ac, Y, LAB, AAB, O2, T = sym.symbols("tau,Glc,Fru,EtOH,LA,Ac,Y,LAB,AAB,O2,T", real=True)

    # Algebraic unknowns
    H, pH = sym.symbols("H,pH", real=True)

    # Algebraic equations
    g_1 = H_RHS(H, Ac, LA, params)
    g_2 = H - pow(10, -pH)

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

    v4 = (mu_AAB_EtOH * EtOH / (EtOH + params['K_EtOH_AAB'])) * (O2 / (O2 + params['K_O2_EtOH'])) * AAB

    v5_denom = LA + params['K_LA_AAB'] * AAB
    v5 = (mu_AAB_LA * LA / v5_denom) * (O2 / (O2 + params['K_O2_LA'])) * AAB

    v11 = (mu_AAB_Ac * Ac / (Ac + params['K_Ac_AAB'])) * (O2 / (O2 + params['K_O2_Ac'])) * AAB

    # 2. Calculate Mortality & Decay Rates
    mu_Y_EtOH = calculate_mu_T(T, params['k_Y'], T_min_Y, T_opt_Y, T_max_Y) * mu_pH_Y
    mu_LAB_LA = calculate_mu_T(T, params['k_LAB'], T_min_LAB, T_opt_LAB, T_max_LAB) * mu_pH_LAB
    mu_AAB_Ac = calculate_mu_T(T, params['k_AAB'], T_min_AAB, T_opt_AAB, T_max_AAB) * mu_pH_AAB

    v6 = Y * EtOH * mu_Y_EtOH
    v7 = LAB * LA * mu_LAB_LA
    v8 = AAB * (Ac**2) * mu_AAB_Ac

    T_Kelvin = T + 273.15

    rate_d1 = params['b_E0'] + params['b_E1'] * sym.exp(-params['Delta_H_EtOH'] / (params['R'] * T_Kelvin))
    d1 = rate_d1 * EtOH

    d2 = params['b_LA'] * LA

    rate_d3 = params['b_AC0'] + params['b_AC1'] * sym.exp(-params['Delta_H_Ac'] / (params['R'] * T_Kelvin))
    d3 = rate_d3 * Ac

    # ODEs
    dtau = 1
    dGlc = -params['Y_Glc_Y'] * v1 - params['Y_Glc_LAB'] * v3
    dFru = -params['Y_Fru_Y'] * v2 - params['Y_Fru_LAB'] * v9
    dEtOH = (params['Y_EtOH_Y_Glc'] * v1 + params['Y_EtOH_Y_Fru'] * v2 +
             params['Y_EtOH_LAB_Glc'] * v3 + params['Y_EtOH_LAB_Fru'] * v9 +
             params['Y_EtOH_Y_LA'] * v10 - params['Y_EtOH_AAB'] * v4 - d1)
    dLA = (params['Y_LA_LAB_Glc'] * v3 + params['Y_LA_LAB_Fru'] * v9 -
           params['Y_LA_AAB'] * v5 - params['Y_LA_Y'] * v10 - d2)
    dAc = (params['Y_Ac_LAB_Glc'] * v3 + params['Y_Ac_LAB_Fru'] * v9 +
           params['Y_Ac_AAB_EtOH'] * v4 + params['Y_Ac_AAB_LA'] * v5 +
           params['Y_Ac_Y_Glc'] * v1 + params['Y_Ac_Y_Fru'] * v2 -
           params['Y_Ac_AAB'] * v11 - d3)
    dY = v1 + v2 + v10 - v6
    dLAB = v3 + v9 - v7
    dAAB = v4 + v5 + v11 - v8

    dO2 = (params['A_max'] / (1 + sym.exp(-(tau - params['t_aer'])))) * (params['C_air'] - O2) - v4 - v5 - v11

    T_e_range = params['T_e_max'] - params['T_e_min']
    T_e = T_e_range / 2 * sym.cos(sym.pi * tau / 12) + (params['T_e_max'] - T_e_range / 2)

    dT = (
            params['Y_Q_Glc'] * (params['Y_Glc_Y'] * v1 + params['Y_Glc_LAB'] * v3) +
            params['Y_Q_Fru'] * (params['Y_Fru_Y'] * v2) +
            params['Y_Q_EtOH'] * (params['Y_EtOH_AAB'] * v4) +
            params['Y_Q_LA'] * (params['Y_LA_AAB'] * v5) -
            params['Q_L'] * (T - T_e)
    )
    _x_list = [tau, Glc, Fru, EtOH, LA, Ac, Y, LAB, AAB, O2, T]
    _f_list = [dtau, dGlc, dFru, dEtOH, dLA, dAc, dY, dLAB, dAAB, dO2, dT]
    _g_list = [g_1, g_2]
    _y_ini_list = [H, pH]
    _y_run_list = [H, pH]

    return _x_list, _f_list, _g_list, _y_ini_list, _y_run_list


def run_model_pH(params, initial_conditions, t_end):
    x_list, f_list, g_list, y_ini_list, y_run_list = setup_system(params)

    u_ini_dict = {}
    u_run_dict, h_dict = {}, {}

    # pydae solver
    sys_dict = {"name": "fermenting_pulp", "params_dict": {}, "f_list": f_list, "g_list": g_list, "x_list": x_list,
                "y_ini_list": y_ini_list, "y_run_list": y_run_list, "u_ini_dict": u_ini_dict,
                "u_run_dict": u_run_dict, "h_dict": h_dict}
    Builder(sys_dict, target="ctypes", sparse=False).build()

    # Initialise model
    model = Model("fermenting_pulp")
    model.decimation = 10
    model.ini(u_ini_dict, xy_0=initial_conditions)

    # Run model
    model.run(t_end, {})
    model.post()
    return model
