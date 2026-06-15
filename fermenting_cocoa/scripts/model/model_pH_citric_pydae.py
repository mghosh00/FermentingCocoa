import json
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from pydae.core import Builder, Model
import io
import contextlib
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


def setup_system(params):
    """
    Computes the derivatives for the FULL cocoa bean fermentation model,
    including M1-M5, O2 dynamics and Temperature dynamics.
    """

    # Nondimensional states (note we include a dummy time variable tau)
    tau_nd, Glc_nd, Fru_nd, Cit_nd, EtOH_nd, LA_nd = sym.symbols("tau,Glc,Fru,Cit,EtOH,LA", real=True)
    Ac_nd, Y_nd, LAB_nd, AAB_nd, O2_nd, T_nd = sym.symbols("Ac,Y,LAB,AAB,O2,T", real=True)

    # Dimensional states (to be inserted into equations)
    tau, Glc, Fru = tau_nd * params["tau_sc"], Glc_nd * params["Glc_sc"], Fru_nd * params["Fru_sc"]
    Cit, EtOH, LA = Cit_nd * params["Cit_sc"], EtOH_nd * params["EtOH_sc"], LA_nd * params["LA_sc"]
    Ac, Y, LAB = Ac_nd * params["Ac_sc"], Y_nd * params["Y_sc"], LAB_nd * params["LAB_sc"]
    AAB, O2, T = AAB_nd * params["AAB_sc"], O2_nd * params["O2_sc"], T_nd * params["T_sc"]

    # Algebraic unknowns
    pH_nd = sym.symbols("pH", real=True)
    pH = pH_nd * params["pH_sc"]

    # Algebraic equations
    g_1 = pH_RHS(pH, Cit, Ac, LA, params)

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
    dtau = 1 / params['tau_sc']
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

    dO2 = ((params['A_max'] / (1 + sym.exp(-(tau - params['t_aer'])))) * (params['C_air'] - O2)
           - v4 - v5 - v11) / params['O2_sc']

    T_e_range = params['T_e_max'] - params['T_e_min']
    T_e = T_e_range / 2 * sym.cos(sym.pi * tau / 12) + (params['T_e_max'] - T_e_range / 2)

    dT = (
            params['Y_Q_Glc'] * (params['Y_Glc_Y'] * v1 + params['Y_Glc_LAB'] * v3) +
            params['Y_Q_Fru'] * (params['Y_Fru_Y'] * v2) +
            params['Y_Q_EtOH'] * (params['Y_EtOH_AAB'] * v4) +
            params['Y_Q_LA'] * (params['Y_LA_AAB'] * v5) -
            params['Q_L'] * (T - T_e)
    ) / params['T_sc']
    # Solve for the nondimensional states
    _x_list = [tau_nd, Glc_nd, Fru_nd, Cit_nd, EtOH_nd, LA_nd, Ac_nd, Y_nd, LAB_nd, AAB_nd, O2_nd, T_nd]
    _f_list = [dtau, dGlc, dFru, dCit, dEtOH, dLA, dAc, dY, dLAB, dAAB, dO2, dT]
    _g_list = [g_1]
    _y_ini_list = [pH_nd]
    _y_run_list = [pH_nd]

    return _x_list, _f_list, _g_list, _y_ini_list, _y_run_list


def build_model_pH_citric(params):
    # Cation concentration (can be found from initial conditions on pH and Cit)
    params["Cat"] = calculate_Cat(params['pH_initial'], params['K_w'],
                                  params['Cit'], params['M_Cit'], params['K_a1_Cit'],
                                  params['K_a2_Cit'], params['K_a3_Cit'])
    sym_params = {k: sym.Symbol(k, real=True) for k in params.keys()}
    x_list, f_list, g_list, y_ini_list, y_run_list = setup_system(sym_params)

    u_ini_dict = {}
    u_run_dict, h_dict = {}, {}

    # pydae solver
    sys_dict = {"name": "fermenting_pulp", "params_dict": params, "f_list": f_list, "g_list": g_list, "x_list": x_list,
                "y_ini_list": y_ini_list, "y_run_list": y_run_list, "u_ini_dict": u_ini_dict,
                "u_run_dict": u_run_dict, "h_dict": h_dict}
    Builder(sys_dict, target="ctypes", sparse=False).build()

    # Initialise model
    return Model("fermenting_pulp")


def run_model_pH_citric(model, params, initial_conditions, t_end, verbose):
    model.decimation = 10
    with contextlib.redirect_stdout(io.StringIO()):
        model.ini(params, xy_0=initial_conditions)
    # model.ini(params, xy_0=initial_conditions)

    # Run model
    model.run(t_end, {})
    model.post()
    return model
