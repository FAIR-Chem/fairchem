from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def get_she(voltage):
    u_she, u_rhe = fullcell2halfcell(voltage)
    return u_she


def fullcell2halfcell(vcell):
    """
    main function to convert a voltage value from full cell to half cell vs she or rhe
    """
    cali_dict = load_calibration_data_for_voltage_conversion()
    urhe = cell2rhe(
        vcell,
        cali_dict["measurements"]["fullcell pot"],
        cali_dict["extracted params"]["cathode pot corr"],
    )  # RHE
    ushe = rhe2she(
        urhe, cali_dict["conditions"]["cathode pH"], cali_dict["conditions"]["ref pot"]
    )  # SHE
    return ushe, urhe


def load_calibration_data_for_voltage_conversion():
    ## experiment conditions: Nuetral CO2RR in 4cm2 cell, Sputtered Copper Catalyst, 0.1M Bicarbonate - ref electrode (3M kcl) 230mV vs SHE
    ref_pot = 0.23  # V Ag/AgCl electrode
    cathode_pH = 12.5
    anode_pH = 3
    Nern_pH_loss = (cathode_pH - anode_pH) * 0.059
    geo_area = 4  # cm2
    membrane_loss = 0.1  # V
    cathode_thermo = +0.08
    anode_thermo = +1.23
    thermo_pot = anode_thermo - cathode_thermo

    # measurements from Fatimeh's work for calibration
    j = np.array([50, 100, 200])
    cathode_pot = np.array([-1.62, -2.0, -2.3])
    cathode_R = np.array([0.48, 0.34, 0.3])
    anode_pot = np.array([1.3, 1.35, 1.4])
    anode_R = np.array([0, 0, 0])  # almost negligible
    fullcell_pot = np.array([3, 3.4, 3.7])
    fullcell_R = np.array([0.47, 0.35, 0.3])

    n = len(cathode_pot)
    cathode_pot_corr = np.zeros(n)
    anode_pot_corr = np.zeros(n)
    cathode_overpot = np.zeros(n)
    anode_overpot = np.zeros(n)
    fullcell_pot_corr = np.zeros(n)

    for i in range(n):
        cathode_pot_corr[i] = correct_potential(
            cathode_pot[i], cathode_R[i], cathode_pH, j[i], geo_area, ref_pot
        )
        anode_pot_corr[i] = correct_potential(
            anode_pot[i], anode_R[i], anode_pH, j[i], geo_area, ref_pot
        )
        cathode_overpot[i] = get_overpotential(cathode_pot_corr[i], 0.08)
        anode_overpot[i] = get_overpotential(anode_pot_corr[i], 1.23)

        fullcell_pot_corr[i] = fullcell_pot[i] - fullcell_R[i] * j[i] / 1000 * geo_area

    conditions_dict = {
        "ref pot": ref_pot,
        "cathode pH": cathode_pH,
        "anode pH": anode_pH,
        "Nern pH loss": Nern_pH_loss,
        "geo area": geo_area,
        "membrane loss": membrane_loss,
        "cathod thermo": cathode_thermo,
        "anode thermo": anode_thermo,
        "thermo pot": thermo_pot,
    }
    measurements_dict = {
        "j": j,
        "cathode pot": cathode_pot,
        "cathode R": cathode_R,
        "anode pot": anode_pot,
        "anode R": anode_R,
        "fullcell pot": fullcell_pot,
        "fullcell R": fullcell_R,
    }

    data_dict = {
        "cathode pot corr": cathode_pot_corr,
        "anode pot corr": anode_pot_corr,
        "cathode overpot": cathode_overpot,
        "anode overpot": anode_overpot,
        "fullcell pot corr": fullcell_pot_corr,
    }
    return {
        "measurements": measurements_dict,
        "conditions": conditions_dict,
        "extracted params": data_dict,
    }


def she2rhe(ushe, pH, ref_pot):
    ushe = ushe + ref_pot + (0.059 * pH)
    return ushe


def rhe2she(urhe, pH, ref_pot):
    urhe = urhe - ref_pot - (0.059 * pH)
    return urhe


def correct_potential(pot, R, pH, j, area, ref_pot):
    if pot < 0:
        corrected_pot = she2rhe(pot + j / 1000 * area * R, pH, ref_pot)
    else:
        corrected_pot = she2rhe(pot - j / 1000 * area * R, pH, ref_pot)
    return corrected_pot


def lin_fxn(x, a, b):
    return a * x + b


def fit_lin(X, Y):
    params, covariance = curve_fit(lin_fxn, X, Y)
    a_fit, b_fit = params
    return (a_fit, b_fit)


def est_x(x, X, Y):
    fit = fit_lin(X, Y)
    x = fit[0] * x + fit[1]
    return x


def cell2rhe(vcell, X, Y):
    vrhe = est_x(vcell, X, Y)
    return vrhe


def get_overpotential(pot, pot_theory):
    overpot = abs(pot - pot_theory)
    return overpot
