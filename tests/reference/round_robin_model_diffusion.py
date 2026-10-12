"""Verbatim copy of get_C_prof and model_diffusion from
"Round_Robin_Diffusion_Modeling Part 2.py" at commit 6492a1d (2026-10-11).

The reference side of tests/test_differential.py. Do not edit: it must stay what
the research script ran. model_diffusion reads the module globals Ol_Diff and
ol_param_db (the latter in its row lookup), so the test sets ol_param_db first.
"""

import ast
from typing import Any

import numpy as np

from olivine_diffusion import engine as Ol_Diff

Ol_Data: Any = None
ol_param_db: Any = None


def get_C_prof(prof_name, DF, Element="Fo#", X="Distance µm"):
    prof = DF.loc[
        (DF.Profile_Name == prof_name) & (DF.Marked_bad != "bad") & (DF.Ignore != "yes")
    ].sort_values("Distance µm")

    distance_um = prof[X]
    concentration = prof[Element]
    return distance_um.to_numpy(), concentration.to_numpy()


def model_diffusion(
    profile_condition_name,
    data_db=Ol_Data,
    parameter_db=ol_param_db,
    kriging_variogram_model="linear",
    kriging_variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
    kriging_nlags=None,
    activation_EFo=201000,
    Total_time_days=200,
    uniform_uncert=0.00001,  # Fractional Fo units
):
    # This function works for a single step, change how edge_x params are input to accept tuples to change for multiple steps

    parameters = parameter_db.loc[(ol_param_db.Sample == profile_condition_name)]

    # load parameters from database
    # theta = parameters.prof_angle.item()
    # phi1 = parameters.phi1.item()
    # Phi = parameters.Phi.item()
    # phi2 = parameters.phi2.item()

    profile_name = parameters.File_Name.item()
    alpha = parameters.alpha.item()
    beta = parameters.beta.item()
    gamma = parameters.gamma.item()

    dx_micron = parameters.dx_um.item()
    dt = parameters.dt_s.item()
    T_Celsius = parameters["T_C"].item()
    T = T_Celsius + 273.15  # T in kelvin
    P = parameters.P_Pa.item()
    # fO2_dQFM = parameters.FO2_dQFM.item()
    fO2 = parameters.fO2_Pa.item()

    # This function
    inflection_x = parameters.inflection_x.item()
    if isinstance(inflection_x, str):
        inflection_x = ast.literal_eval(inflection_x)

    inflection_c = parameters.inflection_c.item()
    if isinstance(inflection_c, str):
        inflection_c = ast.literal_eval(inflection_c)
        inflection_c = [x / 100 for x in inflection_c]

    if isinstance(inflection_c, (int, float)):
        inflection_c = inflection_c / 100

    edge_x1, edge_x2 = parameters.edge_x1.item(), parameters.edge_x2.item()
    edge_c, center_c = parameters.edge_c.item(), parameters.center_c.item()

    # alpha, beta, gamma = g.vector_direction(theta, phi1, Phi, phi2)

    x, y = get_C_prof(profile_name, data_db)
    edge_c, center_c = edge_c / 100, center_c / 100
    y = y / 100

    # fO2 = Ol_Diff.fo2buffer(T, P, delta=fO2_dQFM, buff="FMQ")

    # generate step function
    length = abs(x.max() - x.min())
    num_x = int(round(length / dx_micron, 0))
    step_x, dx_micron = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)

    # interpolate data and uncertainty to step spacing.
    X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
        x,
        y,
        step_x,
        variogram_model=kriging_variogram_model,
        variogram_parameters=kriging_variogram_parameters,
        # nlags = kriging_nlags
    )

    low_x_idx = (np.abs(step_x - edge_x1)).argmin()
    high_x_idx = (np.abs(step_x - edge_x2)).argmin()

    model_x = step_x[low_x_idx : high_x_idx + 1]
    data_interp = Y_interp[low_x_idx : high_x_idx + 1]
    std_interp = Y_interp_std[low_x_idx : high_x_idx + 1] + uniform_uncert

    Total_time = Total_time_days * 24 * 60 * 60  # seconds
    timesteps = int(Total_time / dt)

    EFo = activation_EFo

    p = (T, P, fO2, inflection_x, low_x_idx, high_x_idx, edge_c, center_c, inflection_c)
   
    # Ol_Diff.D_FO_Func = D_Fo(
    #     T=T,
    #     P=P,
    #     fO2=fO2,
    #     alpha=alpha,
    #     beta=beta,
    #     gamma=gamma,
    #     EFo=EFo,
    # )
    time, idx_min, sum_chi2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
        # Fo_diffusion_results = Ol_Diff.Diffusion_call(
        p,
        alpha,
        beta,
        gamma,
        EFo,
        timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
        model_x,
        data_interp,
        std_interp,
        dx_micron,
        dt=dt,
        output_full=True,
    )
    # return Fo_diffusion_results
    return {
        "time": time,
        "idx_min": idx_min,
        "dt": dt,
        "sum_chi2": sum_chi2,
        "reduced_chi2": sum_chi2 / (len(data_interp) - 1),
        "model_x": model_x,
        "Fo_diffusion_results": Fo_diffusion_results,
        "x": x,
        "y": y,
        "X_interp": X_interp,
        "Y_interp": Y_interp,
        "Y_interp_std": Y_interp_std + uniform_uncert,
        "T_Celsius": T_Celsius,
        "P_MPa": P,
        "fO2_MPa": fO2,
        "profile_name": profile_name,
        "model_run_name": profile_condition_name,
        "descriptive_title": parameters.Descriptive_Name.item(),
        "Uncert_Category": parameters.Uncert_Category.item(),
        "Category": parameters.Category.item(),
        "edge_x_micron": [edge_x1, edge_x2],
        "edge_c": [edge_c, center_c],
        "inflection_x_µm": inflection_x,
        "inflection_c": inflection_c,
    }
