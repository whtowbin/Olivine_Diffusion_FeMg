# %%w
# from scipy.optimize import fsolve, root
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as interp
#import mc3

# from numba import jit  #
from pykrige import OrdinaryKriging, UniversalKriging 

# import mc3

#%matplotlib inline


def diffusion_kernel(dt, dx):
    """
    returns the relevant kernel for 1D diffusion and a defivative of the Fo#
    dt = time step Seconds
    dx = spatial step Meters

    """
    delta = (dt) / ((dx) ** 2)

    # Diffusion Term
    kernel_1 = np.zeros(3)
    kernel_1[0] = 1
    kernel_1[1] = -2
    kernel_1[2] = 1

    # Central Difference Derivative. This is different than what is used in DIPRA which is a forward difference approx.
    # Remember to divide by 2 in diffusion step
    kernel_2 = np.zeros(3)
    kernel_2[0] = 1
    kernel_2[1] = 0
    kernel_2[2] = -1

    return kernel_1, kernel_2, delta


def VectorMaker(init_Concentration, N_points):
    """
    Creates a profile with a flat initial concentration
    """
    return init_Concentration * np.ones(N_points)


def boundary_cond(bounds_c):
    # C: single concentration at boundary ( in the future we will support changing boundary conditions.
    # This can probably be accomplished with a list that pops next value
    pad = np.ones(3) * bounds_c

    if len(bounds_c) > 1:
        pad = (np.ones(3) * bounds_c[0], np.ones(3) * bounds_c[1])

    return pad


# def old_step_condition(edge_x1, edge_x2, inflect_x, interval_concentrations, dx
# ):
#     """
#     Creates a step function for diffusion models
#     Parameters:
#         X_Intervals - List of tuples - each tuple has start and end point of step
#         interval_concentrations - list of concentrations coresponding to each interval,m_intervals = n_inflection_points+1
#         dx - spacing between x points.

#     Returns
#         step_x - array of x coordinates
#         step_c - array of concentrations
#     """


#     length = abs(x.max() - x.min())
#     num_x = int(round(length / dx_micron,0))
#     step_x, dx_micron = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)


#     length = abs(np.max(X_Intervals) - np.min(X_Intervals))
#     num_x = int(length / dx)

#     segments_x = []
#     segments_c = []

#     # Define concentration steps.
#     for idx, interval in enumerate(X_Intervals):
#         interval_num = int((interval[1] - interval[0]) / length * num_x)

#         # alternate version
#         # int_x = step_x = np.arange(interval[0], interval[1], dx)

#         # Original method- Fails on some odd length profiles.
#         int_x = np.linspace(start=interval[0], stop=interval[1] - dx, num=interval_num)
#         int_c = np.ones_like(int_x) * Interval_concentrations[idx]

#         segments_x.append(int_x)
#         segments_c.append(int_c)

#     step_x = np.concatenate(segments_x)
#     step_c = np.concatenate(segments_c)
#     if len(step_x) < results_length:
#         print("Model Array lenght is too short. Array will be extended by 1")
#         step_x = np.concatenate(step_x, (step_x[-1] + dx))
#         step_c = np.concatenate(step_c, (step_c[-1]))
#     return step_x, step_c
#%%


def step_condition(x_interp, inflection_points, Interval_concentrations):
    """Generates an array of initial concentrations with steps defined at specific distances from the origin (usually the crystal's rim)

    Args:
        x_interp (array): array of distances in microns with even spaces, typically this corresponds to an interpolation of measurement positions output by an interpolation function.
        inflection_points (array or list of floats): positions in microns where inflection point occurs. Will be rounded to nearest position in the x_interp array
        Interval_concentrations (array or list of floats): Concentrations for intervals between step positions 
        dx_micron (_type_): _description_

    Returns:
        _type_: _description_
    """

    inflection_idx = [
        (np.abs(x_interp - inflect)).argmin() for inflect in inflection_points
    ]
    x_segments = np.array_split(x_interp, inflection_idx)
    y_segments = [
        np.ones_like(segment) * Interval_concentrations[idx]
        for idx, segment in enumerate(x_segments)
    ]
    return np.concatenate(y_segments)


# %%

# stability criterion dt*Di/(dx^2) < 0.5
def diffusion_step(
    vector_c_in,
    vector_Fo_in,
    diffusivity_function,
    diff_kernel_1,
    der_kernel_2,
    delta,
    bounds_c,
    bounds_Fo,
):
    """
    Function that takes one step forward for Forsterite dependent diffusion.
    This is an inefficient step so I have tried to move the creation of kernels and vectors outside of this single step
    Parameters:
    bounds_c = tuple of left and right boundary conditions for diffusing species (Fixed bounds at the moment)
    bounds_Fo = tuple of left and right boundary conditions for Fo
    Output:

    """
    pad = np.ones(3)
    pad_c = (bounds_c[0] * pad, bounds_c[1] * pad)
    pad_Fo = (bounds_Fo[0] * pad, bounds_Fo[1] * pad)
    # pad generation can probably be taken out of the loop

    vector_c = np.concatenate([pad_c[0], vector_c_in, pad_c[1]])

    vector_Fo = np.concatenate(
        [pad_Fo[0], vector_Fo_in, pad_Fo[1]]
    )  # This might need to step through a larger matrix of values

    vector_D = diffusivity_function(vector_Fo)

    Diffusion = (np.convolve(vector_c, diff_kernel_1, mode="same") * vector_D)[3:-3]

    Diff_C = np.convolve(vector_c, der_kernel_2, mode="same")[
        3:-3
    ]  # Difference Concentration

    Diff_D = np.convolve(vector_D, der_kernel_2, mode="same")[
        3:-3
    ]  # Difference Concentration

    # Note that the diff terms are divided by 2 to make them central difference approximations of the first derivatve nto forward liek in Girona et al. 2013.
    vector_out = vector_c_in + delta * (Diffusion + (Diff_C * Diff_D) / 2)

    # vector_c_in + delta*(Diffusion + (Diff_C* Diff_D))
    # out = (Diff_C * Diff_D) * delta
    return vector_out


def diffusion_step_Ca(
    vector_c_in,
    diffusivity,
    diff_kernel_1,
    delta,
    bounds_c,
):

    """
    Function that takes one step forward in finite difference model of calcium diffusion in olivine.
    This is an efficient step so I have tried to move the creation of kernels and vectors outside of this single step
    Parameters:
    bounds_c = tuple of left and right boundary conditions for diffusing species (Fixed bounds at the moment)
    bounds_Fo = tuple of left and right boundary conditions for Fo
    Output:

    """
    pad = np.ones(3)
    pad_c = (bounds_c[0] * pad, bounds_c[1] * pad)
    # pad generation can probably be taken out of the loop

    vector_c = np.concatenate([pad_c[0], vector_c_in, pad_c[1]])

    Diffusion = (np.convolve(vector_c, diff_kernel_1, mode="same") * diffusivity)[3:-3]

    vector_out = vector_c_in + delta * Diffusion

    return vector_out


"""
3 Vectors
1) Diffusing Concentration 
2) Diffusing or previously diffused Fo array
3) Diffusivities  Column 
"""
# %%

# %%
"""
One Idea is to do a refining grid search. Do a really sparse dt model with tight dX and then refine. based on when the best fit is bracketed 
"""
# %%
# Elemental Diffusivities


def fo2buffer(T, P_pa, delta, buff = "NNO"):
    """
    

    Args:
        Modified from Dan Rasmussen
        Reference: Frost 1991 Reviews in Mineralogy Volume 25
        T is in Kelvin
        P in Pa
        delta is the delta value from NNO or QFM
        buff - text indicating NNO/FMQ/QFM assumes NNO 

        Returns in Pa
    """

    P = P_pa * 1e-5  # Converts to Bar

    if buff in ["FMQ", "QFM", "fmq", "qfm"]:
        FO2 = 10 ** ((-25096.3 / T) + 8.735 + (0.110 * (P - 1) / T) + delta)
    elif buff in ["NNO", "nno"]:
        FO2 = 10 ** ((-24930 / T) + 9.36 + (0.046 * (P - 1) / T) + delta)
    else:
        return "Buffer not supported must be NNO, FMQ, or QFM"
    return FO2 * 1e5




def D_Fo(T, P, fO2, alpha, beta, gamma, XFo=None, EFo=201000):
    """
    Function that calculates the diffusivity for Forsterite (and Mn) in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of NNO  Pa
        E, - Activation Energy 201000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin
        alpha, -  minimum angle to [100] axis a -- degrees
        beta, - minimum angle to [010] axis b -- degrees
        gamma - minimum angle to [001] axis c -- degrees

    Returns: Diffusivity function That's only input it is:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s

    """

    def D_Func_Fo(XFo):
        """Returns diffusivity and derivative of diffusivity at each point in an olivine for a given oxygen fugacity, proportion of forsterite, activation energy, pressure, gas constant, temperature, and crystallographic orientation."""
        R = 8.3145
        tenterm = 10**-9.21
        fugacityterm = (fO2 / (1e-7)) ** (1.0 / 6.0)
        forsteriteterm = 10 ** (3.0 * (0.9 - XFo))
        D = (
            tenterm
            * fugacityterm
            * forsteriteterm
            * np.exp(-(EFo + 7 * (10**-6 * (P - 10**5))) / (R * T))
        )
        # This next term should be calculated with angle in degrees.
        alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))
        Di = (
            ((1 / 6) * D * (np.cos(alpha_rad) ** 2))
            + ((1 / 6) * D * (np.cos(beta_rad) ** 2))
            + (D * (np.cos(gamma_rad) ** 2))
        )  # Use this term for crystallographic orientation constraints from EBSD.

        return Di  # units of m2/s

    if XFo is not None:
        return D_Func_Fo(XFo)

    return D_Func_Fo


def D_Ni(T, P, fO2, alpha, beta, gamma, XFo=None, EFo=220000):
    """
    Function that calculates the diffusivity for Mn in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of NNO Pa
        E, - Activation Energy 220000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin
        alpha, -  minimum angle to [100] axis a
        beta, - minimum angle to [010] axis b
        gamma - minimum angle to [001] axis c

    Returns: Diffusivity function That's only input it is:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s

    """

    def D_Func_Ni(XFo):
        """Returns diffusivity and derivative of diffusivity at each point in an olivine for a given oxygen fugacity, proportion of forsterite, activation energy, pressure, gas constant, temperature, and crystallographic orientation."""
        R = 8.3145
        tenterm = 3.84 * 10**-9
        fugacityterm = (fO2 / (1e-6)) ** (1.0 / 4.25)
        forsteriteterm = 10 ** (1.5 * (0.9 - XFo))
        D = (
            tenterm
            * fugacityterm
            * forsteriteterm
            * np.exp(-(EFo + 7 * (10**-6 * (P - 10**5))) / (R * T))
        )
        # This next term should be calculated with angle in degrees.

        alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))
        Di = (
            ((1 / 6) * D * (np.cos(alpha_rad) ** 2))
            + ((1 / 6) * D * (np.cos(beta_rad) ** 2))
            + (D * (np.cos(gamma_rad) ** 2))
        )  # Use this term for crystallographic orientation constraints from EBSD.

        return Di  # units of m2/s

    if XFo is not None:
        return D_Func_Ni(XFo)

    return D_Func_Ni


def D_Func_Ca(
    T,
    fO2,
    alpha,
    beta,
    gamma,
):
    """
    Function that calculates the diffusivity for Mn in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of NNO Pa
        E, - Activation Energy 201000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin
        alpha, -  minimum angle to [100] axis a
        beta, - minimum angle to [010] axis b
        gamma - minimum angle to [001] axis c

    Returns: Diffusivity function That's only input is XFo:
                XFo, - Forsterite in Fractional Units This can be a numpy array of the data.

                If XFo is given as an input a diffusivity or an array of diffusivities is returned.
                Diffusivity returned in m2/s):
    """
    R = 8.3145
    fugacityterm = (fO2 / (1e-7)) ** (0.3)

    Da = 16.59 * 10**-12 * fugacityterm * np.exp(-(193000) / (R * T))
    Db = 34.67 * 10**-12 * fugacityterm * np.exp(-(201000) / (R * T))
    Dc = 95.49 * 10**-12 * fugacityterm * np.exp(-(207000) / (R * T))
    # This next term should be calculated with angle in degrees.

    alpha_rad, beta_rad, gamma_rad = np.deg2rad((alpha, beta, gamma))

    Di = (
        Da * (np.cos(alpha_rad) ** 2)
        + Db * (np.cos(beta_rad) ** 2)
        + (Dc * (np.cos(gamma_rad) ** 2))
    )
    # Use this term for crystallographic orientation constraints from EBSD.

    return Di  # units of m2/s


# %%
"""
One idea is to write this function so that if vector_Fo_in is an array it iterates through that for the Fo Vector 

"""


def timestepper(
    vector_c_in,
    vector_Fo_in,
    diffusivity_function,
    bounds_c,
    timesteps,
    dt,
    dx,
    **kwargs
):
    """
    Iterates multiple diffusion steps
    Built for Fo# Diffusion. Can be written for other elements by simultaneous Fo and Trace element diffusion.
    """
    kernel_1, kernel_2, delta = diffusion_kernel(dt=dt, dx=dx)

    # At the moment only handles Fo but should diffuse other elements too with a little modification
    results = np.zeros((timesteps + 1, len(vector_c_in)))
    results[0] = vector_c_in
    for n, _ in enumerate(range(timesteps)):
        vector_c_in = diffusion_step(
            vector_c_in=vector_c_in,
            vector_Fo_in=vector_Fo_in,
            diffusivity_function=diffusivity_function,
            diff_kernel_1=kernel_1,
            der_kernel_2=kernel_2,
            delta=delta,
            bounds_c=bounds_c,
            bounds_Fo=bounds_c,  # This needs to get updated for the Ni, Mn, or
        )

        vector_Fo_in = (
            vector_c_in  # This step needs refining to evaluate other elements.
        )
        results[n + 1] = vector_Fo_in
    return results


def timestepper_Ni_Mn(
    vector_c_in,
    vector_Fo_in,
    diffusivity_function,
    bounds_c,
    timesteps,
    dt,
    dx,
    bounds_Fo=None,
    **kwargs
):
    """

    vector_Fo_in: If running for Ni, or Mn this must be an 2D array of size(n,v) that where n is the total number of timesteps and v is the size of the Fo vector.
    Iterates multiple diffusion steps
    Built for Fo# Diffusion. Can be written for other elements by simultaneous Fo and Trace element diffusion.
    """
    kernel_1, kernel_2, delta = diffusion_kernel(dt=dt, dx=dx)

    # At the moment only handles Fo but should diffuse other elements too with a little modification
    results = np.zeros((timesteps + 1, len(vector_c_in)))
    results[0] = vector_c_in

    if (
        bounds_Fo == None and len(vector_Fo_in) > 1
    ):  # assumes That if Bounds Fo isnt input Fo is diffusing species
        bounds_Fo = (vector_Fo_in[0], vector_Fo_in[1])

        for n, _ in enumerate(range(timesteps)):

            vector_c_in = diffusion_step(
                vector_c_in=vector_c_in,
                vector_Fo_in=vector_Fo_in[n, :],
                diffusivity_function=diffusivity_function,
                diff_kernel_1=kernel_1,
                der_kernel_2=kernel_2,
                delta=delta,
                bounds_c=bounds_c,
                bounds_Fo=bounds_Fo,
            )

            results[n + 1] = vector_c_in
        return results

    else:
        vector_c_in = diffusion_step(
            vector_c_in=vector_c_in,
            vector_Fo_in=vector_Fo_in,
            diffusivity_function=diffusivity_function,
            diff_kernel_1=kernel_1,
            der_kernel_2=kernel_2,
            delta=delta,
            bounds_c=bounds_c,
            bounds_Fo=bounds_c,
        )
        return results


# %%


def Best_fit_R2(results, data_interp, dt):
    # Should be Chi2 but looks more like likelihood?

    residual = results - data_interp
    sum_r2 = np.sum(residual**2, axis=1)
    idx_min = np.argmin(sum_r2)

    # sum_r2[idx_min] * 1.05

    time = (idx_min + 1) * dt  # seconds
    time_days = time / (60 * 60 * 24)
    return time, idx_min, sum_r2


def Best_fit_Chi2(results, data_interp, sigma, dt, sigma_min=1e-4, scale_error=False):
    # This minimizes for sum of residuals^2/sigma

    residual = results - data_interp
    sum_Chi2 = np.sum((residual**2) / (sigma + sigma_min) ** 2, axis=1)
    idx_min = np.argmin(sum_Chi2)

    min_Chi2 = sum_Chi2.min()
    Chi2_Bound = (2 * (len(data_interp) - 1)) ** (1 / 2)
    reduced_chi2 = 1

    if scale_error == True:
        reduced_chi2 = sum_Chi2 / (len(data_interp) - 1)

    Chi2_Bound = (
        Chi2_Bound / reduced_chi2
    )  # Scale Chi_Squared Boundary based on reduced Chi2 scaled to 1

    Chi2_error = np.where(
        (np.round(sum_Chi2, 1) < np.round(min_Chi2 + Chi2_Bound, 1))
        & (np.round(sum_Chi2, 1) > np.round(min_Chi2 + Chi2_Bound, 1) - 1)
    )
    # Trying to reduce errors by taking out rounding.
    # Chi2_error = np.where(
    #     (sum_Chi2 < (min_Chi2 + Chi2_Bound)) & (sum_Chi2 > (min_Chi2 + Chi2_Bound - 1))
    # )

    Chi2_error_idx_low = Chi2_error[0].min()
    Chi2_error_idx_high = Chi2_error[0].max()
    fit_idx = np.array(
        [idx_min, Chi2_error_idx_low - idx_min, Chi2_error_idx_high - idx_min]
    )

    time = (fit_idx + 1) * dt  # seconds
    time_days = time / (60 * 60 * 24)

    return time, (idx_min, Chi2_error_idx_low, Chi2_error_idx_high), sum_Chi2


# np.where((Z[2]< 155)) Use to find where minimization intersects (2(n-m))^(1/2)
# n= number of points; m = number of parameters being fit
# %%
# Write Pad function. Each Step needs a pad
# 1) Constant Boundary - Diffusion
# 2) Constant Boundary - No_Diffusion at edge
# 3) Changing Boundary - Ascent Path
# %%


# %%


def Krige_Interpolate(
    X, Y, new_X, variogram_model="linear", variogram_parameters={"slope": 1e-4, "nugget": 1e-5},
):

    # uk = OrdinaryKriging(
    uk = UniversalKriging(
        X,
        np.zeros(X.shape),
        Y,
        pseudo_inv=True,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        # nlags = nlags
    )

    y_pred, y_std = uk.execute("grid", new_X, np.array([0.0]))
    y_pred = np.squeeze(y_pred)
    y_std = np.squeeze(y_std)

    return new_X, y_pred, y_std


#%%

"""
Find max time steps from 3 point diffusion model 
1   
"""
# TODO Sort Variables by whether they will be sampled with PyMC or not.
# Write subfunctions to handle general and specific Diffusion model set ups.

# I should annotate these functions better. 
def Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calcualate the max timesteps based on the slowest diffusivity I expect.
    x_interp,  # evenly spaced positional array
    data_interp,
    std_interp,
    dx_micron,
    dt,
    output_full=False,
    **kwargs
):

    T, P, fO2, inflect_x, low_x_idx, high_x_idx, edge_c, center_c, inflection_c = p

    D_FO_Func = D_Fo(
        T=T,
        P=P,
        fO2=fO2,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        EFo=EFo,
    )
    dx = dx_micron * 1e-6  # m

    # sets up a single stairstep for diffusion models
    #
    inflection_points = inflect_x
    if not isinstance(inflect_x, (list, tuple)):

        inflection_points = [inflect_x]

    Interval_concentrations = [edge_c, center_c]
    if not np.isnan(inflection_c):
        if not isinstance(inflection_c, (list, tuple)):
            inflection_c = [inflection_c]
        Interval_concentrations[1:1] = inflection_c

    step_c = step_condition(
        x_interp, inflection_points, Interval_concentrations
    )

    #  Only implmented for Fo# Zoning at the moment.

    Fo_diffusion_results = timestepper(
        vector_c_in=step_c,
        vector_Fo_in=step_c,
        diffusivity_function=D_FO_Func,
        bounds_c=(edge_c, center_c),
        timesteps=timesteps,
        dx=dx,
        dt=dt,
    )
    # return Fo_diffusion_results
    time, idx_min, sum_r2 = Best_fit_Chi2(
        Fo_diffusion_results, data_interp, std_interp, dt, scale_error=False, **kwargs
    )

    if output_full:
        return time, idx_min, sum_r2, Fo_diffusion_results
    return Fo_diffusion_results[idx_min]


# %%

"""
Conditions to solve for:
Position of edge accounts for crystallization etc... Maybe not as important 
Inflection Point for step 

Initial concentration 
Edge Concentration

Is there some way to constrain simultaneous crystallization and diffusion by shape?

Kd for Olivine that have lost central concentration 
"""


"""
Modeling should probably include a dP/dT term but I might also want to include a pT

This also requires an evolution of the diffusivity function at each timestep 

"""
# %%

# %%

# %%

# %%


def D_Fo_For_PT_Sampling(T, P, fO2, EFo=201000):
    """
    Function that calculates the diffusivity for Forsterite (and Mn) in olivine.
    Returns a function that only requires XFo = XMg/(XMg+XFe)
    this assumes that the only thing changing during diffusion is XFo.
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    Parameters:
        fO2, - Oxygen Fugacity with a reference of in Pa
        E, - Activation Energy 201000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin


    """

    R = 8.3145
    tenterm = (
        10**-9.21
    )  # I am not modeling uncertainty in this term but I probably should
    fugacityterm = (fO2 / (1e-7)) ** (1.0 / 6.0)

    D = (
        tenterm
        * fugacityterm
        * np.exp(-(EFo + 7 * (10**-6 * (P - 10**5))) / (R * T))
    )
    return D  # units of m2/s


# Notes
#  #Delta QFM between -2 and +1
# P T Paths Maybe give 1 pt path and provide one offset number




#%%

# def model_diffusion(
#     profile_name, data_db=Ol_Data, parameter_db=ol_param_db, Total_time_days=200
# ):
#     # This function works for a single step, change how edge_x params are input to accept tuples to change for multiple steps
#     parameters = parameter_db.loc[(ol_param_db.File_Name == profile_name)]

#     # load parameters from database
#     theta = parameters.prof_angle.item()
#     phi1 = parameters.phi1.item()
#     Phi = parameters.Phi.item()
#     phi2 = parameters.phi2.item()

#     dx_micron = parameters.dx.item()
#     dt = parameters.dt.item()
#     T_Celsius = parameters["T"].item()
#     T = T_Celsius + 273.15  # T in kelvin
#     P = parameters.P.item()
#     fO2_dQFM = parameters.FO2_dQFM.item()

#     # "Category"

#     # This section interprets the string in the database row and converts it to segments to make the initial step function
#     inflection_x = parameters.inflection_x.item()
#     if isinstance(inflection_x, str):
#         inflection_x = ast.literal_eval(inflection_x)

#     inflection_c = parameters.inflection_c.item()
#     if isinstance(inflection_c, str): # If several numbers are given as strings in list form.
#         inflection_c = ast.literal_eval(inflection_c)
#         inflection_c = [x / 100 for x in inflection_c]

#     if isinstance(inflection_c, (int, float)): # If only one number is given
#         inflection_c = inflection_c / 100 # Converts Fo scaled to max of 100 to fractional units

#     edge_x1, edge_x2 = parameters.edge_x1.item(), parameters.edge_x2.item()
#     edge_c, center_c = parameters.edge_c.item(), parameters.center_c.item()

#     alpha, beta, gamma = g.vector_direction(theta, phi1, Phi, phi2)


#     x, y = get_C_prof(profile_name, data_db)
#     edge_c, center_c = edge_c / 100, center_c / 100        # Concentrations measured in units scaled to max of 100
#     y = y / 100         # Concentrations measured in units scaled to max of 100

#     fO2 = Ol_Diff.fo2buffer(T, P, delta=fO2_dQFM, buff="FMQ")

#     # generate step function
#     length = abs(x.max() - x.min())
#     num_x = int(round(length / dx_micron, 0))
#     step_x, dx_micron = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)

#     # interpolate data and uncertainty to step spacing.
#     X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
#         x,
#         y,
#         step_x,
#         variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
#     )

#     low_x_idx = (np.abs(step_x - edge_x1)).argmin()
#     high_x_idx = (np.abs(step_x - edge_x2)).argmin()

#     model_x = step_x[low_x_idx : high_x_idx + 1]
#     data_interp = Y_interp[low_x_idx : high_x_idx + 1]
#     std_interp = Y_interp_std[low_x_idx : high_x_idx + 1]

#     Total_time = Total_time_days * 24 * 60 * 60  # seconds
#     timesteps = int(Total_time / dt)

#     EFo = 201000

#     p = (T, P, fO2, inflection_x, low_x_idx, high_x_idx, edge_c, center_c, inflection_c)

#     time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
#         # Fo_diffusion_results = Ol_Diff.Diffusion_call(
#         p,
#         alpha,
#         beta,
#         gamma,
#         EFo,
#         timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
#         model_x,
#         data_interp,
#         std_interp,
#         dx_micron,
#         dt=dt,
#         output_full=True,
#     )
#     # return Fo_diffusion_results
#     return {
#         "time": time,
#         "idx_min": idx_min,
#         "sum_r2": sum_r2,
#         "model_x": model_x,
#         "Fo_diffusion_results": Fo_diffusion_results,
#         "x": x,
#         "y": y,
#         "X_interp": X_interp,
#         "Y_interp": Y_interp,
#         "Y_interp_std": Y_interp_std,
#         "T_Celsius": T_Celsius,
#         "P": P,
#         "fO2": fO2,
#         "profile_name": profile_name,
#         "Category": parameters.Category.item(),
#     }
