# %%
# from Users.henry.Python Files.Electrical Conductivity SIMS Data.NS_ConductivityOlivines import Sample_Interpolate
# import Fe_Mg_Diffusion_Convolution


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.interpolate as interp
from matplotlib.backends.backend_pdf import PdfPages

# from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (
    MultipleLocator,
    FormatStrFormatter,
    AutoMinorLocator,
    ScalarFormatter,
)


import Fe_Mg_Diffusion_Convolution_Streamlined as Ol_Diff
# import EBSD_Profile_Rotation as EBSD

import seaborn as sns
import itertools
from cycler import cycler
import ast
import textwrap

import importlib
from scipy.interpolate import splrep, splev

# import janitor


# from pykrige import OrdinaryKriging

from matplotlib import rc

rc("axes", linewidth=2)
rc("font", weight="bold", stretch="condensed", size=14, family="Avenir")

# %%

excel_db_path = "Diffusion Round Robin/Round_Robin_Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"

ol_param_db = pd.read_excel(
    excel_db_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

# %%
excel_path = "Diffusion Round Robin/Diffusion Round Robin Data Reorg.xlsx"

Ol_Data = pd.read_excel(
    excel_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

Ol_profiles = Ol_Data.loc[
    # (Ol_Data.Profile == "yes")
    (Ol_Data.Marked_bad != "bad") & (Ol_Data.Ignore != "yes")
]


# %%
# Original

sample_names = Ol_profiles.Profile_Name.unique()

# %%


def get_C_prof(prof_name, DF, Element="Fo#", X="Distance µm"):
    prof = DF.loc[
        (DF.Profile_Name == prof_name) & (DF.Marked_bad != "bad") & (DF.Ignore != "yes")
    ].sort_values("Distance µm")

    distance_um = prof[X]
    concentration = prof[Element]
    return distance_um.to_numpy(), concentration.to_numpy()


def plot_c_prof_diff(prof_name, DF, Element="Fo#", diff_number=1, ax=None):
    if ax is None:
        ax = plt.gca()
    x, y = get_C_prof(prof_name, DF, Element)
    diff_c = np.diff(y, n=diff_number) / np.diff(x, n=diff_number)
    plt.plot(diff_c)


# %%


def select_data(DF, selection_dict):
    query = " and ".join(
        [f"{key} == {repr(value)}" for key, value in selection_dict.items()]
    )
    # query = ' and '.join(['{} == {}'.format(k, repr(v)) for k, v in m.items()])
    new_df = DF.query(query)
    return new_df


# %%


def plot_trace(
    DF,
    Selection_dict,
    Element_x="Fo#",
    Element_y="NiO",
    ax=None,
    Category=None,
    Distance_Color=True,
    Marker=".",
    Point_color="tab:blue",
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    DF = select_data(DF, Selection_dict)

    prof = DF.loc[(DF.Marked_bad != "bad") & (DF.Ignore != "yes")]

    color = Point_color
    if Distance_Color is True:
        color = prof["Edge distance x mm"]

    ax.scatter(
        x=prof[Element_x],
        y=prof[Element_y],
        c=color,
        **kwargs,
    )

    ax.set_xlabel(Element_x, weight="bold")
    ax.set_ylabel(Element_y, weight="bold")
    return ax


def plot_2_elements(
    Ol_Profiles, Sample_name, element_1="Fo#", element_2="CaO", ax=None
):
    if ax == None:
        ax = plt.gca()
    # fig, ax1 = plt.subplots()
    ax1 = ax
    # plt.title(Sample_name)
    color = "tab:red"
    ax1.set_xlabel("Micron (µm)")
    ax1.set_ylabel(element_1, color=color)

    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(which="both", width=2)
    ax.tick_params(which="major", length=7)
    ax.tick_params(
        which="minor",
        length=4,
    )

    x_1, y_1 = get_C_prof(Sample_name, Ol_Profiles, Element=element_1)
    ax1.plot(x_1, y_1, color=color, marker="o", linestyle="dashed")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel(element_2, color=color)
    x_2, y_2 = get_C_prof(Sample_name, Ol_Profiles, Element=element_2)
    ax2.plot(x_2, y_2, color=color, marker="s", linestyle="dashed")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.tight_layout()  # otherwise the right y-label is slightly clipped

    return ax1, ax2


# %%
def Calculate_Gradient(
    profile_name,
    data_db=Ol_Data,
    ax=None,
    dx_micron=0.5,
):
    if ax is None:
        ax = plt.gca()

    ax1 = ax
    # plt.title(Sample_name)
    color = "tab:red"
    ax1.set_xlabel("Micron (µm)")
    ax1.set_ylabel("Fo#")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    x, y = get_C_prof(profile_name, data_db)
    y = y

    # generate step function
    length = abs(x.max() - x.min())
    num_x = int(round(length / dx_micron, 0))
    step_x, dx_micron = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)

    f = splrep(x, y, k=5, s=3)

    ax1.plot(x, y, label="noisy data")
    ax1.plot(step_x, splev(step_x, f), label="fitted")

    ax2.set_ylabel("Derivatives")
    ax2.plot(
        step_x, splev(step_x, f, der=1), label="1st derivative", linestyle="dashed"
    )
    ax2.plot(
        step_x,
        splev(step_x, f, der=2) * 10,
        label="2nd derivative * 10",
        linestyle="dotted",
    )

    handles, labels = [
        (a + b)
        for a, b in zip(
            ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()
        )
    ]
    plt.legend(handles, labels)
    return ax


# %%
"""
Fo# Diffusion Modeling
"""
# %%


"""
Objects to make Inputs:
profile_name
param db
data_db

dx_micron
dt 

objects to make outputs:
Output the number of timesteps, the dt*D and the initial DT for each. 

# Make three functions:
#  one to load data from DB and process it
#  One to model uncertainty in time scale given PT path, number of timesteps, dt, D
#  a plotting function for each sample 


"""
# %%

# To Do convert data to fraction Fo not normed to 100.

importlib.reload(Ol_Diff)


# import Fe_Mg_Diffusion_Convolution_Streamlined as Ol_Diff
def model_diffusion(
    profile_condition_name,
    data_db=Ol_Data,
    parameter_db=ol_param_db,
    kriging_variogram_model="linear",
    kriging_variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
    kriging_nlags=None,
    activation_EFo=201000,
    Total_time_days=200,
    uniform_uncert = 0.00001, # Fractional Fo units
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
    Uncert_Category = parameters.Uncert_Category.item()
    # "Category"

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
        "sum_chi2": sum_chi2,
        "reduced_chi2": sum_chi2/(len(data_interp)-1),
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
        "Uncert_Category": Uncert_Category,
        "Category": parameters.Category.item(),
        "edge_x_micron": [edge_x1, edge_x2],
        "edge_c": [edge_c, center_c],
        "inflection_x_µm": inflection_x,
        "inflection_c": inflection_c,
    }


# %%
def plot_diffusion_results(
    Model_dict, ax=None, tick_mark_major_multiple=10, tick_mark_minor_multiple=2.5
):
    if ax is None:
        ax = plt.gca()
    ax.plot(
        Model_dict["x"],
        Model_dict["y"] * 100,
        marker="o",
        linewidth=2,
        markersize=12,
        label="Data",
        linestyle="dashed",
        color="#F47E1F",
    )
    ax.plot(
        Model_dict["X_interp"],
        Model_dict["Y_interp"] * 100,
        linestyle="dashed",
        linewidth=1,
        color="g",
        label="Interpolation",
    )

    ax.plot(
        Model_dict["X_interp"],
        (Model_dict["Y_interp"] + Model_dict["Y_interp_std"] * 2) * 100,
        linestyle="dashed",
        linewidth=1,
        color="k",
        label="Interpolation \n  Uncertainty",
    )
    ax.plot(
        Model_dict["X_interp"],
        (Model_dict["Y_interp"] - Model_dict["Y_interp_std"] * 2) * 100,
        linestyle="dashed",
        linewidth=1,
        color="k",
    )

    ax.plot(
        Model_dict["model_x"],
        Model_dict["Fo_diffusion_results"][0] * 100,
        linewidth=2,
        label="Initial \n Condition",
        color="#1B878C",
        linestyle="dashed",
    )

    idx_min = Model_dict["idx_min"][0]
    ax.plot(
        Model_dict["model_x"],
        Model_dict["Fo_diffusion_results"][idx_min] * 100,
        linewidth=6,
        alpha=0.8,
        label="Best fit",
        color="#1FBECF",
    )

    ax.set_xlabel("Distance from Rim µm", weight="bold", fontsize=20)
    ax.set_ylabel("Fo#", weight="bold", fontsize=20)

    time_days = Model_dict["time"] / (24 * 60 * 60)
    ax.set_title(
        Model_dict["profile_name"]
        + "\n"
        + f"Best fit time: {round(time_days[0], 1)}±{round(np.abs(time_days[1:None]).max(), 1)} days"
        + f" Temperature: {Model_dict['T_Celsius']} ˚C",
        weight="bold",
    )

    

    parameter_annotation = f"Best fit time: {time_days[0]:.1f} ({time_days[1:None].min():.1f}/+{time_days[1:None].max():.1f}) Temperature: {Model_dict['T_Celsius']:.1f} ˚C, Pressure: {Model_dict['P_MPa']:.2e} MPa, fO2:{Model_dict['fO2_MPa']:.2e} MPa, Minimum Chi2: {Model_dict['sum_chi2'].min():.2f}, Reduced_Chi2 {Model_dict['reduced_chi2'].min():.2f} \n bounds_edge_positions: {Model_dict["edge_x_micron"]} µm, bounds_edge_concentration: {Model_dict["edge_c"]} Fo, Inflection_point_positions: {Model_dict["inflection_x_µm"]} µm, Central Interval Concentrations: {Model_dict["inflection_c"]} Fo "
    wrapped_text = textwrap.fill(parameter_annotation, width=150)
    # Place the text using annotate
    # xy=(0.5, 0) specifies the reference point at the bottom-center of the axes (axes fraction coords)
    # xytext=(0, -40) specifies an offset of -40 points vertically downwards from the reference
    ax.annotate(
        wrapped_text,
        xy=(0.5, 0),
        xytext=(0, -120),
        xycoords="axes fraction",
        textcoords="offset points",
        size=12,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Adjust layout to make room for the text
    plt.subplots_adjust(bottom=0.2)
    # Make a plot with major ticks that are multiples of 20 and minor ticks that
    # are multiples of 5.  Label major ticks with '%d' formatting but don't label
    # minor ticks.
    ax.xaxis.set_major_locator(MultipleLocator(tick_mark_major_multiple))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(tick_mark_minor_multiple))

    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.tick_params(which="both", width=2)
    ax.tick_params(which="major", length=7)
    ax.tick_params(
        which="minor",
        length=4,
    )
    # Color for Plotting P2O3 or Al2O3 is '#2FA048'
    ax.legend()

    ax.set_xlim(-2)

    return ax


# %%

# %%
ol_param_db = pd.read_excel(
    excel_db_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

# sample_name = "KS20-527_2-transect"  #
# sample_name = "KS20-527_5-transect-2"
sample_name = "KS20-527-8_transect-3"

# sample_name = "SH63olv066"
# sample_name = "SH63olv084"

# sample_name = "AZ18 WHT06_ol43_prof 1"

Model_dict = model_diffusion(
    sample_name,
    data_db=Ol_Data,
    parameter_db=ol_param_db,
    Total_time_days=365*2,
    # kriging_variogram_model="gaussian",
    # kriging_variogram_parameters=None,
    kriging_variogram_model="linear",
    kriging_variogram_parameters= {"slope": ((5/100)**2)/10, "nugget":  (.2/100)**2},
    uniform_uncert = 0.0005,
 ) 
#kriging_variogram_parameters={"slope": 1e-4, "nugget": 5e-4}

# I think a slope of 5% within 10 µm  should be (5/100)**2/10
# and a Nugget of .2 Fo is (.2/100)**2} since units are likely in variance so squared.
# Model_dict = model_diffusion(sample_name, Total_time_days=365, kriging_variogram_model="linear", kriging_variogram_parameters=None, kriging_nlags = 2)
# %%
element_2 = "NiO"  #
fig, ax = plt.subplots(figsize=(12, 9))
plot_diffusion_results(
    Model_dict, tick_mark_major_multiple=20, tick_mark_minor_multiple=10, ax=ax
)
# Calculate_Gradient(sample, ax = ax, dx_micron=5) # This should probably be more explicit where I can turn off smoothing and interpolation


ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:red"
ax2.set_ylabel(element_2, color=color)
x_2, y_2 = get_C_prof(sample_name, Ol_profiles, Element=element_2)
ax2.plot(x_2, y_2, color=color, marker="s", linestyle="dashed", label=element_2)
ax2.tick_params(axis="y", labelcolor=color)
# ax.set_xticks(np.arange(0,350,50))
plt.legend()


# ax.xaxis.set_major_locator(MultipleLocator(20))
plt.savefig(f"{sample_name}_diffusion_plus_{element_2}.svg")

# %%

ol_param_db = pd.read_excel(
    excel_db_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

files_to_model = ol_param_db.Sample.dropna().unique()
sample_models = {}

for idx, sample in enumerate(files_to_model):
    try:
        Model_dict = model_diffusion(
            sample,
            data_db=Ol_Data,
            parameter_db=ol_param_db,
            Total_time_days=2*365,
            kriging_variogram_model="linear",
            kriging_variogram_parameters={
                "slope": ((5 / 100) ** 2 )/ 10,
                "nugget": (0.2 / 100) ** 2,
            },
            uniform_uncert = 0.0005,
        )  
        sample_models[sample] = Model_dict

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_diffusion_results(
            Model_dict, tick_mark_major_multiple=20, tick_mark_minor_multiple=10, ax=ax
        )
        # Calculate_Gradient(sample, ax = ax)
        plt.savefig(f"Xenolith_Ol_Diffusion_Models/{sample}_model.svg")
        plt.savefig(f"{sample}_model.png")
        print(f"{idx + 1} of {len(files_to_model)} completed")
    except:
        print(f"An Exception Occcured and {sample} Could not be calculated")
    Model_dict["time"] / (60 * 60 * 24)

# %%sample_models
# %%


# #%%
# model_times_1280= model_times
# %%
def D_Fo_For_PT_Uncert_Sampling(T, P, fO2_Pa, EFo=201000):
    """
    Function that calculates the diffusivity for Forsterite excluding terms for xFo and crystallographic Orientation
    If Temperature, Pressure, or Oxygen fugacity change significantly
    during the diffusion period consider inputting all terms in main function.

    This function should only be used to scale diffusivity in monte carlo samples relative to values used in main model fits. 

    Parameters:
        fO2, - Oxygen Fugacity in Pa
        E, - Activation Energy 201000. # J/mol
        P, - Pressure in Pa
        R, Ideal Gas Constant 8.3145 # J/mol
        T,  - temperature in absolute degrees Kelvin


    """
    R = 8.3145
    tenterm = (
        10**-9.21
    )  # I am not modeling uncertainty in this term but I probably should
    fugacityterm = (fO2_Pa / (1e-7)) ** (1.0 / 6.0)

    D = tenterm * fugacityterm * np.exp(-(EFo + 7 * (10**-6 * (P - 10**5))) / (R * T))

    return D  # units of m2/s

#%%
# Monte Carlo Modeling of Diffusivites for error propagation 


def sample_PT_fO2(T_C, T_C_uncert, P_Pa, P_Pa_uncert, fO2_LogPa, fO2_LogPa_uncert, return_fO2_in_Pa = True, n_samples = 1000000):
    T_C_rand = np.random.normal(T_C, T_C_uncert, n_samples)
    P_Pa_rand = np.random.normal(P_Pa, P_Pa_uncert, n_samples)
    fO2_LogPa_rand = np.random.normal(fO2_LogPa, fO2_LogPa_uncert, n_samples)

    if return_fO2_in_Pa is True:
        fO2_Pa_rand = 10**fO2_LogPa_rand
    else:
        fO2_Pa_rand = None


    return {"T_C_rand": T_C_rand, "P_Pa_rand": P_Pa_rand, "fO2_LogPa_rand":fO2_LogPa_rand, "fO2_Pa_rand":fO2_Pa_rand}
#%%

# Samples 1 - 3 
Sample_1to3 = sample_PT_fO2(T_C=1200,T_C_uncert=30, P_Pa=42*10**6, P_Pa_uncert=10 * 10**6, fO2_LogPa=(-7.86+5), fO2_LogPa_uncert=0.2)

Default_D_FO_1to3 = D_Fo_For_PT_Uncert_Sampling(T= 1200, P = 42*10**6, fO2_Pa= 10**(-7.86+5))
Sample_1to3_Diffusivity_array =D_Fo_For_PT_Uncert_Sampling(T= Sample_1to3['T_C_rand'], P = Sample_1to3['P_Pa_rand'] , fO2_Pa = Sample_1to3['fO2_Pa_rand'])

Sample_1to3_Diffusivity_array_scale = Sample_1to3_Diffusivity_array/ Default_D_FO_1to3



#%%

# Samples 4 - 5
Sample_4to5 = sample_PT_fO2(T_C=1102,T_C_uncert=30, P_Pa=70*10**6, P_Pa_uncert=10 * 10**6, fO2_LogPa=(-8.74+5), fO2_LogPa_uncert=0.2)

Default_D_FO_4to5 = D_Fo_For_PT_Uncert_Sampling(T= 1102, P = 70*10**6, fO2_Pa= 10**(-8.74+5))

Sample_4to5_Diffusivity_array =D_Fo_For_PT_Uncert_Sampling(T= Sample_4to5['T_C_rand'], P = Sample_4to5['P_Pa_rand'] , fO2_Pa = Sample_4to5['fO2_Pa_rand'])

Sample_4to5_Diffusivity_array_scale = Sample_4to5_Diffusivity_array / Default_D_FO_4to5


Diffusivity_Scale_Dict = {"Sample_1to3": Sample_1to3_Diffusivity_array_scale, "Sample_4to5": Sample_4to5_Diffusivity_array_scale }
# %%


# %%


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


# Diffusivity_Scale = Diffusivity_array.mean() / Default_D

# Diffusivity_std = (Diffusivity_array.mean() / Default_D) * (
#     Diffusivity_array.std() / Diffusivity_array.mean()
# )


# %%


"""
total time = iterations * dt 

solving for new dt. 
dt_i*dx_i = dt_n * dx_n 

"""
model_times = {}
model_category = []
model_title = []
PT_corrected_times = []

for sample in sample_models.keys():
    model_times[sample] = sample_models[sample]["time"]
    model_category.append(sample_models[sample]['model_run_name'])
    model_title.append(sample_models[sample]['model_run_name'])
    match sample_models[sample]['Uncert_Category']:
        case "Diffusion_Uncert1":
            Diffusivity_Scale = Sample_1to3_Diffusivity_array_scale
        case "Diffusion_Uncert2":
            Diffusivity_Scale = Sample_1to3_Diffusivity_array_scale

    PT_corrected_times.append(
        np.array(
            sample_models[sample]["time"][0]
            / Diffusivity_Scale
            / (60 * 60 * 24)
        )
    )


DB_of_all_time = pd.DataFrame(PT_corrected_times)
DB_of_all_time['Category'] = model_category
DB_of_all_time.set_index('Category', inplace=True)

Timescales = pd.DataFrame(np.quantile(DB_of_all_time,[0.5, 0.025, 0.3333, 0.6666, 0.975], axis = 1))
Timescales['Quantiles'] = ['50% Quantile', '2.5% Quantile', '33.33% Quantile', '66.66% Quantile', '97.5% Quantile']
Timescales.set_index('Quantiles', inplace=True)

Timescales.columns =model_category



#%%
from matplotlib.ticker import ScalarFormatter
formatter = ScalarFormatter()
formatter.set_scientific(False)


#%%
fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T, log_scale=True, ax = ax, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")
ax.yaxis.set_major_formatter(formatter)


#%%
fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T.iloc[:, 0:3], ax = ax, log_scale=True, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")

ax.yaxis.set_major_formatter(formatter)

#%%
fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T.iloc[:, 3:6], ax = ax, log_scale=True, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")

ax.yaxis.set_major_formatter(formatter)

#%%
fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T.iloc[:, 6:8], ax = ax, log_scale=True, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")

ax.yaxis.set_major_formatter(formatter)
#%%
fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T.iloc[:, 6:11], ax = ax, log_scale=True, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")
ax.yaxis.set_major_formatter(formatter)
#%%

fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T.iloc[:, 11:13], ax = ax, log_scale=True, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")

ax.yaxis.set_major_formatter(formatter)
#%%

fig, ax = plt.subplots()
sns.boxenplot(DB_of_all_time.T.iloc[:, 13], ax = ax, log_scale=True, showfliers=False)
ax.tick_params(axis='x', labelrotation=90)
ax.set_ylabel("Days")

ax.yaxis.set_major_formatter(formatter)

#%%

# times_DB = pd.DataFrame(model_times, index=("Bestfit Time", "Uncert-", "Uncert+")).T / (
#     60 * 60 * 24
# )

# times_DB["Diffusivity_adjusted_time"] = times_DB["Bestfit Time"] / Diffusivity_Scale



# times_DB["Category"] = model_category
# times_DB["PT_corrected_times"] = PT_corrected_times

# DB_of_all_time = pd.DataFrame(PT_corrected_times)
# DB_of_all_time["Category"] = model_category


# times_DB.to_excel("Times_database_Diffusivities.xlsx")

# %%

# plt.figure(figsize=(12, 8))
# sns.boxenplot(
#     x="Category",
#     y="Bestfit Time",
#     data=times_DB,
# )

# %%
# Code for producing time uncertainty for diffusion models.


# categories_to_model = times_DB.Category.unique()
# # Plot each xenolith as a plot
# for category in categories_to_model:
#     fig, axs = plt.subplots(figsize=(12, 8))
#     profs = times_DB.loc[times_DB["Category"] == category].index.unique()
#     for idx, prof in enumerate(profs):
#         sns.kdeplot(
#             times_DB.loc[prof]["PT_corrected_times"], ax=ax, palette="crest"
#         )  # log_scale= True

#     plt.title(category)
#     plt.show()
# %%
# Plot all categories on different subplots
# fig, axs = plt.subplots(
#     nrows=len(categories_to_model), figsize=(8.2, 10.5), sharex=True, squeeze=True
# )

# for idx, category in enumerate(categories_to_model):
#     profs = times_DB.loc[times_DB["Category"] == category].index.unique()
#     for prof in profs:
#         sns.kdeplot(
#             times_DB.loc[prof]["PT_corrected_times"],
#             ax=axs[idx],
#             palette="colorblind",
#             log_scale=True,
#             common_norm=False,
#         )  # log_scale= True
#         axs[idx].set_title(category, pad=-15)

# plt.xlable = "Days for Diffusion Profile to Form"


# %%

# fig, axs = plt.subplots(
#     nrows=len(categories_to_model), figsize=(8.2, 10.5), sharex=True, squeeze=True
# )

# for idx, category in enumerate(categories_to_model):
#     profs = times_DB.loc[times_DB["Category"] == category].index.unique()
#     for prof in profs:
#         sns.kdeplot(
#             times_DB.loc[prof]["PT_corrected_times"],
#             ax=axs[idx],
#             palette="colorblind",
#             log_scale=True,
#             common_norm=False,
#         )  # log_scale= True
#         axs[idx].set_title(category, pad=-15)

# plt.xlable = "Days for Diffusion Profile to Form"

# %%

categories_to_model = times_DB.Category.unique()
fig, axs = plt.subplots(
    nrows=len(categories_to_model), figsize=(8.2, 10.5), sharex=True, squeeze=True
)

for idx, category in enumerate(categories_to_model):
    profs = times_DB.loc[times_DB["Category"] == category].index.unique()
    Dict = {}
    for prof in profs:
        Dict[prof] = times_DB.loc[times_DB.index == prof][
            "PT_corrected_times"
        ].to_numpy()[0]

    DB = pd.DataFrame(Dict)

    sns.histplot(
        data=DB,
        ax=axs[idx],
        palette="rocket",
        legend=False,
        stat="probability",
        bins=50,
        log_scale=True,
    )
    axs[idx].set_title(category, pad=-15, loc="left")

    axs[idx].tick_params(axis="y", which="both", width=2)
    axs[idx].tick_params(axis="y", which="major", length=7)
    axs[idx].tick_params(axis="y", which="minor", length=4)
plt.xlabel("Days for Diffusion Profile to Develop")

axs[-1].xaxis.set_major_formatter(ScalarFormatter())
axs[-1].tick_params(which="both", width=2)
axs[-1].tick_params(which="major", length=7)
axs[-1].tick_params(which="minor", length=4)
# plt.savefig("Melt_Vein_Olivine_Mg-Fe_Diffusion_Timescales.svg")


# %%

categories_to_model = times_DB.Category.unique()
fig, axs = plt.subplots(
    nrows=len(categories_to_model), figsize=(8.2, 10.5), sharex=True, squeeze=True
)

for idx, category in enumerate(categories_to_model):
    profs = times_DB.loc[times_DB["Category"] == category].index.unique()
    axs2 = axs[idx].twinx()

    Dict = {}
    Timescales = []
    for prof in profs:
        Timescales.append(
            times_DB.loc[times_DB.index == prof]["PT_corrected_times"].to_numpy()[0]
        )

        Dict[prof] = times_DB.loc[times_DB.index == prof][
            "PT_corrected_times"
        ].to_numpy()[0]

    Timescales = np.concatenate(Timescales)
    sns.kdeplot(
        data=Timescales,
        ax=axs2,
        palette="rocket",
        legend=False,
        # common_norm = "False",
        log_scale=True,
        linewidth=3.0,
        linestyle="dashed",
    )

    DB = pd.DataFrame(Dict)

    sns.histplot(
        data=DB,
        ax=axs[idx],
        palette="rocket",
        legend=False,
        stat="density",  # "probability",
        bins=50,
        log_scale=True,
    )
    axs[idx].set_title(category, pad=-15, loc="left")

    axs[idx].tick_params(axis="y", which="both", width=2)
    axs[idx].tick_params(axis="y", which="major", length=7)
    axs[idx].tick_params(axis="y", which="minor", length=4)
axs[-1].set_xlabel("Days for Diffusion Profile to Develop")

axs[-1].xaxis.set_major_formatter(ScalarFormatter())
axs[-1].tick_params(which="both", width=2)
axs[-1].tick_params(which="major", length=7)
axs[-1].tick_params(which="minor", length=4)

plt.savefig("Melt_Vein_Olivine_Mg-Fe_Diffusion_Timescales_sum.svg")
# %%

# %%

# %%
# Plot Histogram Plots
categories_to_model = times_DB.Category.unique()
fig, axs = plt.subplots(
    nrows=len(categories_to_model), figsize=(8.2, 10.5), sharex=True, squeeze=True
)

for idx, category in enumerate(categories_to_model):
    profs = times_DB.loc[times_DB["Category"] == category].index.unique()
    Dict = {}
    for prof in profs:
        Dict[prof] = times_DB.loc[times_DB.index == prof][
            "PT_corrected_times"
        ].to_numpy()[0]

    DB = pd.DataFrame(Dict)

    sns.histplot(
        data=DB,
        ax=axs[idx],
        palette="rocket",
        legend=False,
        stat="probability",
        bins=50,
        log_scale=True,
    )
    axs[idx].set_title(category, pad=-15, loc="left")

    axs[idx].tick_params(axis="y", which="both", width=2)
    axs[idx].tick_params(axis="y", which="major", length=7)
    axs[idx].tick_params(axis="y", which="minor", length=4)
plt.xlabel("Days for Diffusion Profile to Develop")

axs[-1].xaxis.set_major_formatter(ScalarFormatter())
axs[-1].tick_params(which="both", width=2)
axs[-1].tick_params(which="major", length=7)
axs[-1].tick_params(which="minor", length=4)


# %%%
# Make Table of timesscales with corrected for diffusivity uncertainties. 
Data_Dict = {}

categories_to_model = times_DB.Category.unique()


for idx, category in enumerate(categories_to_model):
    profs = times_DB.loc[times_DB["Category"] == category].index.unique()

    Category_times = []
    for prof in profs:
        Times = times_DB.loc[times_DB.index == prof]["PT_corrected_times"].to_numpy()[0]
        Category_times.append(Times)

        BestTime = np.percentile(Times, 50)
        MaxTime95CI = np.percentile(Times, 95)
        MinTime5CI = np.percentile(Times, 5)

        Timescales = {
            "BestTime": BestTime,
            "MinTime5Percentile": MinTime5CI,
            "MaxTime95Percentile": MaxTime95CI,
            "Category": category,
        }

        Data_Dict[prof] = Timescales

    Category_times = np.concatenate(Category_times)
    BestTime = np.percentile(Category_times, 50)
    MaxTime95CI = np.percentile(Category_times, 95)
    MinTime5CI = np.percentile(Category_times, 5)
    Timescales = {
        "BestTime": BestTime,
        "MinTime5Percentile": MinTime5CI,
        "MaxTime95Percentile": MaxTime95CI,
        "Category": category,
    }
    Data_Dict[category] = Timescales

Timescale_Summary_DF = pd.DataFrame(Data_Dict).T
Timescale_Summary_DF.to_excel("Diffusion_Timescales_Summary.xlsx")


# %%
