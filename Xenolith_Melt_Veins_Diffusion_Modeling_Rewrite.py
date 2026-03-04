#%%
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

#%%

excel_db_path = "Diffusion Round Robin/Round_Robin_Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"

ol_param_db = pd.read_excel(
    excel_db_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

#%%
excel_path = "Diffusion Round Robin/Diffusion Round Robin Data Reorg.xlsx"

Ol_Data = pd.read_excel(
    excel_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

Ol_profiles = Ol_Data.loc[
    # (Ol_Data.Profile == "yes")
    (Ol_Data.Marked_bad != "bad")
    & (Ol_Data.Ignore != "yes")
]


# %%
# Original

sample_names =Ol_profiles.Profile_Name.unique()

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


#%%


def select_data(DF, selection_dict):
    query = " and ".join(
        [f"{key} == {repr(value)}" for key, value in selection_dict.items()]
    )
    # query = ' and '.join(['{} == {}'.format(k, repr(v)) for k, v in m.items()])
    new_df = DF.query(query)
    return new_df


#%%


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
#%%

# To Do convert data to fraction Fo not normed to 100.

importlib.reload(Ol_Diff)
# import Fe_Mg_Diffusion_Convolution_Streamlined as Ol_Diff
def model_diffusion(
    profile_name, data_db=Ol_Data, parameter_db=ol_param_db, kriging_variogram_model="linear", kriging_variogram_parameters={"slope": 1e-4, "nugget": 2e-4}, kriging_nlags = None,  activation_EFo = 201000, Total_time_days=200
):
    # This function works for a single step, change how edge_x params are input to accept tuples to change for multiple steps
    parameters = parameter_db.loc[(ol_param_db.File_Name == profile_name)]

    # load parameters from database
    # theta = parameters.prof_angle.item()
    # phi1 = parameters.phi1.item()
    # Phi = parameters.Phi.item()
    # phi2 = parameters.phi2.item()
    alpha = parameters.alpha.item()
    beta = parameters.beta.item()
    gamma = parameters.gamma.item()

    dx_micron = parameters.dx.item()
    dt = parameters.dt.item()
    T_Celsius = parameters["T"].item()
    T = T_Celsius + 273.15  # T in kelvin
    P = parameters.P.item()
    # fO2_dQFM = parameters.FO2_dQFM.item()
    fO2 = parameters.fO2.item()

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
        variogram_model = kriging_variogram_model,
        variogram_parameters= kriging_variogram_parameters, 
        # nlags = kriging_nlags
    )

    low_x_idx = (np.abs(step_x - edge_x1)).argmin()
    high_x_idx = (np.abs(step_x - edge_x2)).argmin()

    model_x = step_x[low_x_idx : high_x_idx + 1]
    data_interp = Y_interp[low_x_idx : high_x_idx + 1]
    std_interp = Y_interp_std[low_x_idx : high_x_idx + 1]

    Total_time = Total_time_days * 24 * 60 * 60  # seconds
    timesteps = int(Total_time / dt)

    EFo = activation_EFo

    p = (T, P, fO2, inflection_x, low_x_idx, high_x_idx, edge_c, center_c, inflection_c)

    time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
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
        "sum_r2": sum_r2,
        "model_x": model_x,
        "Fo_diffusion_results": Fo_diffusion_results,
        "x": x,
        "y": y,
        "X_interp": X_interp,
        "Y_interp": Y_interp,
        "Y_interp_std": Y_interp_std,
        "T_Celsius": T_Celsius,
        "P": P,
        "fO2": fO2,
        "profile_name": profile_name,
        "Category": parameters.Category.item(),
        "edge_x_micron": [edge_x1, edge_x2],
        "edge_c": [edge_c, center_c],
        "inflection_x_µm": inflection_x,
        "inflection_c": inflection_c

    }


#%%
def plot_diffusion_results(Model_dict, ax=None, tick_mark_major_multiple = 10, tick_mark_minor_multiple = 2.5):
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
        + f"Best fit time: {round(time_days[0],1)}±{round(time_days[1:None].max(),1)} days"
        + f" Temperature: {Model_dict['T_Celsius']} ˚C",
        weight="bold",
    )

    Model_dict["time"] / (60 * 60 * 24)

    textstr = '...Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()......Additional information using ax.annotate()...'
    wrapped_text = textwrap.fill(textstr, width=150)
    # Place the text using annotate
    # xy=(0.5, 0) specifies the reference point at the bottom-center of the axes (axes fraction coords)
    # xytext=(0, -40) specifies an offset of -40 points vertically downwards from the reference
    ax.annotate(wrapped_text, xy=(0.5, 0), xytext=(0, -120),
                xycoords='axes fraction',
                textcoords='offset points',
                size=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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


#%%
# Model_dict = model_diffusion("AZ18 WHT06_ol44_prof 1")
# Model_dict = model_diffusion('GCB1A R31_MP_Ol5_prof_2')
# Model_dict = model_diffusion("GCB1A R31_MP_Ol5_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_bigol2_overgrowth_prof_1")
# Model_dict = model_diffusion("AZ18 WHT06_ol_overgrowth2_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_Ol_181_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_Ol_180_prof_1")
#

# Model_dict = model_diffusion('Xenocryst_ol6_prof_1')
# Model_dict = model_diffusion('Xenocryst_ol6_prof_2')
# Model_dict = model_diffusion("AZ18 WHT06_ol43_prof 1")
# Model_dict = model_diffusion('Xenocryst_ol23_prof_1')
# Model_dict = model_diffusion('AZ18 WHT06_ol48_prof 1')

# Model_dict = model_diffusion("AZ18 WHT01_Ol_181_prof_2") #something is going wrong here. Not sure what.
# Model_dict = model_diffusion("AZ18 WHT01_Ol_192_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_bigol1_overgrowth_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_bigol2_overgrowth_prof_1")

# Model_dict = model_diffusion("AZ18 WHT01_Ol_181_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_Ol_191_prof_1")
# Model_dict = model_diffusion("AZ18 WHT01_Ol_193_prof_1")
# #Model_dict = model_diffusion('GCB1A R31_MP_Ol5_prof_2')


#%%
ol_param_db = pd.read_excel(
    excel_db_path,
    # sheet_name="Olivine",
    # index_col="DataSet/Point",
    engine="openpyxl",
)

# sample_name = "KS20-527_2-transect"  #
# sample_name = "KS20-527_5-transect"
sample_name = "KS20-527-8_transect"

# sample_name = "SH63olv066"
# sample_name =  "SH63olv084"

# sample_name = "AZ18 WHT06_ol43_prof 1"

Model_dict = model_diffusion(sample_name, data_db=Ol_Data, parameter_db=ol_param_db, Total_time_days=365, kriging_variogram_model="linear", kriging_variogram_parameters={"slope": (5/100)**2/10, "nugget":  (.2/100)**2}) #kriging_variogram_parameters={"slope": 1e-4, "nugget": 5e-4}
# I think a slope of 5% within 10 µm  should be (5/100)**2/10
# and a Nugget of .2 Fo is (.2/100)**2} since units are likely in variance so squared. 
# Model_dict = model_diffusion(sample_name, Total_time_days=365, kriging_variogram_model="linear", kriging_variogram_parameters=None, kriging_nlags = 2)
#%%
element_2 = "NiO"  #
fig, ax = plt.subplots(figsize=(12, 9))
plot_diffusion_results(Model_dict, tick_mark_major_multiple = 20, tick_mark_minor_multiple = 10, ax=ax)
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

#%%
files_to_model = ol_param_db.File_Name.dropna().unique()
sample_models = {}

for idx, sample in enumerate(files_to_model):
    try:
        Model_dict = model_diffusion(sample, data_db=Ol_Data, parameter_db=ol_param_db, Total_time_days=365, kriging_variogram_model="linear", kriging_variogram_parameters={"slope": (5/100)**2/10, "nugget":  (.2/100)**2}) #kriging_variogram_parameters={"slope": 1e-4, "nugget": 5e-4}
        sample_models[sample] = Model_dict

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_diffusion_results(Model_dict, tick_mark_major_multiple = 20, tick_mark_minor_multiple = 10, ax=ax)
        # Calculate_Gradient(sample, ax = ax)
        plt.savefig(f"Xenolith_Ol_Diffusion_Models/{sample}_model.svg")
        plt.savefig(f"{sample}_model.png")
        print(f"{idx+1} of {len(files_to_model)} completed")
    except:
        print(f"An Exception Occcured and {sample} Could not be calculated")
    Model_dict["time"] / (60 * 60 * 24)

# %%sample_models
# %%


# #%%
# model_times_1280= model_times
# %%
def D_Fo_For_PT_Sampling(T, P, fO2_delta, EFo=201000):
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


    """

    fO2 = Ol_Diff.fo2buffer(1250, 1e8, fO2_delta, "QFM")

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


#%%
# """

# Original Code that samples start and end Temp (But fails to account for covariance of those)
# Default_Fo2 = Ol_Diff.fo2buffer(
#     1250+ 273.15, 5e8, 0.3, buff="QFM"
# )  # Inputs are(T, P_pa, delta, buff)

# Default_D = Ol_Diff.D_Fo_For_PT_Sampling(
#     1250 + 273.15, 5e8, Default_Fo2
# )  # Ol_Diff.D_Fo_For_PT_Sampling(1250 + 273.15, 5e8, 0.3)  # T, P, fO2_delta,

# # All_Deep_D = Ol_Diff.D_Fo_For_PT_Sampling(
# #     1278 + 273.15, 1.2e9, Default_Fo2
# # )

# # All_Shallow_D = Ol_Diff.D_Fo_For_PT_Sampling(
# #     1216 + 273.15, 1e7, Default_Fo2
# # )

# num_samples = 10000

# num_time_steps = 1000

# #T_start = np.random.normal(1278.0, 14.0, num_samples) + 273.15  # Kelvin Original estimate of start temp
# T_start = np.random.normal(1292.0, 20.0, num_samples) + 273.15 # Kelvin Bigger estiamte of Start
# # T_end = (
# #     np.random.normal(1189.0, 19.0, num_samples) + 273.15
# # )  # Kelvin # Just average from Putirka Eq 22
# # T_end = (
# #     np.random.normal(1195, 21, num_samples) + 273.15
# # )  # Kelvin #  Putirka eq. 22, averaged comp for whole rock and Melt inclusions calcualted at 0.5 and 0.1 GPa

# T_end = (
#     np.random.normal(1216, 40, num_samples) + 273.15
# ) # Kelvin #  Putirka eq. 22 and 21, averaged comp for whole rock and Melt inclusions


# T_path = np.linspace(start=T_start, stop=T_end, num=num_time_steps)


# P_start = np.random.normal(1.25, 0.21, num_samples) * (1e9)
# P_end = np.zeros(num_samples)

# P_path = np.linspace(P_start, P_end, num_time_steps)

# # fO2 modeling along a limited PT range
# # Rough relationship between PT for the melt inclusions calcucalted with Putirka et al. T = 3E-08P + 1212.9 +(rand norm(1˚C))
# P_fO2 = np.random.normal(0.8, 0.2, num_samples) * (1e9)  # Pa
# T_fO2 = 3e-08 * P_fO2 + 1212.9 + np.random.normal(0, 10, num_samples) + 273.15  # Kelvin

# delta = np.random.normal(0.85, 0.35, num_samples)  # delta log(FMQ)
# fO2 = Ol_Diff.fo2buffer(
#     T_fO2, P_fO2, delta, buff="QFM"
# )  # Inputs are(T, P_pa, delta, buff)

# Diffusivity_array = Ol_Diff.D_Fo_For_PT_Sampling(T=T_path, P=P_path, fO2=fO2)

# dtD_array = np.ones(num_time_steps) * 60 * Default_D
# #

# Diffusivity_Scale = Diffusivity_array.mean() / Default_D

# Diffusivity_std = (Diffusivity_array.mean() / Default_D) * (
#     Diffusivity_array.std() / Diffusivity_array.mean()
# )
# """

#%%

# Updated PT path Based on Well known parameters
# Magma Temp at last equilibrium with solid. 1330 +/- 15 ˚C
# Starting pressure 2.26+-.2? Gpa. Uncertainty is a bit less known here
# Slope is a uniform distribution between 25 and 60 ˚C
# Pressure at which xenoliths are picked up: 1.15 +- 0.04 Roughly based on P&F16 Geotherm and B&K temps from Li et al. 2008 GCbx (920˚C)

Default_Fo2 = Ol_Diff.fo2buffer(
    1250 + 273.15, 5e8, 0.3, buff="QFM"
)  # Inputs are(T, P_pa, delta, buff)

Default_D = Ol_Diff.D_Fo_For_PT_Sampling(1250 + 273.15, 5e8, Default_Fo2)


num_samples = 10000

num_time_steps = 1000


def point_slope_line(slope, P1, T1, P2_eval):
    """Takes a slope and coordiantes of a point in PT space, Given a pressure to evaluate returns a temp in C or K depending on starting T units"""
    return slope * (P2_eval - P1) + T1


T_slope = np.random.uniform(46, 90, num_samples) / 1e9
T_Magma_start = np.random.normal(1330, 15, num_samples) + 273.15
P_Magma_start = np.random.normal(2.26, 0.1, num_samples) * (1e9)

P_Xenolith_start = np.random.normal(1.15, 0.04, num_samples) * (1e9)

P_end = np.zeros(num_samples)
P_path = np.linspace(P_Xenolith_start, P_end, num_time_steps)

T_Xenolith_start = point_slope_line(
    slope=T_slope, P1=P_Magma_start, T1=T_Magma_start, P2_eval=P_Xenolith_start
)
T_Xenolith_end = point_slope_line(
    slope=T_slope, P1=P_Magma_start, T1=T_Magma_start, P2_eval=0
)

T_path = np.linspace(start=T_Xenolith_start, stop=T_Xenolith_end, num=num_time_steps)

T_path_magma_full = np.linspace(
    start=T_Magma_start, stop=T_Xenolith_end, num=num_time_steps
)
P_path_magma_full = np.linspace(P_Magma_start, P_end, num_time_steps)
#%%

# fO2 modeling along a limited PT range
# Rough relationship between PT for the melt inclusions calcucalted with Putirka et al. T = 3E-08P + 1212.9 +(rand norm(1˚C))
P_fO2 = np.random.normal(0.8, 0.2, num_samples) * (1e9)  # Pa
T_fO2 = point_slope_line(
    slope=T_slope, P1=P_Magma_start, T1=T_Magma_start, P2_eval=P_fO2
)

# delta = np.random.normal(0.85, 0.35, num_samples)  # delta log(FMQ) # this is the average of the Melt Inclusions and the Xenolith Spinel MP Veins.
delta = np.random.normal(0.88, 0.17, num_samples)

fO2 = Ol_Diff.fo2buffer(
    T_fO2, P_fO2, delta, buff="QFM"
)  # Inputs are(T, P_pa, delta, buff)

Diffusivity_array = Ol_Diff.D_Fo_For_PT_Sampling(T=T_path, P=P_path, fO2=fO2)

dtD_array = np.ones(num_time_steps) * 60 * Default_D
#


def geo_mean_overflow(iterable):
    return np.exp(np.log(iterable).mean())


Diffusivity_Scale = Diffusivity_array.mean() / Default_D

Diffusivity_std = (Diffusivity_array.mean() / Default_D) * (
    Diffusivity_array.std() / Diffusivity_array.mean()
)


# %%


"""
total time = iterations * dt 

solving for new dt. 
dt_i*dx_i = dt_n * dx_n 

"""
model_times = {}
model_category = []
PT_corrected_times = []

for sample in sample_models.keys():
    model_times[sample] = sample_models[sample]["time"]
    model_category.append(sample_models[sample]["Category"])
    # model_category.append(sample_models[sample]["Category"])
    PT_corrected_times.append(
        np.array(
            Default_D
            * sample_models[sample]["time"][0]
            / Diffusivity_array.mean(
                0
            )  # Mean(0) is mean of PT Path), Mean(1) is mean at each PT step of path
            / (60 * 60 * 24)
        )
    )


times_DB = pd.DataFrame(model_times, index=("Bestfit Time", "Uncert-", "Uncert+")).T / (
    60 * 60 * 24
)

times_DB["Diffusivity_adjusted_time"] = times_DB["Bestfit Time"] / Diffusivity_Scale

times_DB["Diffusivity_adjusted_time_+std"] = times_DB["Bestfit Time"] / (
    Diffusivity_Scale + Diffusivity_std
)
times_DB["Diffusivity_adjusted_time_-std"] = times_DB["Bestfit Time"] / (
    Diffusivity_Scale - Diffusivity_std
)

# times_DB["All_Deep_time"] = times_DB["Bestfit Time"] * Default_D / All_Deep_D

# times_DB["All_Shallow_time"] = times_DB["Bestfit Time"] * Default_D / All_Shallow_D

times_DB["Category"] = model_category
times_DB["PT_corrected_times"] = PT_corrected_times

DB_of_all_time = pd.DataFrame(PT_corrected_times)
DB_of_all_time["Category"] = model_category


times_DB.to_excel("Times_database_Diffusivities.xlsx")

#%%

# plt.figure(figsize=(12, 8))
# sns.boxenplot(
#     x="Category",
#     y="Bestfit Time",
#     data=times_DB,
# )

#%%
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


#%%

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
plt.close("all")
# Plotting PT Path
fig, ax = plt.subplots(figsize=(12, 8))

ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")


def gpa_to_km(gpa_in):
    return (gpa_in + 0.1588) / 0.0319


def km_to_gpa(km_in):
    return 0.0319 * km_in - 0.1588  # taken from Plank and Forsythe Supplement


secax = ax.secondary_yaxis("right", functions=(gpa_to_km, km_to_gpa))
secax.set_ylabel("km", weight="bold")

# plt.plot(T_path_magma_full - 273.15, P_path_magma_full / 1e9, color="r", alpha=0.003)
plt.plot(
    T_path_magma_full.mean(axis=1) - 273.15,
    P_path_magma_full.mean(axis=1) / 1e9,
    color="k",
    alpha=1,
    label="Average Magma PT path",
    linewidth=3.0,
)


plt.plot(
    T_path_magma_full.mean(axis=1) + T_path_magma_full.std(axis=1) - 273.15,
    P_path_magma_full.mean(axis=1) / 1e9,
    color="k",
    alpha=1,
    linestyle="dashed",
)

plt.plot(
    T_path_magma_full.mean(axis=1) - T_path_magma_full.std(axis=1) - 273.15,
    P_path_magma_full.mean(axis=1) / 1e9,
    color="k",
    alpha=1,
    linestyle="dashed",
)

sns.kdeplot(
    x=T_Xenolith_start - 273.15,
    y=P_Xenolith_start / 1e9,
    fill=True,
    levels=5,
)
sns.kdeplot(
    x=np.random.normal(925, 5, num_samples),
    y=P_Xenolith_start / 1e9,
    fill=True,
    levels=5,
    cmap="Greens",
)

# mamga_red = sns.color_palette("Reds", light = 0.4, as_cmap=True)
sns.kdeplot(
    x=T_Magma_start - 273.15, y=P_Magma_start / 1e9, fill=True, levels=5, cmap="Reds"
)


T_mi = np.array(
    [
        1218,
        1228,
        1148,
        1163,
        1168,
        1122,
        1198,
        1183,
        1197,
        1193,
        1086,
        1078,
        1142,
        1161,
        1141,
        1168,
    ]
)
T_mi_1sigma = np.array(
    [
        31.8,
        34.4,
        27.8,
        30.7,
        21.6,
        19.1,
        23.2,
        27.7,
        23.1,
        26.5,
        21.4,
        16.0,
        23.1,
        29.3,
        22.6,
        21.7,
    ]
)

P_mi = (
    np.array(
        [401, 452, 574, 432, 271, 160, 510, 453, 235, 420, 137, 304, 199, 276, 379, 408]
    )
    / 1000
)
P_mi_1sigma = (
    np.array(
        [
            209.50,
            192.17,
            224.93,
            118.29,
            72.25,
            39.65,
            156.40,
            200.94,
            117.36,
            188.98,
            42.92,
            130.98,
            55.38,
            58.49,
            69.25,
            62.90,
        ]
    )
    / 1000
)

ax.errorbar(
    x=T_mi,
    y=P_mi,
    xerr=T_mi_1sigma,
    yerr=P_mi_1sigma,
    marker="o",
    ls="none",
    label=" Melt Inclusions MIMiC",
)

ax.errorbar(
    x=T_mi[0:2],
    y=P_mi[0:2],
    xerr=T_mi_1sigma[0:2],
    yerr=P_mi_1sigma[0:2],
    marker="o",
    ls="none",
    label=" Whole Rock Putirka",
)

plt.plot([1330, 1280, 1245], [2.2, 1.2, 0.5])
Geotherm_T = np.array(
    [
        905,
        950,
        996,
        1041,
        1086,
        1132,
        1177,
        1222,
        1267,
        1310,
        1319,
        1323,
        1326,
        1326,
        1327,
        1328,
        1329,
        1330,
        1330,
        1331,
        1332,
        1333,
        1334,
        1334,
        1335,
        1336,
        1337,
        1338,
        1338,
        1339,
        1340,
        1341,
        1342,
        1342,
        1343,
        1344,
        1345,
        1346,
        1346,
        1347,
        1348,
        1349,
        1350,
        1350,
        1351,
        1352,
        1353,
        1354,
        1354,
        1355,
        1356,
    ]
)
Geotherm_P = np.array(
    [
        1.12,
        1.18,
        1.24,
        1.31,
        1.37,
        1.44,
        1.50,
        1.56,
        1.63,
        1.69,
        1.76,
        1.82,
        1.88,
        1.95,
        2.01,
        2.07,
        2.14,
        2.20,
        2.27,
        2.33,
        2.39,
        2.46,
        2.52,
        2.58,
        2.65,
        2.71,
        2.78,
        2.84,
        2.90,
        2.97,
        3.03,
        3.10,
        3.16,
        3.22,
        3.29,
        3.35,
        3.41,
        3.48,
        3.54,
        3.61,
        3.67,
        3.73,
        3.80,
        3.86,
        3.92,
        3.99,
        4.05,
        4.12,
        4.18,
        4.24,
        4.31,
    ]
)
ax.plot(Geotherm_T, Geotherm_P, label="Geotherm")

T_Solidous = np.array(
    [1100, 1227.336264, 1337.226374, 1447.116484, 1557.006593, 1666.896703, 1776.786813]
)
Depth_Km_Solidous = np.array(
    [0, 36.38178964, 67.77896389, 99.17613815, 130.5733124, 161.9704867, 193.3676609]
)


ax.plot(
    T_Solidous, km_to_gpa(Depth_Km_Solidous), color="g", label="Dry Peridotite Solidous"
)


# ax.errorbar(
#     x=[1184.349136, 1206.367069, 1232.707056],
#     y=[0.1, 0.5, 1.0],
#     xerr=[18.9929899, 19.05559075, 17.72716284],
#     label="Melt Inclusions Putirka",
#     linewidth=3,
#     marker="o",
# )


ax.set_xlabel("˚C", weight="bold")
ax.set_ylabel("GPa", weight="bold")
plt.legend()

ax.set_xlim(900, 1400)
ax.set_ylim(3, 0)

ax.xaxis.set_minor_locator(MultipleLocator(25))

# ax.yaxis.set_minor_locator(AutoMinorLocator())

ax.tick_params(which="both", width=2)
ax.tick_params(which="major", length=7)
ax.tick_params(which="minor", length=4)

secax.yaxis.set_minor_locator(MultipleLocator(10))
secax.tick_params(which="both", width=2)
secax.tick_params(which="major", length=7)
secax.tick_params(which="minor", length=4)
plt.savefig("Xenlith+Magma_ascent_path_Draft_version.svg")
# %%


# %%

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


#%%%

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
