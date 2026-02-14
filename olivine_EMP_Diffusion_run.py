# %%
# from Users.henry.Python Files.Electrical Conductivity SIMS Data.NS_ConductivityOlivines import Sample_Interpolate
# import Fe_Mg_Diffusion_Convolution
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import scipy.interpolate as interp
from matplotlib.backends.backend_pdf import PdfPages


import Fe_Mg_Diffusion_Convolution_Streamlined as Ol_Diff
%matplotlib inline

plt.rcParams['pdf.fonttype'] = 42
#%%

Feb_excel_path = "Feb 2021_EMP_GCB_version_2_16_20.xlsx"

Ol_Profiles_Feb = pd.read_excel(
    Feb_excel_path,
    sheet_name="Sorted",
    # header=60,
    index_col="DataSet/Point",
    engine="openpyxl",
)
#%%
July_excel_path = "July_2020_EMP_olivine_profiles.xlsx"

Ol_Profiles_July = pd.read_excel(
    July_excel_path,
    sheet_name="EDS Profiles",
    header=1,
    index_col="DataSet/Point",
    engine="openpyxl",
)
# %%
Names_Feb = Ol_Profiles_Feb.Name.unique()
Names_July = Ol_Profiles_July.Name.unique()
# %%

# Function to get distance and element data from a Dataframe
def get_C_prof(prof_name, DF, Element="Fo#", X="Distance µm"):
    prof = DF.loc[(DF.Name == prof_name) & (DF.Bad != "bad")]
    distance_um = prof[X]
    concentration = prof[Element]
    return distance_um.to_numpy(), concentration.to_numpy()



# %%
"""
Fo# Diffusion Modeling
"""
# %%

# Get data for one profile. 
x, y = get_C_prof("AZ18 WHT01_bigol1_overgrowth_prof_1_Original", Ol_Profiles_Feb)


#  Set the total time and dt to run the model
dt = 600
Total_time = 40 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

dx_micron = 1
step_x = np.arange(0, x.max(), dx_micron)


# Interpolate data to model dx spacing
X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std)# + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std)# - 0.00001)


# %%
# Set up Diffusivity function
# Convert Fo2 from log units with 10 ^(fo2) * 10^5 *1.02

EFo = 201000.0  # J/mol
P = 200000000.0  # Pa  100000
R = 8.3145  # J/molK
T_Celsius = 1200 #1250
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin

alpha = 40.39356164
beta = 49.78792654
gamma = 86.79535782
# Alpha (to 100) :[[40.39356164]]
# Beta (to 010):[[49.78792654]]
# Gamma (to 001):[[86.79535782]]

fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, "FMQ")

Di = Ol_Diff.D_Fo(T, P, fO2, alpha, beta, gamma, XFo=0.90, EFo=201000)
# # Check for obeying the CFL Condition


CFL = (dt * Di) / ((dx_micron / 1e6) ** 2)
print("Stability Condition: " + str(CFL))
print("Keep Below 0.5")

#%%
# Setup initial step function
inflection_x = 35 # Step Position
edge_x1 = 15 # Start on left
edge_x2 = 65  # Stop on right

edge_c =0.9215 # Left Step Concentration 
center_c = 0.8965 # Right Step Concentration. 

# Match interpolated data with step function
# This indexing only works when dx is 1 µm. make more universal
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]



# p variable that can be used with MC3 modeling software
p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)

# Runs a diffusion moder 
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    X_interp,
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)
time_days = time / (60 * 60 * 24)

# %%

x_P2O5, y_P2O5 = get_C_prof("AZ18 WHT01_bigol1_overgrowth_prof_1_Original", Ol_Profiles_Feb, Element='P2O5')


fig, ax = plt.subplots(figsize=(8, 6))

ax1 = ax
twin_2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
twin_2.spines["right"].set_position(("axes", 1.02))

Multiplier = 1
#plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min[0]], linewidth=9,color = "tab:cyan", alpha = .8)
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[int(idx_min[0]/Multiplier)], linewidth=9,color = "tab:cyan", alpha = 0.8)
ax1.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth=3, linestyle = 'dashed', color = "tab:blue")
#ax1.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth=9, color = "tab:cyan", )
 


ax1.plot(x, y, marker="o", linestyle='dashed', markersize=12, color='tab:orange', alpha = 0.7)
#twin_2.plot(x_P2O5, y_P2O5, marker="o", linestyle='dashed', markersize=12, color='tab:green', alpha = 0.7)


# plt.plot(step_x, Y_interp, linestyle="dashed", color = 'k', alpha = 0.7)
# plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle="dashed", color = 'k', alpha = 0.7)
# plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle="dashed", color = 'k', alpha = 0.7)

ax1.set_xlabel("Distance from Rim (µm)",fontsize=14,)
ax1.set_ylabel("Forsterite #",fontsize=14,  )
ax1.tick_params(axis="y", labelcolor='tab:orange',)

twin_2.set_ylabel("P2O5",fontsize=14,   )
twin_2.tick_params(axis="y", labelcolor='tab:green')

#plt.title("AZ18 WHT01_bigol1_overgrowth_prof_1")
# ax1.annotate(
#     f"Time: {round(time_days[0]/Multiplier,1)} days", xy=(0.74, 0.9), xycoords="axes fraction"
# )
ax1.annotate(f"Temperature: {T_Celsius} ˚C", xy=(0.74, 0.95), xycoords="axes fraction")
ax1.annotate(
    f"Bestfit time: {round(time_days[0],1)} days", xy=(0.74, 0.9), xycoords="axes fraction"
)
ax1.annotate(
    f"StDev:{round(time_days[1],1)} / +{round(time_days[2],1)} days", xy=(0.74, 0.85), xycoords="axes fraction"
)
plt.savefig('AZ18 WHT01_bigol1_overgrowth_prof_1_1200C_goldschmidt6.pdf', transparent= True)
#plt.savefig('AZ18 WHT01_bigol1_overgrowth_prof_1_1200C.pdf', transparent= True)
# %%
# %%


# %%
"""
New Model!
"""


x, y = get_C_prof("AZ18 WHT01_bigol2_overgrowth_prof_1", Ol_Profiles_Feb)


dx_micron = 1
dt = 400
step_x = np.arange(0, x.max(), dx_micron)


X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
    x,
    y,
    step_x,
    variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
)

plt.plot(x, y, marker="o")
plt.plot(step_x, Y_interp)
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001)
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001)


# %%


# Di = D_FO_Func(0.8)
# # Check for obeying the CFL Condition
# CFL = (dt * Di) / (dx ** 2)
# print(CFL)


EFo = 201000.0  # J/mol
P = 200000000.0  # Pa
R = 8.3145  # J/molK
T_Celsius = 1250 #1200
T = T_Celsius + 273.15  # 1200 + 273.15  # T in kelvin
fO2 = Ol_Diff.fo2buffer(T_Celsius, P, 0.3, "FMQ")

inflection_x = 15
edge_x1 = 0
edge_x2 = 45

edge_c = 0.9150
center_c = 0.897

alpha = 79.4629894
beta = 13.10529937
gamma = 82.29618331

# Alpha (to 100) :[[79.4629894]]
# Beta (to 010):[[13.10529937]]
# Gamma (to 001):[[82.29618331]]

# This indexing only works when dx is 1 µm. make more universal
data_interp = Y_interp[edge_x1:edge_x2]
std_interp = Y_interp_std[edge_x1:edge_x2]

Total_time = 100 * 24 * 60 * 60  # seconds
timesteps = int(Total_time / dt)

p = (T, P, fO2, inflection_x, edge_x1, edge_x2, edge_c, center_c)
time, idx_min, sum_r2, Fo_diffusion_results = Ol_Diff.Diffusion_call(
    p,
    alpha,
    beta,
    gamma,
    EFo,
    timesteps,  # I should calculate the max timesteps based on the slowest diffusivity I expect.
    data_interp,
    std_interp,
    dx_micron,
    dt=dt,
    output_full=True,
)


# %%

# Ignore this!
Z = Ol_Diff.Best_fit_Chi2(
    Fo_diffusion_results, data_interp, std_interp, dt, sigma_min=1e-4
)

reduced_chi = Z[2] / Z[2].min()
time_range = np.where(reduced_chi.round(1) == 2)[0]
time_days = time / (60 * 60 * 24)

time_max_days = time_range.max() * dt / (60 * 60 * 24)
time_min_days = time_range.min() * dt / (60 * 60 * 24)
# %%
fig, ax = plt.subplots(figsize=(8, 6))


plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[idx_min[0]], linewidth=6)
plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[0], linewidth=3)
# plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.min()],linewidth = 3)
# plt.plot(X_interp[edge_x1:edge_x2], Fo_diffusion_results[time_range.max()], linewidth = 3)
# plt.plot(X_interp, Fo_diffusion_results[3842])

plt.plot(x, y, marker="o", linestyle=None)
plt.plot(step_x, Y_interp, linestyle="dashed")
plt.plot(step_x, Y_interp + 2 * Y_interp_std + 0.00001, linestyle="dashed")
plt.plot(step_x, Y_interp - 2 * Y_interp_std - 0.00001, linestyle="dashed")

plt.xlabel("Distance from Rim µm")
plt.ylabel("Fo#")

plt.title("AZ18 WHT01_bigol2_overgrowth_prof_1")

plt.annotate(f"Temperature: {T_Celsius} ˚C", xy=(0.74, 0.95), xycoords="axes fraction")
plt.annotate(
    f"Bestfit time: {round(time_days[0],1)} days", xy=(0.74, 0.9), xycoords="axes fraction"
)
plt.annotate(
    f"StDev:{round(time_days[1],1)} / +{round(time_days[2],1)} days", xy=(0.74, 0.85), xycoords="axes fraction"
)
plt.savefig('AZ18 WHT01_bigol2_overgrowth_prof_1_1250C.pdf')
# %%
