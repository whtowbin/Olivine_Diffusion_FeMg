import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
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
    import EBSD_Profile_Rotation as EBSD

    import seaborn as sns
    import itertools
    from cycler import cycler
    import ast

    import importlib
    from scipy.interpolate import splrep, splev

    # import janitor


    # from pykrige import OrdinaryKriging

    from matplotlib import rc

    rc("axes", linewidth=2)
    rc("font", weight="bold", stretch="condensed", size=14, family="Avenir")
    return (
        FormatStrFormatter,
        MultipleLocator,
        Ol_Diff,
        ast,
        np,
        pd,
        plt,
        splev,
        splrep,
    )


@app.cell
def _(Ol_Diff, np):
    # Promt 1a
    # QFM+0.4 at 1200°C and 420 bars. Calc Log log(fO2) would you calculate in bars?
    P_pa = 420 * 100000
    fo2_pa = Ol_Diff.fo2buffer(T=1200+273.15, P_pa=P_pa, delta = 0.4, buff = 'QFM')
    fo2_bar = fo2_pa/ 100000
    np.log10(fo2_bar)
    return P_pa, fo2_bar, fo2_pa


@app.cell
def _():
    10**-7.87/ 1e-7
    return


@app.cell
def _(Ol_Diff):

    Ol_Diff.fo2buffer()
    return


@app.cell
def _(Ol_Diff):
    # Promt 1b
    # Assume a Fo82 olivine at 1200°C, 420 bars, and log(fO2) = -7.87 in bars and a profile parallel to the c-axis [001] 
    Ol_Diff.D_Fo(1200 + 273.15,420*1e5,10**-7.87,90,90,0,0.82)

    return


@app.cell
def _(Ol_Diff):
    #promt 1c same but Fo 88
    Ol_Diff.D_Fo(1200 + 273.15,420*1e5,10**-7.87,90,90,0,0.88)
    return


@app.cell
def _(Ol_Diff):
    #promt 1d same but Fo 1050°C, 420 bars, and log(fO2) = -7.87 Fo82
    Ol_Diff.D_Fo(1050 + 273.15,420*1e5,10**-7.87,90,90,0,0.82)
    return


@app.cell
def _(Ol_Diff):
    # promt 1e
    # 1050°C, 7500 bars, and log(fO2) = -7.87,  Fo88
    Ol_Diff.D_Fo(1050 + 273.15,7500*1e5,10**-7.87,90,90,0,0.88)
    return


@app.cell
def _(Ol_Diff):
    # promt 1d
    # 1050°C, 7500 bars, and log(fO2) = -9,  Fo88
    Ol_Diff.D_Fo(1050 + 273.15,7500*1e5,10**-9,90,90,0,0.88)
    return


@app.cell
def _(Ol_Diff):
    # Promt 1g
    # Assume a Fo82 olivine at 1200°C, 420 bars, and log(fO2) = -7.87 in bars and a profile parallel to alpha = 70, beta = 152, gamma 108
    Ol_Diff.D_Fo(1200 + 273.15,420*1e5,10**-7.87,70,152,108,0.82)

    return


@app.cell
def _(Ol_Diff, P_pa, fo2_bar, fo2_pa, np):
    # Promt 1h
    # NNO + 1 at 1200°C and 420 bars. Calc Log log(fO2) would you calculate in bars?
    P_pa_2 = 420 * 100000
    fo2_pa_2 = Ol_Diff.fo2buffer(T=1200+273.15, P_pa=P_pa, delta = 1, buff = 'QFM')
    fo2_bar_2 = fo2_pa/ 100000
    np.log10(fo2_bar)
    return


@app.cell
def _(pd):
    #%%
    # These need to be updated to the latest paths
    excel_db_path = "Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"

    ol_param_db = pd.read_excel(
        excel_db_path,
        # sheet_name="Olivine",
        # index_col="DataSet/Point",
        engine="openpyxl",
    )



    excel_path = "Xenolith_MeltVein_EMPA_Master.xlsx"

    Ol_Data = pd.read_excel(
        excel_path,
        sheet_name="Olivine",
        index_col="DataSet/Point",
        engine="openpyxl",
    )

    Ol_profiles = Ol_Data.loc[
        (Ol_Data.Profile == "yes")
        & (Ol_Data.Marked_bad != "bad")
        & (Ol_Data.Ignore != "yes")
    ]
    return Ol_Data, ol_param_db


@app.cell
def _():
    # Load Data
    # This cell needs to load data. I need to figure out a new schema to load all data from spreadsheet. 
    return


@app.cell
def _(FormatStrFormatter, MultipleLocator, Ol_Data, np, plt, splev, splrep):
    # These need to be updated to the new schema.
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


    return (get_C_prof,)


@app.cell
def _(mo):
    mo.md(r"""
    # Ideas for laying out calcs.
    # I think it might be best to split each sample into a seperate notebook. That way I can keep things clearly defined and easy to follow.

    # I need to modify the diffusion call functions so that they are less hard coded to my spreadsheet layouts. I can still use a spreadsheet input but will make more flexible for other schema.
    """)
    return


@app.cell
def model_diffusion(Ol_Data, Ol_Diff, ast, g, get_C_prof, np, ol_param_db):
    # I will edit this to take manual inputs
    # I ne

    def model_diffusion(
        profile_name, data_db=Ol_Data, parameter_db=ol_param_db, Total_time_days=200
    ):
        # This function works for a single step, change how edge_x params are input to accept tuples to change for multiple steps
        parameters = parameter_db.loc[(ol_param_db.File_Name == profile_name)]

        # load parameters from database
        theta = parameters.prof_angle.item()
        phi1 = parameters.phi1.item()
        Phi = parameters.Phi.item()
        phi2 = parameters.phi2.item()

        dx_micron = parameters.dx.item()
        dt = parameters.dt.item()
        T_Celsius = parameters["T"].item()
        T = T_Celsius + 273.15  # T in kelvin
        P = parameters.P.item()
        fO2_dQFM = parameters.FO2_dQFM.item()

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

        alpha, beta, gamma = g.vector_direction(theta, phi1, Phi, phi2)

        x, y = get_C_prof(profile_name, data_db)
        edge_c, center_c = edge_c / 100, center_c / 100
        y = y / 100




        fO2 = Ol_Diff.fo2buffer(T, P, delta=fO2_dQFM, buff="FMQ")


        # generate step function
        length = abs(x.max() - x.min())
        num_x = int(round(length / dx_micron, 0))
        step_x, dx_micron = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)




        # interpolate data and uncertainty to step spacing.
        X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
            x,
            y,
            step_x,
            variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
        )

        low_x_idx = (np.abs(step_x - edge_x1)).argmin()
        high_x_idx = (np.abs(step_x - edge_x2)).argmin()

        model_x = step_x[low_x_idx : high_x_idx + 1]
        data_interp = Y_interp[low_x_idx : high_x_idx + 1]
        std_interp = Y_interp_std[low_x_idx : high_x_idx + 1]

        Total_time = Total_time_days * 24 * 60 * 60  # seconds
        timesteps = int(Total_time / dt)

        EFo = 201000

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
        }

    return


@app.cell
def _(Ol_Data, Ol_Diff, ast, get_C_prof, np, ol_param_db):

    def model_diffusion(
        profile_name, data_db=Ol_Data, parameter_db=ol_param_db, Total_time_days=200
    ):
        # This function works for a single step, change how edge_x params are input to accept tuples to change for multiple steps
        parameters = parameter_db.loc[(ol_param_db.File_Name == profile_name)]

        # load parameters from database
        # theta = parameters.prof_angle.item()
        # phi1 = parameters.phi1.item()
        # Phi = parameters.Phi.item()
        # phi2 = parameters.phi2.item()




        dx_micron = parameters.dx.item()
        dt = parameters.dt.item()
        T_Celsius = parameters["T"].item()
        T = T_Celsius + 273.15  # T in kelvin
        P = parameters.P.item()
        fO2_dQFM = parameters.FO2_dQFM.item()

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

        alpha, beta, gamma = [] #g.vector_direction(theta, phi1, Phi, phi2)

        x, y = get_C_prof(profile_name, data_db)
        edge_c, center_c = edge_c / 100, center_c / 100
        y = y / 100

        fO2 = Ol_Diff.fo2buffer(T, P, delta=fO2_dQFM, buff="FMQ")

        # generate step function
        length = abs(x.max() - x.min())
        num_x = int(round(length / dx_micron, 0))
        step_x, dx_micron = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)




        # interpolate data and uncertainty to step spacing.
        X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
            x,
            y,
            step_x,
            variogram_parameters={"slope": 1e-4, "nugget": 2e-4},
        )

        low_x_idx = (np.abs(step_x - edge_x1)).argmin()
        high_x_idx = (np.abs(step_x - edge_x2)).argmin()

        model_x = step_x[low_x_idx : high_x_idx + 1]
        data_interp = Y_interp[low_x_idx : high_x_idx + 1]
        std_interp = Y_interp_std[low_x_idx : high_x_idx + 1]

        Total_time = Total_time_days * 24 * 60 * 60  # seconds
        timesteps = int(Total_time / dt)

        EFo = 201000

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
        }

    return


@app.function
# This is the format for the diffusion call I need to rewrite the function that calls this. the P term is what gets optimized 
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


if __name__ == "__main__":
    app.run()
