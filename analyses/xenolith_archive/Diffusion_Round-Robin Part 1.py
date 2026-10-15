import marimo

__generated_with = "0.20.4"
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
    return Ol_Diff, np


@app.cell
def _(Ol_Diff, np):
    # Promt 1a
    # QFM+0.4 at 1200°C and 420 bars. Calc Log log(fO2) would you calculate in bars?
    P_pa = 420 * 100000
    fo2_pa = Ol_Diff.fo2buffer(T=1200+273.15, P_pa=P_pa, delta = 0.4, buff = 'QFM')
    fo2_bar = fo2_pa/ 100000
    np.log10(fo2_bar)
    return fo2_bar, fo2_pa


@app.cell
def _(fo2_pa):
    fo2_pa
    return


@app.cell
def _():
    10**-7.87/ 1e-7
    return


@app.cell
def _():

    # Ol_Diff.fo2buffer()
    return


@app.cell
def _(Ol_Diff):
    # Promt 1b
    # Assume a Fo82 olivine at 1200°C, 420 bars, and log(fO2) = -7.87 in bars and a profile parallel to the c-axis [001] 
    print(f'{Ol_Diff.D_Fo(1200 + 273.15,420*1e5,10**-7.87,90,90,0,0.82):.3e} m^2/s')
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
    # promt 1f
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
def _(Ol_Diff, fo2_bar, fo2_pa, np):
    # Promt 1h
    # NNO + 1 at 1200°C and 420 bars. Calc Log log(fO2) would you calculate in bars?
    P_pa_2 = 420 * 100000
    fo2_pa_2 = Ol_Diff.fo2buffer(T=1200+273.15, P_pa=P_pa_2, delta = 1, buff = 'QFM')
    fo2_bar_2 = fo2_pa/ 100000
    np.log10(fo2_bar)
    return


@app.cell
def _():
    1e5
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
