#%%
import pandas as pd
from pathlib import Path

# Define the directory to search (current directory or specify a path)
# The fit CSVs are written next to this file by Round_Robin_Diffusion_Modeling Part 2.py.
data_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Initialize dictionaries
data_interp_dfs = {}
model_bestfits_dfs = {}

# Find and load _data_interp.csv files
for file in data_dir.glob("*_data_interp.csv"):
    key = file.stem.replace("_data_interp", "")
    data_interp_dfs[key] = pd.read_csv(file)

# Find and load _model_bestfits.csv files
for file in data_dir.glob("*_model_bestfits.csv"):
    key = file.stem.replace("_model_bestfits", "")
    model_bestfits_dfs[key] = pd.read_csv(file)

# Display loaded files
print("Data interpolation files loaded:", list(data_interp_dfs.keys()))
print("Model best fits files loaded:", list(model_bestfits_dfs.keys()))

# Access dataframes like:
# data_interp_dfs["SampleName"]
# model_bestfits_dfs["SampleName"]
# %%

import matplotlib.pyplot as plt

def plot_model_and_interp(key, data_interp_dfs, model_bestfits_dfs, ax=None):
    """
    Plot model fits and interpolated data on the same plot.
    
    Parameters:
    -----------
    key : str
        The sample name key to retrieve from dictionaries
    data_interp_dfs : dict
        Dictionary of interpolated data dataframes
    model_bestfits_dfs : dict
        Dictionary of model best fits dataframes
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
    
    Returns:
    --------
    ax : matplotlib axis
        The axis with the plot
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get the dataframes
    interp_df = data_interp_dfs[key]
    model_df = model_bestfits_dfs[key]
    
    # Plot interpolated data
    ax.plot(
        interp_df['X_interp_µm'],
        interp_df['Y_interp_Fo'] * 100,
        label='Interpolated Data',
        linewidth=2,
        color='green',
        linestyle='dashed'
    )
    
    # Plot interpolated data uncertainty as shaded region
    ax.fill_between(
        interp_df['X_interp_µm'],
        (interp_df['Y_interp_Fo'] - interp_df['Y_interp_Fo_std']) * 100,
        (interp_df['Y_interp_Fo'] + interp_df['Y_interp_Fo_std']) * 100,
        alpha=0.2,
        color='green',
        label='Interp. Uncertainty'
    )
    
    # Plot model best fit
    ax.plot(
        model_df['position_µm_interp'],
        model_df['best_fit'] * 100,
        label='Best Fit',
        linewidth=3,
        color='cyan'
    )
    
    # Plot +1 sigma
    ax.plot(
        model_df['position_µm_interp'],
        model_df['pos_1_sigma'] * 100,
        label='+1 sigma',
        linewidth=1.5,
        color='blue',
        alpha=0.7,
        linestyle='--'
    )
    
    # Plot -1 sigma
    ax.plot(
        model_df['position_µm_interp'],
        model_df['neg_1_sigma'] * 100,
        label='-1 sigma',
        linewidth=1.5,
        color='darkblue',
        alpha=0.7,
        linestyle='--'
    )
    
    # Labels and formatting
    ax.set_xlabel('Distance from Rim (µm)', fontsize=12, weight='bold')
    ax.set_ylabel('Fo#', fontsize=12, weight='bold')
    ax.set_title(f'Model Fit and Interpolated Data: {key}', fontsize=14, weight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return ax
# %%

for key in model_bestfits_dfs.keys():
    plot_model_and_interp(key, data_interp_dfs, model_bestfits_dfs)
    plt.show()