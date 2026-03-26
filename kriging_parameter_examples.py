"""
Example: Relating Kriging Parameters to Your Diffusion Profile Data

Shows how to calculate sill, range, and nugget from:
1. Your measurement uncertainties
2. Your data statistics  
3. Expected diffusion length scales
"""

import numpy as np
import matplotlib.pyplot as plt
from kriging_parameter_estimation import (
    estimate_variogram_parameters,
    estimate_diffusion_length_scale,
    validate_variogram_parameters,
    suggest_kriging_parameters,
)


# ============================================================================
# APPROACH 1: From Physical Properties
# ============================================================================

def estimate_params_from_physics(
    data_uncertainty_Fo,      # e.g., 0.2 for ±0.2 Fo# uncertainty
    diffusivity_m2s,          # D in m^2/s from your model
    diffusion_time_days,      # Total time from your experiment/model
    concentration_array,      # Your measured Fo# values (in percent, 0-100)
):
    """
    Estimate kriging parameters from your physical data properties.
    
    Params
    ------
    data_uncertainty_Fo : float
        Individual point uncertainty in Fo# units (not percentage)
        e.g., 0.2 means ±0.2 Fo#
    diffusivity_m2s : float
        Diffusivity in m^2/s from Arrhenius equation
    diffusion_time_days : float
        Duration in days
    concentration_array : ndarray
        Measured concentrations as Fo# (0-100 scale)
    """
    
    # Convert concentration to fraction (required for kriging)
    y = concentration_array / 100
    
    # Convert data uncertainty to fraction
    uncertainty_fraction = data_uncertainty_Fo / 100
    
    # 1. NUGGET: From measurement uncertainty
    # Kriging expects variance, so square the uncertainty
    nugget = uncertainty_fraction ** 2
    
    # 2. SILL: From data variance
    # Total variance of your concentration profile
    sill = np.var(y)
    
    # 3. RANGE: From diffusion physics
    # The diffusion length scale is sqrt(D*t)
    D = diffusivity_m2s
    time_seconds = diffusion_time_days * 24 * 3600
    diffusion_length_m = np.sqrt(D * time_seconds)
    diffusion_length_um = diffusion_length_m * 1e6  # Convert to micrometers
    
    range_param = diffusion_length_um
    
    # Validation
    is_valid, messages = validate_variogram_parameters(sill, range_param, nugget)
    
    return {
        'sill': sill,
        'range': range_param,
        'nugget': nugget,
    }, {'valid': is_valid, 'messages': messages, 'diffusion_length_um': diffusion_length_um}


# ============================================================================
# APPROACH 2: From Empirical Data Statistics
# ============================================================================

def estimate_params_from_data(x, y, uncertainty_array=None):
    """
    Estimate kriging parameters directly from your profile data.
    
    Params
    ------
    x : ndarray
        Distance in micrometers
    y : ndarray
        Concentration as fraction (0-1)
    uncertainty_array : ndarray, optional
        Uncertainty for each point
    """
    
    # 1. NUGGET: From measurement uncertainties
    if uncertainty_array is not None:
        # Use the RMS of uncertainties
        nugget = np.mean(uncertainty_array ** 2)
    else:
        # Default: assume small measurement noise
        # Could estimate as noise_std = signal_std / SNR
        nugget = np.std(y) / 100  # Assume 100:1 SNR
    
    # 2. SILL: Total variance
    sill = np.var(y)
    
    # 3. RANGE: From concentration gradient
    # Approximate as distance over which concentration changes by 1 std
    dx = np.diff(x)
    dy = np.diff(y)
    grad = np.abs(dy / dx)
    
    # Range ≈ std(y) / mean(|gradient|)
    range_est = np.std(y) / np.nanmean(grad)
    
    return {
        'sill': sill,
        'range': range_est,
        'nugget': nugget,
    }


# ============================================================================
# PRACTICAL EXAMPLE: Using your actual session parameters
# ============================================================================

def apply_to_your_model():
    """
    Example showing exactly how to apply this to model_diffusion()
    """
    
    print("="*70)
    print("KRIGING PARAMETER SELECTION FOR YOUR MODEL")
    print("="*70)
    
    # Your typical experimental parameters
    T_C = 1200  # Temperature in Celsius
    T = T_C + 273.15
    duration_days = 200  # Duration from your model_diffusion call
    
    # Your data characteristics
    measured_Fo_uncertainty = 0.2  # ±0.2 Fo# measurement uncertainty
    measured_concentrations = np.array([90, 88, 85, 80, 75, 70])  # Example Fo# values
    
    # Diffusivity from your model (example Fe-Mg diffusion)
    # From olivine diffusion studies: D ~ 10^-15 m^2/s at 1200°C
    D_example = 1e-15  # m^2/s
    
    print(f"\nInputs:")
    print(f"  Temperature: {T_C}°C")
    print(f"  Duration: {duration_days} days")
    print(f"  Measurement uncertainty: ±{measured_Fo_uncertainty} Fo#")
    print(f"  Estimated diffusivity: {D_example:.2e} m²/s")
    
    # Method 1: Physics-based
    print("\n" + "-"*70)
    print("METHOD 1: Physics-based parameter estimation")
    print("-"*70)
    
    params_phys, info_phys = estimate_params_from_physics(
        data_uncertainty_Fo=measured_Fo_uncertainty,
        diffusivity_m2s=D_example,
        diffusion_time_days=duration_days,
        concentration_array=measured_concentrations,
    )
    
    print(f"\nDiffusion length scale = sqrt(D*t) = {info_phys['diffusion_length_um']:.1f} µm")
    print(f"\nEstimated parameters:")
    print(f"  sill:   {params_phys['sill']:.6f}")
    print(f"  range:  {params_phys['range']:.2f} µm")
    print(f"  nugget: {params_phys['nugget']:.6f}")
    print(f"\nValidation: {params_phys['sill']/params_phys['nugget']:.1f}:1 signal-to-noise ratio")
    
    # Method 2: Data-based
    print("\n" + "-"*70)
    print("METHOD 2: Empirical estimation from data")
    print("-"*70)
    
    x_example = np.linspace(0, 100, len(measured_concentrations))  # distance in µm
    y_example = measured_concentrations / 100  # convert to fraction
    uncertainty_per_point = np.full_like(y_example, measured_Fo_uncertainty/100)
    
    params_emp = estimate_params_from_data(
        x_example, 
        y_example,
        uncertainty_array=uncertainty_per_point
    )
    
    print(f"\nEstimated parameters:")
    print(f"  sill:   {params_emp['sill']:.6f}")
    print(f"  range:  {params_emp['range']:.2f} µm")
    print(f"  nugget: {params_emp['nugget']:.6f}")
    
    # How to use in model_diffusion
    print("\n" + "="*70)
    print("USE IN YOUR CODE:")
    print("="*70)
    print(f"""
Model_dict = model_diffusion(
    sample_name,
    Total_time_days={duration_days},
    kriging_variogram_model='linear',  # or 'exponential', 'spherical'
    kriging_variogram_parameters={{
        'sill': {params_phys['sill']:.6f},
        'range': {params_phys['range']:.2f},
        'nugget': {params_phys['nugget']:.6f}
    }},
)
""")
    
    return params_phys, params_emp


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def test_parameter_sensitivity():
    """
    Show how the parameters affect interpolation quality.
    """
    
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Synthetic profile
    x = np.linspace(0, 100, 20)
    y = 0.9 - 0.4 * np.tanh((x - 50) / 15)  # Sigmoid diffusion profile
    
    print("\nHow nugget affects interpolation:")
    print("  nugget → measurements noise level")
    print("  - Small nugget: tracks noise in data (overfitting)")
    print("  - Large nugget: smooth interpolation, ignores noise")
    
    print("\nHow range affects interpolation:")
    print("  range → spatial correlation length")
    print("  - Small range: only neighbors influence each point")
    print("  - Large range: distant points affect interpolation")
    
    print("\nHow sill affects interpolation:")
    print("  sill → total variance")
    print("  - Should match data variance")
    print("  - sill > nugget means structured variance (good)")
    
    # Example: ratio of nugget to sill
    print("\n" + "-"*70)
    print("Recommended nugget/sill ratios:")
    print("-"*70)
    print("  < 0.1  (nugget much smaller): High SNR, trust measurements")
    print("  0.1-0.3: Moderate noise, typical case")
    print("  > 0.5  : High noise, smooth heavily")


# ============================================================================

if __name__ == "__main__":
    try:
        params_1, params_2 = apply_to_your_model()
        test_parameter_sensitivity()
        
        print("\n" + "="*70)
        print("Remember: Start with physics-based estimates, then validate")
        print("by comparing interpolated profiles to your raw data.")
        print("="*70)
    except ImportError as e:
        print(f"Import error (expected if running outside the project): {e}")
        print("To use these functions, install scipy: pip install scikit-learn")
