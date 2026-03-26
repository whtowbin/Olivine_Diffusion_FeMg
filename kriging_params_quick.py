"""
Quick kriging parameter estimation for Fe-Mg diffusion profiles.

Directly integrates with model_diffusion() to automatically calculate
sill, range, and nugget from your data and physical parameters.

Usage:
------
    from kriging_params_quick import estimate_kriging_params
    
    krig_params = estimate_kriging_params(
        concentration_Fo_array=your_Fo_data,  # 0-100 scale
        measurement_uncertainty_Fo=0.2,  # ±0.2 Fo#
        diffusivity_m2s=1e-15,  # From temperature/pressure model
        diffusion_time_days=200,
    )
    
    Model_dict = model_diffusion(
        sample_name,
        kriging_variogram_parameters=krig_params
    )
"""

import numpy as np
import warnings


def estimate_kriging_params(
    concentration_Fo_array,
    measurement_uncertainty_Fo=0.2,
    diffusivity_m2s=None,
    diffusion_time_days=None,
    x_position_array=None,
    return_summary=False,
):
    """
    Estimate kriging variogram parameters for diffusion profiles.
    
    Calculates sill, range, and nugget from data characteristics and physics.
    
    Parameters
    ----------
    concentration_Fo_array : array-like
        Measured forsterite mole fraction in Fo# units (0-100).
        Must have at least 3 points.
        
    measurement_uncertainty_Fo : float, default 0.2
        Measurement uncertainty as absolute Fo# units.
        E.g., 0.2 means ±0.2 Fo#, not ±0.2%.
        
    diffusivity_m2s : float, optional
        Diffusivity in m²/s. If provided with diffusion_time_days,
        range is estimated from diffusion physics (recommended).
        Example: 1e-15 for Fe-Mg diffusion at ~1200°C.
        
    diffusion_time_days : float, optional
        Total diffusion duration in days.
        Required with diffusivity_m2s for physics-based range.
        
    x_position_array : array-like, optional
        Distance array in micrometers.
        If provided without diffusivity, range estimated from gradient.
        
    return_summary : bool, default False
        If True, returns (parameters_dict, summary_string).
        If False, returns just parameters_dict.
        
    Returns
    -------
    parameters : dict
        Dictionary with keys 'sill', 'range', 'nugget' ready for
        use in model_diffusion(..., kriging_variogram_parameters=parameters)
        
    summary : str (optional)
        Human-readable summary of parameter values and quality checks.
        
    Examples
    --------
    >>> # Simplest: just from concentration data and uncertainty
    >>> params = estimate_kriging_params(Fo_array, measurement_uncertainty_Fo=0.2)
    
    >>> # With physics: better range estimates
    >>> params = estimate_kriging_params(
    ...     Fo_array,
    ...     measurement_uncertainty_Fo=0.2,
    ...     diffusivity_m2s=1e-15,
    ...     diffusion_time_days=365,
    ...     return_summary=True
    ... )
    
    >>> # With position data: empirical range estimate
    >>> params = estimate_kriging_params(
    ...     Fo_array,
    ...     x_position_array=x_data,
    ...     measurement_uncertainty_Fo=0.2,
    ...     return_summary=True
    ... )
    """
    
    # Validate input
    y = np.asarray(concentration_Fo_array, dtype=float)
    if len(y) < 3:
        raise ValueError("Need at least 3 concentration data points")
    
    if np.max(y) > 100 or np.min(y) < 0:
        warnings.warn(
            f"Concentration values outside 0-100 range: [{np.min(y)}, {np.max(y)}]. "
            "Assuming Fo# units; if not, convert first."
        )
    
    # Convert to fraction for variance calculations
    y_frac = y / 100.0
    
    # ====================================================================
    # 1. NUGGET: From measurement uncertainty
    # ====================================================================
    # Nugget is variance of measurement error
    # If measurement uncertainty is ±0.2 Fo# = ±0.002 fraction
    # Then nugget = (0.002)² = 4e-6
    
    nugget = (measurement_uncertainty_Fo / 100.0) ** 2
    
    # ====================================================================
    # 2. SILL: From concentration data variance
    # ====================================================================
    # Sill is the total variance of concentrations
    # This is the "structured variance" you want to model
    
    sill = np.var(y_frac)
    
    if sill <= nugget:
        warnings.warn(
            f"sill ({sill:.2e}) ≤ nugget ({nugget:.2e}). "
            "This means measurement noise dominates signal. "
            "Check measurement calibration."
        )
    
    # ====================================================================
    # 3. RANGE: From physics OR data gradient
    # ====================================================================
    range_um = None
    range_method = None
    
    # Method A: Physics-based (preferred)
    if diffusivity_m2s is not None and diffusion_time_days is not None:
        time_seconds = diffusion_time_days * 24 * 3600
        diffusion_length_m = np.sqrt(diffusivity_m2s * time_seconds)
        range_um = diffusion_length_m * 1e6  # Convert to micrometers
        range_method = f"Physics (√Dt at t={diffusion_time_days}d)"
    
    # Method B: From spatial gradient
    elif x_position_array is not None:
        x = np.asarray(x_position_array, dtype=float)
        if len(x) != len(y):
            raise ValueError("x_position_array must have same length as concentration_Fo_array")
        
        dx = np.diff(x)
        dy = np.diff(y_frac)
        
        # Avoid division by zero
        valid = dx != 0
        if not np.any(valid):
            raise ValueError("All x positions are identical")
        
        gradient = np.abs(dy[valid] / dx[valid])
        mean_grad = np.nanmean(gradient)
        
        if mean_grad > 0:
            # Range ≈ std / mean_gradient
            # This is the distance over which concentration changes by ~1 std
            range_um = np.std(y_frac) / mean_grad
            range_method = f"From gradient ({np.max(x)-np.min(x):.1f}µm profile)"
        else:
            range_um = (np.max(x) - np.min(x)) / 2
            range_method = "Default (half profile width)"
    
    # Method C: Default fallback
    if range_um is None:
        # Estimate as half the spread of values if we had positions
        # Otherwise use a generic default
        range_um = 20.0  # Generic default for olivine diffusion
        range_method = "Default (20 µm)"
    
    # ====================================================================
    # Build results
    # ====================================================================
    
    params = {
        'sill': sill,
        'range': range_um,
        'nugget': nugget,
    }
    
    if not return_summary:
        return params
    
    # ====================================================================
    # Generate summary string
    # ====================================================================
    
    summary = f"""
================================================================================
                    KRIGING PARAMETER ESTIMATION
================================================================================

Input Data:
  Concentration points: {len(y)}
  Range: {np.min(y):.1f} – {np.max(y):.1f} Fo#
  Variance: {sill:.6f} ({np.std(y):.2f} Fo# std dev)
  Measurement uncertainty: ±{measurement_uncertainty_Fo} Fo#

Estimated Variogram Parameters:
  • Sill:   {sill:.6e}  (total concentration variance)
  • Range:  {range_um:.2f} µm  (spatial correlation length)
              Method: {range_method}
  • Nugget: {nugget:.6e}  (measurement noise variance = ±{measurement_uncertainty_Fo} Fo#)

Quality Metrics:
  Signal-to-Noise Ratio: {sill / nugget:.1f}:1
    - SNR > 10: Excellent (trust measurements)
    - SNR 5-10: Good (typical case)
    - SNR < 5: High noise (may need spatial smoothing)
  
  Nugget as % of sill: {100*nugget/sill:.1f}%
    - < 5%: Measurement noise negligible (rare)
    - 5-30%: Normal case (good)
    - > 50%: Noise dominates signal (check calibration)

Use in Your Code:
  kriging_variogram_parameters = {{
      'sill': {sill:.6e},
      'range': {range_um:.2f},
      'nugget': {nugget:.6e}
  }}
  
  Model_dict = model_diffusion(
      sample_name,
      kriging_variogram_parameters=kriging_variogram_parameters,
  )

================================================================================
"""
    
    return params, summary


def quick_check(concentration_Fo, uncertainty_Fo=0.2):
    """
    Ultra-quick parameter estimate (just from concentration and uncertainty).
    
    Use when you only have concentration data and measurement uncertainty.
    Returns parameters ready to paste into model_diffusion().
    """
    params = estimate_kriging_params(concentration_Fo, uncertainty_Fo)
    return params


def apply_kriging_params(
    model_diffusion_func,
    sample_names,
    data_db,
    param_db,
    measurement_uncertainty_Fo=0.2,
    diffusivity_m2s=1e-15,  # Typical for Fe-Mg at ~1200°C
    **model_kwargs
):
    """
    Apply estimated kriging parameters across multiple samples.
    
    Convenience function to estimate parameters for each sample
    and run model_diffusion automatically.
    
    Parameters
    ----------
    model_diffusion_func : callable
        The model_diffusion function from your code
    sample_names : list of str
        Sample names to process
    data_db : DataFrame
        Your Ol_Data database
    param_db : DataFrame
        Your ol_param_db parameters database
    measurement_uncertainty_Fo : float
        Measurement uncertainty in Fo# units
    diffusivity_m2s : float
        Diffusivity for your modeling conditions
    **model_kwargs
        Additional arguments to pass to model_diffusion
        (e.g., Total_time_days=365)
    
    Yields
    ------
    sample_name : str
    model_dict : dict
        Output from model_diffusion
    krig_params : dict
        Estimated kriging parameters used
    """
    
    from Fe_Mg_Diffusion_Convolution_Streamlined import OrdinaryKriging
    
    for sample_name in sample_names:
        # Get data for this sample
        params_row = param_db.loc[param_db.Sample == sample_name]
        if len(params_row) == 0:
            print(f"Sample {sample_name} not found in parameter database")
            continue
        
        profile_name = params_row.File_Name.item()
        time_days = model_kwargs.get('Total_time_days', 200)
        
        # Extract concentration profile
        prof = data_db.loc[
            (data_db.Profile_Name == profile_name) 
            & (data_db.Marked_bad != "bad") 
            & (data_db.Ignore != "yes")
        ].sort_values("Distance µm")
        
        if len(prof) < 3:
            print(f"Not enough data for {sample_name}")
            continue
        
        y = prof["Fo#"].to_numpy()
        x = prof["Distance µm"].to_numpy()
        
        # Estimate kriging parameters
        krig_params = estimate_kriging_params(
            y,
            measurement_uncertainty_Fo=measurement_uncertainty_Fo,
            diffusivity_m2s=diffusivity_m2s,
            diffusion_time_days=time_days,
            x_position_array=x,
        )
        
        # Run model
        model_dict = model_diffusion_func(
            sample_name,
            data_db=data_db,
            parameter_db=param_db,
            kriging_variogram_parameters=krig_params,
            **model_kwargs
        )
        
        yield sample_name, model_dict, krig_params


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*80)
    print("KRIGING PARAMETER ESTIMATION - QUICK START")
    print("="*80 + "\n")
    
    # Example concentration profile (synthetic)
    Fo_example = np.array([88.5, 87.2, 80.3, 65.4, 50.2, 45.1])
    
    print("Example 1: Basic estimation")
    print("-" * 80)
    params, summary = estimate_kriging_params(
        Fo_example,
        measurement_uncertainty_Fo=0.2,
        return_summary=True
    )
    print(summary)
    
    print("\nExample 2: Physics-based (with diffusivity)")
    print("-" * 80)
    params2, summary2 = estimate_kriging_params(
        Fo_example,
        measurement_uncertainty_Fo=0.2,
        diffusivity_m2s=1e-15,
        diffusion_time_days=200,
        return_summary=True
    )
    print(summary2)
