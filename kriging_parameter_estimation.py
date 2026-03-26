#%%
"""
Estimate kriging variogram parameters from diffusion profile data and uncertainties.
Relates physical properties (measurement noise, data uncertainty) to pykrige parameters.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def estimate_nugget_from_uncertainty(
    x, y, measurement_uncertainty=None, relative_uncertainty=None
):
    """
    Estimate nugget parameter from measurement uncertainty.
    
    The nugget represents measurement error variance at short distances.
    
    Parameters
    ----------
    x : ndarray
        Position array (µm)
    y : ndarray
        Concentration array (Fo# as fraction, 0-1)
    measurement_uncertainty : float, optional
        Absolute measurement uncertainty (same units as y)
        e.g., 0.2 for ±0.2 Fo#
    relative_uncertainty : float, optional
        Relative uncertainty as fraction (e.g., 0.02 for ±2%)
        
    Returns
    -------
    nugget : float
        Estimated nugget variance
        
    Examples
    --------
    # If you have ±0.2 Fo# measurement uncertainty
    nugget = estimate_nugget_from_uncertainty(x, y, measurement_uncertainty=0.2/100)
    
    # Or as relative uncertainty
    nugget = estimate_nugget_from_uncertainty(x, y, relative_uncertainty=0.02)
    """
    
    if measurement_uncertainty is None and relative_uncertainty is None:
        raise ValueError("Must specify either measurement_uncertainty or relative_uncertainty")
    
    if relative_uncertainty is not None:
        # Convert relative uncertainty to absolute
        measurement_uncertainty = np.mean(y) * relative_uncertainty
    
    # Nugget is the variance of measurement error
    # Assuming Gaussian errors, variance = (uncertainty/1.96)^2 or simply uncertainty^2
    # Using squared uncertainty as variance estimate
    nugget = measurement_uncertainty ** 2
    
    return nugget


def estimate_sill_from_data(y):
    """
    Estimate sill parameter from data variance.
    
    The sill is the total variance of the concentration profile.
    For a variogram, sill ≈ var(y) for data without strong trends.
    
    Parameters
    ----------
    y : ndarray
        Concentration array
        
    Returns
    -------
    sill : float
        Estimated sill (variance)
    """
    return np.var(y)


def estimate_range_from_gradient(x, y, characteristic_length=None):
    """
    Estimate range parameter from concentration profile gradient.
    
    For diffusion profiles, the range relates to the length scale over which
    concentration changes significantly. This can be estimated from:
    1. The distance over which concentration reaches 63% of its variation
    2. The characteristic diffusion length scale: L = sqrt(D*t)
    
    Parameters
    ----------
    x : ndarray
        Position array (µm)
    y : ndarray
        Concentration array
    characteristic_length : float, optional
        If known from physics (e.g., sqrt(D*t) from diffusion equation)
        Supply this instead of estimating from gradient
        
    Returns
    -------
    range : float
        Estimated range (in same units as x)
        
    """
    
    if characteristic_length is not None:
        return characteristic_length
    
    # Method 1: From concentration gradient
    # Find the distance over which concentration changes significantly
    dy = np.diff(y)
    dx = np.diff(x)
    gradient = np.abs(dy / dx)
    
    # Range ≈ std of concentrations / mean gradient
    # This gives characteristic length scale
    range_est = np.std(y) / np.mean(gradient)
    
    return range_est


def estimate_diffusion_length_scale(D, time_seconds=None, time_days=None):
    """
    Calculate diffusion length scale: L = sqrt(D * t)
    
    This gives the expected range for diffusion profiles.
    
    Parameters
    ----------
    D : float
        Diffusivity (m^2/s)
    time_seconds : float, optional
        Duration in seconds
    time_days : float, optional
        Duration in days (converted to seconds if provided)
        
    Returns
    -------
    length_scale : float
        Length scale (in same units as D^0.5)
        Returns value in micrometers if D is in m^2/s
    """
    
    if time_days is not None:
        time_seconds = time_days * 24 * 3600
    
    # sqrt(D * t) gives length in meters if D in m^2/s and t in seconds
    length_m = np.sqrt(D * time_seconds)
    length_um = length_m * 1e6  # Convert to micrometers
    
    return length_um


def estimate_variogram_parameters(
    x, 
    y,
    measurement_uncertainty=None,
    relative_uncertainty=None,
    D=None,
    time_days=None,
    uncertainty_array=None,
):
    """
    Estimate all kriging variogram parameters from profile data.
    
    Parameters
    ----------
    x : ndarray
        Position (µm)
    y : ndarray
        Concentration (Fo# as fraction)
    measurement_uncertainty : float, optional
        Absolute measurement uncertainty
    relative_uncertainty : float, optional
        Relative measurement uncertainty (fraction)
    D : float, optional
        Diffusivity (m^2/s) for diffusion length scale
    time_days : float, optional
        Duration of diffusion (days)
    uncertainty_array : ndarray, optional
        Per-point uncertainty array if available
        
    Returns
    -------
    parameters : dict
        Dictionary with 'sill', 'range', 'nugget' for kriging
        
    Example
    -------
    params = estimate_variogram_parameters(
        x, y,
        measurement_uncertainty=0.002,  # ±0.2 Fo#
        D=1e-15,  # m^2/s
        time_days=365
    )
    kriging_parameters = params  # Use directly in model_diffusion
    """
    
    # Handle per-point uncertainties
    if uncertainty_array is not None:
        # Use mean squared uncertainty as nugget estimate
        nugget = np.mean(uncertainty_array ** 2)
    else:
        nugget = estimate_nugget_from_uncertainty(
            x, y, 
            measurement_uncertainty=measurement_uncertainty,
            relative_uncertainty=relative_uncertainty
        )
    
    #  Calculate sill
    sill = estimate_sill_from_data(y)
    
    # Calculate range
    if D is not None and time_days is not None:
        # Use diffusion physics to estimate range
        range_est = estimate_diffusion_length_scale(D, time_days=time_days)
    else:
        # Estimate from gradient
        range_est = estimate_range_from_gradient(x, y)
    
    return {
        'sill': sill,
        'range': range_est,
        'nugget': nugget,
    }


def validate_variogram_parameters(sill, range_param, nugget):
    """
    Validate that variogram parameters are physically reasonable.
    
    Parameters
    ----------
    sill : float
        Sill variance
    range_param : float
        Range distance
    nugget : float
        Nugget variance
        
    Returns
    -------
    is_valid : bool
        True if parameters are reasonable
    messages : list
        List of warnings if parameters are questionable
    """
    
    messages = []
    
    if nugget < 0:
        messages.append("ERROR: nugget must be ≥ 0")
    
    if nugget > sill:
        messages.append("WARNING: nugget > sill is unusual (means measurement noise dominates)")
    
    if range_param <= 0:
        messages.append("ERROR: range must be > 0")
    
    if sill <= 0:
        messages.append("ERROR: sill must be > 0")
    
    return len([m for m in messages if m.startswith("ERROR")]) == 0, messages


def suggest_kriging_parameters(
    profile_dict,
    element="Fo#",
    relative_measurement_uncertainty=0.02,
):
    """
    Suggest kriging parameters for a concentration profile.
    
    Parameters
    ----------
    profile_dict : dict
        Dictionary with keys 'x' (position), 'y' (concentration),
        optionally 'dy' (point uncertainties), 'D' (diffusivity), 
        'time_days' (duration)
    element : str
        Element being analyzed (e.g., "Fo#")
    relative_measurement_uncertainty : float
        Relative uncertainty as fraction (default 2%)
        
    Returns
    -------
    kriging_params : dict
        Parameters ready for OrdinaryKriging
    summary : str
        Human-readable summary of parameter estimation
    """
    
    x = np.asarray(profile_dict['x'])
    y = np.asarray(profile_dict['y'])
    
    # Normalize concentration to be a fraction if percentage
    if np.max(y) > 1:
        y = y / 100
    
    uncertainty_array = profile_dict.get('dy', None)
    measurement_unc = np.mean(y) * relative_measurement_uncertainty
    
    D = profile_dict.get('D', None)
    time_days = profile_dict.get('time_days', None)
    
    params = estimate_variogram_parameters(
        x, y,
        measurement_uncertainty=measurement_unc,
        D=D,
        time_days=time_days,
        uncertainty_array=uncertainty_array,
    )
    
    is_valid, messages = validate_variogram_parameters(
        params['sill'], params['range'], params['nugget']
    )
    
    summary = f"""
Kriging Parameter Suggestions for {element}:
{'='*50}
Sill:   {params['sill']:.6f}  (total variance of profile)
Range:  {params['range']:.2f} µm  (spatial correlation length)
Nugget: {params['nugget']:.6f}  (measurement noise variance)

Validation:
{'\n'.join('  - ' + m for m in messages)}

Physical Interpretation:
- Range implies correlation extends ~{params['range']:.1f} µm
  (diffusion length scale or gradient length scale)
- Nugget = {np.sqrt(params['nugget'])*100:.2f} Fo# uncertainty
  (measurement error at short distances)
- Sill/Nugget ratio = {params['sill']/params['nugget']:.1f}
  (signal-to-noise ratio; >10 is good)
"""
    
    return params, summary


# Example usage function
def example_parameter_estimation():
    """
    Example showing how to estimate parameters from synthetic data.
    """
    
    # Synthetic diffusion profile
    x = np.linspace(0, 100, 50)  # micrometers
    y = 0.5 - 0.3 * sp.special.erf(x / 20)  # Erfc diffusion profile
    y = y / 100  # Convert to fraction
    
    # Add measurement noise
    measurement_uncertainty = 0.002  # ±0.2 Fo#
    y_noisy = y + np.random.normal(0, measurement_uncertainty, len(y))
    
    # Estimate parameters
    print("\n" + "="*60)
    print("KRIGING PARAMETER ESTIMATION EXAMPLE")
    print("="*60)
    
    profile_dict = {
        'x': x,
        'y': y_noisy,
        'D': 1e-17,  # m^2/s (example diffusivity)
        'time_days': 200,  # days
    }
    
    params, summary = suggest_kriging_parameters(profile_dict)
    print(summary)
    print(f"\nUse in model_diffusion:")
    print(f"  kriging_variogram_parameters={params}")
    
    return params


if __name__ == "__main__":
    import scipy as sp
    params = example_parameter_estimation()

# %%
