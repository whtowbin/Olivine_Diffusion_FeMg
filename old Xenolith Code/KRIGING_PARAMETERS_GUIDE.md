# Kriging Parameter Estimation for Diffusion Profiles
## A Practical Guide to Setting sill, range, and nugget

---

## Quick Summary

The three kriging variogram parameters relate to your physical data properties:

| Parameter | Physical Meaning | How to Calculate |
|-----------|-----------------|-----------------|
| **nugget** | Measurement noise variance | `(measurement_uncertainty)²` |
| **sill** | Total variance of concentration | `numpy.var(concentration_array)` |
| **range** | Spatial correlation length | `sqrt(diffusivity × time)` or from data gradient |

---

## 1. NUGGET (Measurement Noise)

The nugget represents short-distance variance, primarily from measurement uncertainty.

### For Your Data:

If you have a measurement uncertainty of **±0.2 Fo#** (which you mentioned):

```python
# Measurement uncertainty
measurement_uncertainty_Fo = 0.2  # ±0.2 in Fo# units

# Convert to fraction (if needed)
measurement_uncertainty_fraction = measurement_uncertainty_Fo / 100  # 0.002

# Nugget is the variance
nugget = measurement_uncertainty_fraction ** 2  # = 4e-6
```

### Practical Range for Nugget:
- **Small uncertainty (±0.1 Fo#)**: nugget = (0.001)² = 1e-6
- **Moderate uncertainty (±0.2 Fo#)**: nugget = (0.002)² = 4e-6  
- **Large uncertainty (±0.5 Fo#)**: nugget = (0.005)² = 2.5e-5

### From Your Code:
Your current value: `nugget: (.2/100)**2 = 4e-6` ✓ **This is correct!**

---

## 2. SILL (Total Variance)

The sill is the total variance of your concentration profile—how much concentrations vary overall.

### Calculation:

```python
import numpy as np

# Your concentration data (in Fo#, 0-100 scale)
Fo_measurements = np.array([90, 85, 75, 60, 50])  # example

# Convert to fraction if needed
concentrations = Fo_measurements / 100  # [0.90, 0.85, 0.75, 0.60, 0.50]

# Sill = variance
sill = np.var(concentrations)  # ≈ 0.0225
```

### Typical Values:
- **Small compositional range** (e.g., 85-95 Fo): sill ≈ 0.001-0.005
- **Medium range** (e.g., 50-100 Fo): sill ≈ 0.01-0.03
- **Large range** (e.g., 10-100 Fo): sill ≈ 0.1-0.2

### From Your Code:
Your current value: `sill: ((1/100)**2) = 0.0001` 

This seems very small for a typical diffusion profile. Unless your profiles have very narrow compositional ranges, consider calculating it from your actual data variance.

---

## 3. RANGE (Spatial Correlation Length)

The range is the distance scale over which concentration gradient matters. 

### Method A: From Diffusion Physics (Recommended)

The diffusion length scale is **L = √(D·t)**

```python
import numpy as np

# Your parameters from model_diffusion
T_celsius = 1200
D_m2s = 1e-15  # Diffusivity in m²/s (from Arrhenius equation)
time_days = 200  # Total duration

# Convert to seconds
time_seconds = time_days * 24 * 3600

# Diffusion length scale in meters
L_m = np.sqrt(D_m2s * time_seconds)

# Convert to micrometers (your distance unit)
L_um = L_m * 1e6  # ≈ 17.7 µm for example above

range_parameter = L_um
```

**This is physically justified:** The range should be roughly equal to the length scale of the diffusion process.

### Method B: From Data Gradient

If you don't have diffusivity estimates:

```python
x = np.array([...])  # distance in µm
y = np.array([...])  # concentration as fraction

# Calculate gradient
dx = np.diff(x)
dy = np.diff(y)
gradient = np.abs(dy / dx)

# Range ≈ std(concentration) / mean(gradient)
range_est = np.std(y) / np.nanmean(gradient)
```

This gives the distance over which concentration changes by ~1 standard deviation.

### Typical Values:
- **Narrow, steep profile**: range = 5-10 µm
- **Moderate gradient**: range = 15-30 µm  
- **Shallow, broad profile**: range = 50-100 µm

### From Your Code:
Your current value: `range: 20 µm` ✓ **This is reasonable!**

---

## 4. Complete Example for Your Use Case

Let's say you're modeling Fe-Mg diffusion in olivine:

```python
# Your experimental/modeling parameters
T_C = 1200  # Temperature in Celsius
duration_days = 365  # From your model_diffusion call

# Your measurement characteristics
measurement_uncertainty_Fo = 0.2  # ±0.2 Fo#

# Your concentration profile (example)
Fo_values = np.array([90.0, 85.3, 72.5, 55.8, 40.2])  # Fo# measurements
x_distance = np.array([0, 10, 50, 100, 150])  # µm

# ============================================================
# CALCULATE KRIGING PARAMETERS
# ============================================================

# 1. NUGGET from measurement uncertainty
nugget = (measurement_uncertainty_Fo / 100) ** 2
# nugget = (0.2/100)² = 4e-6

# 2. SILL from data variance
y_fraction = Fo_values / 100  # Convert to fraction
sill = np.var(y_fraction)
# For this example: sill ≈ 0.0544

# 3. RANGE from diffusion length scale
# Fe-Mg diffusivity in olivine at 1200°C: ~1-2 × 10^-15 m²/s
D = 1.5e-15  # m²/s
time_s = duration_days * 24 * 3600  # seconds
range_um = np.sqrt(D * time_s) * 1e6
# range_um ≈ 20.8 µm

# ============================================================
# USE IN YOUR CODE
# ============================================================

kriging_parameters = {
    'sill': sill,
    'range': range_um,
    'nugget': nugget
}

Model_dict = model_diffusion(
    sample_name,
    Total_time_days=duration_days,
    kriging_variogram_model='linear',
    kriging_variogram_parameters=kriging_parameters,
)
```

---

## 5. Practical Guidance

### Step-by-Step:

1. **Collect your actual data uncertainties**
   - Measurement uncertainty per point
   - Estimated noise in concentration values
   
2. **Extract your concentration data**
   ```python
   x_data, y_data = get_C_prof(profile_name, data_db)
   y_fraction = y_data / 100  # Convert to fraction
   ```

3. **Calculate from physics**
   ```python
   nugget = (measurement_uncertainty / 100) ** 2
   sill = np.var(y_fraction)
   range_um = np.sqrt(diffusivity * time_s) * 1e6
   ```

4. **Validate the parameters**
   - nugget < sill (measurement noise < total variance) ✓
   - range should be comparable to profile width
   - nugget/sill ratio should be < 0.5 (ideally < 0.1)

5. **Test empirically**
   - Run model_diffusion with estimated parameters
   - Visually inspect: Do interpolated profiles look reasonable?
   - Do they undersmooth (track noise) or oversmooth (lost detail)?

### Quality Checks:

```python
# Check the signal-to-noise ratio
SNR = sill / nugget
print(f"SNR = {SNR:.1f}")
# SNR > 10: High confidence, trust measurements
# SNR 5-10: Moderate noise, reasonable estimate
# SNR < 5: High noise, may need smoothing

# Check if nugget dominates
nugget_fraction = nugget / sill
print(f"Nugget as fraction of sill: {nugget_fraction:.2%}")
# Should be < 30-50%
```

---

## 6. Connecting to Your Current Code

In your [model_diffusion function](Xenolith_Melt_Veins_Diffusion_Modeling_Rewrite.py#L276):

```python
def model_diffusion(
    profile_condition_name,
    ...
    kriging_variogram_parameters={'sill': 0.01, 'range': 20, 'nugget': 4e-6},
    ...
):
    # Inside the function, these are used here:
    X_interp, Y_interp, Y_interp_std = Ol_Diff.Krige_Interpolate(
        x,
        y,
        step_x,
        variogram_model=kriging_variogram_model,
        variogram_parameters=kriging_variogram_parameters,  # ← Your parameters
    )
```

### Current Values in Your Code:
Line 602: `kriging_variogram_parameters={'sill': 0.0001, 'range': 20, 'nugget': 4e-6}`

- **nugget**: 4e-6 = (0.2/100)² ✓ Good (matches ±0.2 Fo# uncertainty)
- **range**: 20 µm ✓ Reasonable (typical diffusion length at 1200°C for 200+ days)
- **sill**: 0.0001 = (0.01)² ← This seems small; should match your actual data variance

---

## 7. Recommended Modifications

To make this more data-driven and robust, consider adding a helper function at the top of your script:

```python
def estimate_kriging_params(x, y, measurement_uncert_Fo, D_m2s, time_days):
    """
    Estimate kriging variogram parameters from data and physics.
    
    Returns dict ready for use in model_diffusion()
    """
    y_frac = y / 100 if np.max(y) > 1 else y
    
    nugget = (measurement_uncert_Fo / 100) ** 2
    sill = np.var(y_frac)
    range_um = np.sqrt(D_m2s * time_days * 24 * 3600) * 1e6
    
    return {'sill': sill, 'range': range_um, 'nugget': nugget}

# Then use it:
krig_params = estimate_kriging_params(
    x_data, y_data,
    measurement_uncert_Fo=0.2,
    D_m2s=1e-15,
    time_days=200
)

Model_dict = model_diffusion(
    sample_name, 
    kriging_variogram_parameters=krig_params,
)
```

---

## 8. References

For more details on kriging variograms:
- Christakos, G. (1992). "Random Field Models in Earth Sciences"
- Journel & Huijbregts (1978). "Mining Geostatistics"  
- PyKrige documentation: https://geostat-framework.readthedocs.io/

For Fe-Mg diffusion in olivine:
- Chakraborty (1997) "Rates and mechanisms of Fe-Mg interdiffusion in olivine"
- Dohmen et al. (2007) "Fe-Mg interdiffusion in olivine"

---

## Summary Checklist

- [ ] Have measurement uncertainty estimate (absolute Fo# units)
- [ ] Calculated nugget = (uncertainty)²
- [ ] Extracted actual concentration data variance → sill
- [ ] Estimated diffusivity for your T and P conditions
- [ ] Calculated range from √(Dt)
- [ ] Checked that nugget < sill
- [ ] Checked that range is physically reasonable for your profile
- [ ] Updated kriging_variogram_parameters in model_diffusion() calls
- [ ] Visually validated interpolated profiles

