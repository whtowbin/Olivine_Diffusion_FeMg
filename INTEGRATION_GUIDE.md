# Integration Guide: Adding Automatic Kriging Parameter Estimation

## Current State

Your current code (line 595-602 in Xenolith_Melt_Veins_Diffusion_Modeling_Rewrite.py):

```python
Model_dict = model_diffusion(
    sample_name,
    data_db=Ol_Data,
    parameter_db=ol_param_db,
    Total_time_days=365*2,
    kriging_variogram_model="gaussian",
    kriging_variogram_parameters={'sill': ((1/100)**2), 'range': 20, 'nugget': (.2/100)**2}, 
    uniform_uncert = 0.0005
)
```

**Issue:** The sill value `((1/100)**2) = 0.0001` is a generic estimate and doesn't reflect your actual data variance.

---

## Solution: Add One Import and One Function Call

### Step 1: Add this import at the top of your file

```python
from kriging_params_quick import estimate_kriging_params, quick_check
```

### Step 2: Replace your model_diffusion call with this:

**Option A: Automatic (Recommended)**

```python
# Get the actual data for this sample
x_data, y_data = get_C_prof(sample_name, Ol_Data)

# Estimate kriging parameters from your actual data
krig_params = estimate_kriging_params(
    concentration_Fo_array=y_data,
    measurement_uncertainty_Fo=0.2,  # ±0.2 Fo# (adjust if different)
    diffusivity_m2s=1e-15,            # Typical Fe-Mg diffusion at ~1200°C
    diffusion_time_days=730,          # Your Total_time_days=365*2
    x_position_array=x_data,
    return_summary=False
)

# Now use the estimated parameters
Model_dict = model_diffusion(
    sample_name,
    data_db=Ol_Data,
    parameter_db=ol_param_db,
    Total_time_days=365*2,
    kriging_variogram_model="gaussian",
    kriging_variogram_parameters=krig_params,  # ← Uses calculated values
    uniform_uncert=0.0005
)
```

**Option B: Quick and Simple (If you want to keep it minimal)**

```python
x_data, y_data = get_C_prof(sample_name, Ol_Data)

krig_params = quick_check(y_data, uncertainty_Fo=0.2)

Model_dict = model_diffusion(
    sample_name,
    data_db=Ol_Data,
    parameter_db=ol_param_db,
    Total_time_days=365*2,
    kriging_variogram_model="gaussian",
    kriging_variogram_parameters=krig_params,
    uniform_uncert=0.0005
)
```

### Step 3 (Optional): Add summary output

```python
# Option A version with summary
krig_params, krig_summary = estimate_kriging_params(
    concentration_Fo_array=y_data,
    measurement_uncertainty_Fo=0.2,
    diffusivity_m2s=1e-15,
    diffusion_time_days=730,
    x_position_array=x_data,
    return_summary=True
)

print(krig_summary)
print(f"\nUsing kriging parameters: {krig_params}")
```

---

## Example: Processing Multiple Samples

If you're looping through samples, here's how to do it:

```python
# Original loop:
for sample_name in sample_names:
    Model_dict = model_diffusion(sample_name, ...)
    # process results

# Improved version with automatic parameter estimation:
from kriging_params_quick import estimate_kriging_params

for sample_name in sample_names:
    # 1. Get the actual profile data
    x_data, y_data = get_C_prof(sample_name, Ol_Data)
    
    # 2. Estimate parameters from that specific data
    krig_params = estimate_kriging_params(
        concentration_Fo_array=y_data,
        measurement_uncertainty_Fo=0.2,
        diffusivity_m2s=1e-15,
        diffusion_time_days=730,
        x_position_array=x_data,
    )
    
    # 3. Run model with data-specific parameters
    Model_dict = model_diffusion(
        sample_name,
        data_db=Ol_Data,
        parameter_db=ol_param_db,
        Total_time_days=730,
        kriging_variogram_model="gaussian",
        kriging_variogram_parameters=krig_params,
        uniform_uncert=0.0005
    )
    
    # 4. Process results
    # ...
```

---

## What About Diffusivity?

The `diffusivity_m2s` parameter used in parameter estimation depends on your T and P.

### Get from Your Model:

You already have Arrhenius parameters in your code! When you call `model_diffusion()`, it calculates diffusivity internally. You can extract the same calculation:

```python
# From your model_diffusion function:
# D_Fo(T, P, fO2, alpha, beta, gamma, EFo)

# For quick estimation, typical values:
def get_diffusivity_estimate(T_celsius, P_GPa=None):
    """Rough estimates for Fe-Mg diffusion in olivine."""
    T_K = T_celsius + 273.15
    
    # Simplified Arrhenius (replace with your actual model if needed)
    if T_celsius > 1150:
        D = 1.5e-15 * np.exp(-(250000 / 8.314 / T_K))  # m²/s
    elif T_celsius > 900:
        D = 2e-15 * np.exp(-(240000 / 8.314 / T_K))
    else:
        D = 1e-15 * np.exp(-(230000 / 8.314 / T_K))
    
    return D

# Then in your loop:
for sample_name in sample_names:
    # Get T from your database
    params_row = ol_param_db.loc[ol_param_db.Sample == sample_name]
    T_C = params_row['T_C'].item()
    P_Pa = params_row['P_Pa'].item()
    
    # Get estimated diffusivity
    D = get_diffusivity_estimate(T_C)
    
    # Get data
    x_data, y_data = get_C_prof(sample_name, Ol_Data)
    
    # Estimate kriging parameters with correct D
    krig_params = estimate_kriging_params(
        y_data,
        measurement_uncertainty_Fo=0.2,
        diffusivity_m2s=D,
        diffusion_time_days=730,
        x_position_array=x_data,
    )
    
    # Run model
    Model_dict = model_diffusion(
        sample_name,
        kriging_variogram_parameters=krig_params,
        ...
    )
```

---

## Troubleshooting

### "Range seems too small/large"

If the estimated range seems unreasonable:

```python
# Check what was calculated:
params, summary = estimate_kriging_params(
    y_data,
    measurement_uncertainty_Fo=0.2,
    diffusivity_m2s=1e-15,
    diffusion_time_days=730,
    x_position_array=x_data,
    return_summary=True
)

print(summary)

# If diffusivity is wrong, override:
params_override = estimate_kriging_params(
    y_data,
    measurement_uncertainty_Fo=0.2,
    diffusivity_m2s=5e-16,  # Try different value
    diffusion_time_days=730,
    x_position_array=x_data,
)
```

### "Sill seems off"

The sill is calculated as `np.var(y_data/100)`. Make sure your `y_data` is in Fo# units (0-100):

```python
y_data = prof["Fo#"].to_numpy()  # This should be 0-100, not 0-1
# If already 0-1, don't divide in estimate_kriging_params
```

### "Want to use your old hardcoded values"

Just keep using them - nothing breaks:

```python
# Old way still works:
krig_params = {'sill': 0.01, 'range': 20, 'nugget': 4e-6}

Model_dict = model_diffusion(
    sample_name,
    kriging_variogram_parameters=krig_params,  # Old or new, doesn't matter
    ...
)
```

---

## Summary

| What | Before | After |
|------|--------|-------|
| Setting parameters | Manual, hardcoded | Data-driven, automatic |
| Sill value | Generic `0.0001` | Actual data variance |
| Range value | Generic `20 µm` | Physics-based or empirical |
| Nugget value | `4e-6` | Matches your uncertainty |
| Effort per sample | Same for all | Customized per sample |

The key insight: **Your parameters should depend on your actual data**, not be the same for every sample.

---

## Next Steps

1. Copy `kriging_params_quick.py` to your project directory
2. Add `from kriging_params_quick import estimate_kriging_params` to your imports
3. Replace one `model_diffusion()` call with the new version
4. Run it and see if the interpolated profiles look better
5. If satisfied, update all your `model_diffusion()` calls

For detailed parameter meanings and validation, see `KRIGING_PARAMETERS_GUIDE.md`.
