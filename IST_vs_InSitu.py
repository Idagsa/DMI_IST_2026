import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib

# Configure matplotlib settings
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 26

# DTU color scheme
DTU_NAVY = '#030F4F'
DTU_RED = '#990000'
DTU_GREY = '#DADADA'
WHITE = '#FFFFFF'
BLACK = '#000000'


# ============================================================================
# SECTION 1: IST RETRIEVAL FUNCTIONS
# ============================================================================

def ist_retrievals(equation_type, Tb11n=0., Tb12n=0., thetan=0.,
                   nedt_Tb11=0., nedt_Tb12=0.):
    """
    Retrieve Ice Surface Temperature (IST) using specified equation type.
    
    Parameters
    ----------
    equation_type : int
        Equation selector (2, 8, or 9)
    Tb11n : float or array
        Brightness temperature at 11 μm (K)
    Tb12n : float or array
        Brightness temperature at 12 μm (K)
    thetan : float or array
        Satellite zenith angle (degrees)
    nedt_Tb11 : float
        Noise equivalent differential temperature for 11 μm (K)
    nedt_Tb12 : float
        Noise equivalent differential temperature for 12 μm (K)
    
    Returns
    -------
    ist : array
        Retrieved ice surface temperature (K)
    sigma_algo : float
        Algorithm uncertainty (standard deviation of residuals)
    sigma_nedt : array
        Noise-induced uncertainty (K)
    """
    # Convert angles to radians
    thetan_rad = np.deg2rad(thetan)
    k = 1. / np.cos(thetan_rad) - 1.
    
    if equation_type == 2:
        # Equation 13 in ATBD - All temperature intervals (dual-channel)
        a = [-0.6384, 2.4444, -1.4417, -0.0633]
        ist = (a[0] + a[1]*Tb11n + a[2]*Tb12n + 
               a[3]*(Tb11n - Tb12n)*k)
        
        var_nedt = ((a[1] + a[3]*k)**2 * nedt_Tb11**2 + 
                    (a[2] - a[3]*k)**2 * nedt_Tb12**2)
        sigma_nedt = np.sqrt(var_nedt)
        sigma_algo = 0.072  # Standard deviation of residuals
        

    elif equation_type == 8:
        # Equation 11 in ATBD (single-channel, no angle correction)
        a = [-1.2541, 1.0065]
        ist = a[0] + a[1]*Tb11n
        
        var_nedt = a[1]**2 * nedt_Tb11**2
        sigma_nedt = np.sqrt(var_nedt)
        sigma_algo = 0.294
    
    elif equation_type == 9:
        # Single-channel with angle correction
        a = [-1.291, 1.0059, 0.0016]
        ist = a[0] + a[1]*Tb11n + a[2]*Tb11n*k
        
        var_nedt = (a[1] + a[2]*k)**2 * nedt_Tb11**2
        sigma_nedt = np.sqrt(var_nedt)
        sigma_algo = 0.262
    
    else:
        raise ValueError(f"Unknown equation_type: {equation_type}. Use 2, 8, or 9.")
    
    return ist, sigma_algo, sigma_nedt



# ============================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(data_path):
    """
    Load satellite and in-situ data from CSV file.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV data file
    
    Returns
    -------
    dict
        Dictionary containing all data arrays
    """
    print(f"Loading data from:\n  {data_path}\n")
    df = pd.read_csv(data_path)
    
    # Select required columns
    cols_needed = [
        "lev1_channel4",           # 11 μm brightness temperature
        "lev1_channel5",           # 12 μm brightness temperature
        "lev1_satzenith_angle",    # Satellite zenith angle
        "LatitudeGPS(degN)",       # Latitude
        "LongitudeGPS(degW)",      # Longitude (note: degW, not degE)
        "SurfaceTemperature(C)"    # In-situ ice temperature
    ]
    
    df_work = df[cols_needed].copy()
    n_before = len(df_work)
    df_work.dropna(inplace=True)
    n_after = len(df_work)
    
    print(f"Rows loaded : {n_before}")
    print(f"Rows valid  : {n_after}  (dropped {n_before - n_after} rows with NaN)\n")
    
    # Extract data into dictionary
    data = {
        'Tb11': df_work["lev1_channel4"].values.astype(float),
        'Tb12': df_work["lev1_channel5"].values.astype(float),
        'theta': df_work["lev1_satzenith_angle"].values.astype(float),
        'lat': df_work["LatitudeGPS(degN)"].values.astype(float),
        'lon': -df_work["LongitudeGPS(degW)"].values.astype(float),
        'ice_temp': df_work["SurfaceTemperature(C)"].values.astype(float) + 273.15
    }
    
    # Print data ranges for verification
    print(f"Latitude range: [{data['lat'].min():.4f}, {data['lat'].max():.4f}]°N")
    print(f"Longitude range: [{data['lon'].min():.4f}, {data['lon'].max():.4f}]°W")
    print(f"Ice Temperature range: [{data['ice_temp'].min():.2f}, "
          f"{data['ice_temp'].max():.2f}] K")
    print(f"Mean Ice Temperature: {data['ice_temp'].mean():.2f} K\n")
    
    # DIAGNOSTIC CHECKS
    print("=" * 70)
    print("DIAGNOSTIC CHECKS")
    print("=" * 70)
    print(f"\nTotal observations: {len(data['ice_temp'])}")
    print(f"Unique ice_temp values: {len(np.unique(data['ice_temp']))}")
    print(f"Unique Tb11 values: {len(np.unique(data['Tb11']))}")
    print(f"Unique Tb12 values: {len(np.unique(data['Tb12']))}")
    
    print(f"\nIce temp range: {data['ice_temp'].min():.2f} - {data['ice_temp'].max():.2f} K")
    print(f"Temperature spread: {data['ice_temp'].max() - data['ice_temp'].min():.2f} K")
    
    # Check correlation between inputs and "reference"
    r_tb11_icetemp = stats.pearsonr(data['Tb11'], data['ice_temp'])[0]
    r_tb12_icetemp = stats.pearsonr(data['Tb12'], data['ice_temp'])[0]
    print(f"\nCorrelation Tb11 vs ice_temp: {r_tb11_icetemp:.6f}")
    print(f"Correlation Tb12 vs ice_temp: {r_tb12_icetemp:.6f}")
    
    # Look at first few rows
    print("\nFirst 5 observations:")
    for i in range(min(5, len(data['Tb11']))):
        print(f"  Tb11={data['Tb11'][i]:.2f}, "
              f"Tb12={data['Tb12'][i]:.2f}, "
              f"IceTemp={data['ice_temp'][i]:.2f}")
    print("=" * 70 + "\n")
    
    return data


# ============================================================================
# SECTION 3: STATISTICAL ANALYSIS
# ============================================================================

def calculate_statistics(predicted, observed):
    """
    Calculate validation statistics between predicted and observed values.
    
    Parameters
    ----------
    predicted : array
        Predicted IST values (K)
    observed : array
        Observed in-situ temperatures (K)
    
    Returns
    -------
    dict
        Dictionary containing bias, RMSE, R², slope, and intercept
    """
    bias = np.mean(predicted - observed)
    rmse = np.sqrt(np.mean((predicted - observed)**2))
    
    # Correlation and linear regression
    r_value = stats.pearsonr(predicted, observed)[0]
    r2 = r_value**2
    slope, intercept, _, p_value, std_err = stats.linregress(observed, predicted)
    
    return {
        'bias': bias,
        'rmse': rmse,
        'r2': r2,
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    }


def print_validation_statistics(stats_dict, equation_names, reference_name):
    """
    Print formatted validation statistics.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary mapping equation names to statistics
    equation_names : list
        List of equation names
    reference_name : str
        Name of reference measurement
    """
    print("\n" + "=" * 70)
    print(f"VALIDATION STATISTICS vs {reference_name}")
    print("=" * 70)
    
    for eq_name in equation_names:
        stats = stats_dict[eq_name]
        print(f"{eq_name}: Bias={stats['bias']:6.3f} K, "
              f"RMSE={stats['rmse']:6.3f} K, R²={stats['r2']:.3f}")
    print()


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_scatter_comparison(ist_data, observed, stats_data, eq_names, 
                           reference_name, filename):
    """
    Create scatter plots comparing IST retrievals to in-situ measurements.
    
    Parameters
    ----------
    ist_data : list
        List of IST arrays from different equations
    observed : array
        Observed in-situ temperatures
    stats_data : list
        List of statistics dictionaries
    eq_names : list
        List of equation names
    reference_name : str
        Name of reference measurement
    filename : str
        Output filename for the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Determine common plot limits
    all_temps = np.concatenate([*ist_data, observed])
    vmin, vmax = all_temps.min() - 2, all_temps.max() + 2
    
    for i, (ist, stats, eq_name) in enumerate(zip(ist_data, stats_data, eq_names)):
        ax = axes[i]
        
        # Scatter plot
        ax.scatter(observed, ist, alpha=0.5, s=30, color=DTU_NAVY, 
                  edgecolors='none', label='Data')
        
        # 1:1 reference line
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=2, 
               label='1:1 line', alpha=0.5)
        
        # Linear regression line
        x_reg = np.array([vmin, vmax])
        y_reg = stats['slope'] * x_reg + stats['intercept']
        ax.plot(x_reg, y_reg, color=DTU_RED, linewidth=2,
               label=f'Fit: y={stats["slope"]:.2f}x+{stats["intercept"]:.2f}')
        
        # Add statistics text box
        stats_text = (f'Bias: {stats["bias"]:.2f} K\n'
                     f'RMSE: {stats["rmse"]:.2f} K\n'
                     f'R²: {stats["r2"]:.3f}')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=16, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', 
                        alpha=0.8, edgecolor=BLACK))
        
        # Labels and formatting
        ax.set_xlabel(f'{reference_name} (K)', fontsize=18)
        ax.set_ylabel(f'IST {eq_name} (K)', fontsize=18)
        ax.set_title(eq_name, fontsize=20)
        ax.legend(fontsize=14, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal')
    
    plt.suptitle(f'IST vs {reference_name}', 
                fontsize=24, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()


def plot_residuals(ist_data, observed, eq_names, reference_name, filename):
    """
    Create residual plots for IST retrievals.
    
    Parameters
    ----------
    ist_data : list
        List of IST arrays from different equations
    observed : array
        Observed in-situ temperatures
    eq_names : list
        List of equation names
    reference_name : str
        Name of reference measurement
    filename : str
        Output filename for the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (ist, eq_name) in enumerate(zip(ist_data, eq_names)):
        ax = axes[i]
        residuals = ist - observed
        
        # Scatter plot of residuals
        ax.scatter(observed, residuals, alpha=0.5, s=30, 
                  color=DTU_NAVY, edgecolors='none')
        
        # Zero line (perfect prediction)
        ax.axhline(0, color='k', linestyle='--', linewidth=2, 
                  alpha=0.5, label='Zero line')
        
        # Mean residual line
        mean_residual = np.mean(residuals)
        ax.axhline(mean_residual, color=DTU_RED, linestyle='-', 
                  linewidth=2, label=f'Mean: {mean_residual:.2f} K')
        
        # Labels and formatting
        ax.set_xlabel(f'{reference_name} (K)', fontsize=18)
        ax.set_ylabel(f'Residual (IST - {reference_name}) (K)', fontsize=18)
        ax.set_title(eq_name, fontsize=20)
        ax.legend(fontsize=14)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Residuals vs {reference_name}', 
                fontsize=24, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()


# ============================================================================
# SECTION 5: MAIN ANALYSIS WORKFLOW
# ============================================================================

def main():
    """Main analysis workflow."""
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    DATA_PATH = r"C:\Users\user\OneDrive\Desktop\DMI_project\Data\filter_mu_atmo_surf_obs2345_promistK.csv"
    
    # Sensor noise levels
    NEDT_TB11 = 0.05  # K (typical value for modern sensors)
    NEDT_TB12 = 0.05  # K (typical value for modern sensors)
    
    # Equation configuration
    EQUATION_TYPES = [2, 8, 9]
    EQUATION_NAMES = ['Equation 2', 'Equation 8', 'Equation 9']
    
    # -------------------------------------------------------------------------
    # Step 1: Load and preprocess data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70 + "\n")
    
    data = load_and_preprocess_data(DATA_PATH)
    
    # -------------------------------------------------------------------------
    # Step 2: Calculate IST retrievals (using ALL data)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: CALCULATING IST RETRIEVALS")
    print("=" * 70 + "\n")
    
    ist_results = {}
    
    for eq_type, eq_name in zip(EQUATION_TYPES, EQUATION_NAMES):
        ist, sigma_algo, sigma_nedt = ist_retrievals(
            eq_type,
            data['Tb11'],
            data['Tb12'],
            data['theta'],
            NEDT_TB11,
            NEDT_TB12
        )
        ist_results[eq_name] = {
            'ist': ist,
            'sigma_algo': sigma_algo,
            'sigma_nedt': sigma_nedt
        }
        
        print(f"{eq_name}:")
        print(f"  Mean IST: {np.mean(ist):.2f} K")
        print(f"  Std IST:  {np.std(ist):.2f} K")
        print(f"  Algorithm uncertainty: {sigma_algo:.3f} K")
        print()
    
    print(f"In-situ Ice Temperature:")
    print(f"  Mean: {np.mean(data['ice_temp']):.2f} K")
    print(f"  Std:  {np.std(data['ice_temp']):.2f} K")
    
    # -------------------------------------------------------------------------
    # Step 3: Calculate validation statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: CALCULATING VALIDATION STATISTICS")
    print("=" * 70)
    
    stats_results = {}
    for eq_name in EQUATION_NAMES:
        stats_results[eq_name] = calculate_statistics(
            ist_results[eq_name]['ist'],
            data['ice_temp']
        )
    
    print_validation_statistics(stats_results, EQUATION_NAMES, "In-situ Temperature")
    
    # -------------------------------------------------------------------------
    # Step 4: Create visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    # Prepare data for plotting
    ist_list = [ist_results[eq_name]['ist'] for eq_name in EQUATION_NAMES]
    stats_list = [stats_results[eq_name] for eq_name in EQUATION_NAMES]
    
    # Scatter plots
    plot_scatter_comparison(
        ist_list,
        data['ice_temp'],
        stats_list,
        EQUATION_NAMES,
        "In-situ Temperature",
        'IST_vs_InSitu.png'
    )
    
    # Residual plots
    plot_residuals(
        ist_list,
        data['ice_temp'],
        EQUATION_NAMES,
        "In-situ Temperature",
        'residuals_vs_InSitu.png'
    )
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")
    
    return ist_results, stats_results, data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    ist_results, stats_results, data = main()