import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

def compare_and_plot(prefix, num_vars_to_plot=3):
    aats_file = f"{prefix}_trajectories.csv"
    julia_file = f"{prefix}_julia_trajectories.csv"
    
    if not os.path.exists(aats_file):
        print(f"Skipping {prefix}: AATS CSV '{aats_file}' not found.")
        return

    # Load AATS Data
    df_aats = pd.read_csv(aats_file)
    time_aats = df_aats['time'].values
    
    # TRAP 1 FIX: Restrict the evaluation to ONLY the plotted variables
    cols_to_evaluate = [f'x{i}' for i in range(min(num_vars_to_plot, len(df_aats.columns) - 1))]
    aats_states = df_aats[cols_to_evaluate]
    
    has_julia = os.path.exists(julia_file)
    
    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    if has_julia:
        df_julia = pd.read_csv(julia_file)
        time_julia = df_julia['time'].values
        julia_states_raw = df_julia[cols_to_evaluate]
        
        # TRAP 2 FIX: Interpolate Julia onto the exact AATS time grid
        # This prevents floating-point time grid misalignments from creating fake errors.
        julia_interpolator = interp1d(time_julia, julia_states_raw.values, axis=0, 
                                      kind='linear', fill_value="extrapolate")
        julia_states_aligned = julia_interpolator(time_aats)
        
        # Calculate Error Metrics strictly on the interpolated, aligned data
        abs_error_matrix = np.abs(aats_states.values - julia_states_aligned)
        L_inf = np.max(abs_error_matrix)
        rmse = np.sqrt(np.mean(abs_error_matrix**2))
        
        print(f"--- Statistics for {prefix.upper()} ---")
        print(f" Evaluated Nodes   : {cols_to_evaluate}")
        print(f" L_inf (Max Error) : {L_inf:.4e}")
        print(f" RMSE (Avg Error)  : {rmse:.4e}\n")

        # Plotting the Overlay
        for i, col_name in enumerate(cols_to_evaluate):
            c = colors[i % len(colors)]
            
            plt.plot(time_julia, df_julia[col_name], color=c, linestyle='-', linewidth=4, 
                     alpha=0.5, label=f'Julia Node {i}')
            plt.plot(time_aats, df_aats[col_name], color=c, linestyle='--', linewidth=1.5, 
                     label=f'AATS Node {i}')

        textstr = f'$L_\\infty$ Error: {L_inf:.2e}\nRMSE: {rmse:.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        title_str = f"AATS vs Julia Overlays: {prefix.upper()}"

    else:
        print(f"--- Statistics for {prefix.upper()} ---")
        print(" Julia data missing (OOM Killed). Plotting AATS only.\n")
        
        for i, col_name in enumerate(cols_to_evaluate):
            c = colors[i % len(colors)]
            plt.plot(time_aats, df_aats[col_name], color=c, linestyle='-', linewidth=2, 
                     label=f'AATS Node {i}')

        textstr = 'Julia Data Missing\n(Out of Memory)'
        props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
        plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        title_str = f"AATS Trajectories: {prefix.upper()} (Julia DNF)"

    plt.title(title_str, fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("State Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    
    output_img = f"{prefix}_comparison_plot.png"
    plt.savefig(output_img, dpi=300)
    print(f"Saved {output_img}\n")
    plt.close()

benchmarks = ["dpde", "multirate", "statedep", "prop_pulse", "diffusion"]

for benchmark in benchmarks:
    # Set to 5 so you can see deeper into the pulse chain!
    compare_and_plot(benchmark, num_vars_to_plot=5)