import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def readdata(filepath='Run1010.csv'):
    """
    Reads and parses the Mott polarimeter data from the specified CSV file.

    Args:
        filepath (str): The path to the input CSV file.

    Returns:
        tuple: A tuple containing:
            - energylossmax (float): Maximum energy loss.
            - energyloss_step (float): Step size for energy loss.
            - photonwavelength (float): Photon wavelength.
            - count_left (pd.DataFrame): DataFrame for the left detector counts.
            - count_right (pd.DataFrame): DataFrame for the right detector counts.
    """
    # 2. Read in metadata from the first row (G1, H1, I1)
    meta_df = pd.read_csv(filepath, header=None, nrows=1)
    photonwavelength = meta_df.iloc[0, 6]
    energylossmax = meta_df.iloc[0, 7]
    energyloss_step = meta_df.iloc[0, 8]
    
    # Read the full data, using the first row as the header
    df = pd.read_csv(filepath)
    
    # 3. Create count_left table
    # The number of rows for each detector is calculated from the metadata
    rows_per_detector = int(energylossmax / energyloss_step) + 1
    
    count_left = df.iloc[:rows_per_detector].copy()
    
    # 4. Create count_right table
    count_right = df.iloc[rows_per_detector:].copy().reset_index(drop=True)
    
    # 5. Return the parsed data
    return energylossmax, energyloss_step, photonwavelength, count_left, count_right

def calculate_and_plot_polarization(count_left, count_right, sherman_function, run_num, 
                                    fit_start_eV, fit_end_eV, energyloss_step, 
                                    txt_output_path, png_output_path, photonwavelength, 
                                    fast_measure_ergloss_single, fast_measure_ergloss_scan=None):
    """
    Calculates asymmetry and polarization, saves results, and plots the data.
    Also extracts asymmetry at a single energy point and for a range of energies.

    Args:
        fast_measure_ergloss_single (float): The single energy loss for fast sherman measurement comparison.
        fast_measure_ergloss_scan (np.ndarray, optional): An array of energies to scan for sherman function.

    Returns:
        tuple: A tuple containing:
            - polarization (float): The final calculated polarization.
            - delta_polarization (float): The uncertainty in the polarization.
            - asymmetry_0 (float): The extrapolated asymmetry at 0 energy loss.
            - asymmetry_fast_single (float): Asymmetry at the single fast measurement energy.
            - delta_asymmetry_fast_single (float): Error in asymmetry at the single fast measurement energy.
            - asymmetry_scan_map (dict): {energy: asymmetry} for the scanned energies.
            - delta_asymmetry_scan_map (dict): {energy: error} for the scanned energies.
    """
    # --- Error Propagation Setup ---
    # Convert percentage error to absolute error for each count
    count_cols = ['X1', 'X2', 'Y1', 'Y2']
    for df in [count_left, count_right]:
        for col in count_cols:
            # Handle potential NaN in 'Error' column by filling with 0
            df['Error'] = df['Error'].fillna(0)
            # User clarification: Error in file is total delta, so divide by 2 for +- delta.
            df[f'delta_{col}'] = df[col] * (df['Error'] / 100.0 / 2.0)

    # Get background counts and their errors
    background_left = count_left.loc[0]
    background_right = count_right.loc[0]
    
    removebackground_left = count_left.copy()
    removebackground_right = count_right.copy()
    
    # Subtract background and propagate errors
    for col in count_cols:
        # Subtract counts
        removebackground_left[col] = removebackground_left[col] - background_left[col]
        removebackground_right[col] = removebackground_right[col] - background_right[col]

        # Propagate errors for background subtraction (add errors in quadrature)
        removebackground_left[f'delta_{col}_prop'] = np.sqrt(removebackground_left[f'delta_{col}']**2 + background_left[f'delta_{col}']**2)
        removebackground_right[f'delta_{col}_prop'] = np.sqrt(removebackground_right[f'delta_{col}']**2 + background_right[f'delta_{col}']**2)
        
    # We perform calculations on data where energy loss is not zero
    calc_df_left = removebackground_left.iloc[1:].reset_index(drop=True)
    calc_df_right = removebackground_right.iloc[1:].reset_index(drop=True)
    
    # Create a results DataFrame to store intermediate calculations
    results = pd.DataFrame()
    results['Energy loss, eV'] = calc_df_left['Energy loss, eV']

    # 9. Calculate asymmetry and its error for X and Y directions
    with np.errstate(divide='ignore', invalid='ignore'):
        # --- X Asymmetry ---
        term_x = np.sqrt(
            (calc_df_left['X1'] * calc_df_right['X2']) / 
            (calc_df_left['X2'] * calc_df_right['X1'])
        )
        results['asymmetry_x'] = (term_x - 1) / (term_x + 1)
        
        # Propagate error for X
        rel_err_sq_x = (calc_df_left['delta_X1_prop']/calc_df_left['X1'])**2 + (calc_df_right['delta_X2_prop']/calc_df_right['X2'])**2 + \
                       (calc_df_left['delta_X2_prop']/calc_df_left['X2'])**2 + (calc_df_right['delta_X1_prop']/calc_df_right['X1'])**2
        delta_term_x = 0.5 * term_x * np.sqrt(rel_err_sq_x)
        results['delta_asymmetry_x'] = np.abs(2 / (term_x + 1)**2) * delta_term_x

        # --- Y Asymmetry ---
        term_y = np.sqrt(
            (calc_df_left['Y1'] * calc_df_right['Y2']) / 
            (calc_df_left['Y2'] * calc_df_right['Y1'])
        )
        results['asymmetry_y'] = (term_y - 1) / (term_y + 1)
        
        # Propagate error for Y
        rel_err_sq_y = (calc_df_left['delta_Y1_prop']/calc_df_left['Y1'])**2 + (calc_df_right['delta_Y2_prop']/calc_df_right['Y2'])**2 + \
                       (calc_df_left['delta_Y2_prop']/calc_df_left['Y2'])**2 + (calc_df_right['delta_Y1_prop']/calc_df_right['Y1'])**2
        delta_term_y = 0.5 * term_y * np.sqrt(rel_err_sq_y)
        results['delta_asymmetry_y'] = np.abs(2 / (term_y + 1)**2) * delta_term_y
    
    # 11. Calculate total asymmetry and propagate its error
    results['asymmetry'] = 100 * np.sqrt(results['asymmetry_x']**2 + results['asymmetry_y']**2)
    
    # Propagate error for total asymmetry
    ax, ay = results['asymmetry_x'], results['asymmetry_y']
    dax, day = results['delta_asymmetry_x'], results['delta_asymmetry_y']
    # Add a small epsilon to prevent division by zero if ax and ay are zero
    epsilon = 1e-18
    results['asymmetry_error'] = (100 / (np.sqrt(ax**2 + ay**2) + epsilon)) * np.sqrt((ax*dax)**2 + (ay*day)**2)
    
    results.replace([np.inf, -np.inf], np.nan, inplace=True)
    results.dropna(inplace=True)
    
    print("\n--- Calculated Asymmetry Results ---")
    print(results.to_string())
    
    # 12. Fit and find asymmetry at energy loss = 0 using a weighted fit
    fit_range = np.arange(fit_start_eV, fit_end_eV + energyloss_step, energyloss_step)
    fit_points = results[results['Energy loss, eV'].isin(fit_range)]
    x_fit = fit_points['Energy loss, eV']
    y_fit = fit_points['asymmetry']
    y_err_fit = fit_points['asymmetry_error']

    # Ensure y_err_fit has no zeros or NaNs before inverting
    y_err_fit = y_err_fit.replace(0, 1e-9).fillna(1e-9)
    
    # Perform weighted linear fit to get parameters and their covariance matrix
    p, V = np.polyfit(x_fit, y_fit, 1, w=1/y_err_fit**2, cov=True)
    slope, intercept = p
    delta_slope, delta_intercept = np.sqrt(np.diag(V))
    
    asymmetry_0 = intercept
    delta_asymmetry_0 = delta_intercept

    # 13. Calculate the polarization and its error
    polarization = asymmetry_0 / sherman_function
    delta_polarization = delta_asymmetry_0 / sherman_function
    
    # Save the results to a txt file in the specified folder
    asymmetry_filename = os.path.join(txt_output_path, f'asymmetry_Run{run_num}.txt')
    with open(asymmetry_filename, 'w') as f:
        f.write(f"Photon Wavelength: {photonwavelength} nm\n")
        f.write(f"Polarization: {polarization:.4f} +/- {delta_polarization:.4f} %\n")
        f.write(f"Asymmetry at 0 eV: {asymmetry_0:.4f} +/- {delta_asymmetry_0:.4f} %\n")
        f.write(f"Linear Fit Function: y = ({slope:.4f} +/- {delta_slope:.4f})x + ({intercept:.4f} +/- {delta_intercept:.4f})\n")
        f.write("="*40 + "\n")
        f.write(results[['Energy loss, eV', 'asymmetry', 'asymmetry_error']].to_string())
    print(f"Asymmetry results saved to '{asymmetry_filename}'")
    
    # Extract asymmetry at the specified single fast measurement energy loss
    asymmetry_fast_single = results.loc[results['Energy loss, eV'] == fast_measure_ergloss_single, 'asymmetry'].iloc[0]
    delta_asymmetry_fast_single = results.loc[results['Energy loss, eV'] == fast_measure_ergloss_single, 'asymmetry_error'].iloc[0]

    # Extract asymmetries for the scanned energy range
    asymmetry_scan_map = {}
    delta_asymmetry_scan_map = {}
    if fast_measure_ergloss_scan is not None:
        for erg in fast_measure_ergloss_scan:
            if erg in results['Energy loss, eV'].values:
                asymmetry_scan_map[erg] = results.loc[results['Energy loss, eV'] == erg, 'asymmetry'].iloc[0]
                delta_asymmetry_scan_map[erg] = results.loc[results['Energy loss, eV'] == erg, 'asymmetry_error'].iloc[0]
            else:
                # Handle cases where the exact energy loss value might be missing
                asymmetry_scan_map[erg] = np.nan
                delta_asymmetry_scan_map[erg] = np.nan

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot asymmetry with error bars
    ax.errorbar(results['Energy loss, eV'], results['asymmetry'], yerr=results['asymmetry_error'], 
                fmt='o', label='Calculated Asymmetry', color='royalblue', alpha=0.7, capsize=3)
    
    ax.plot(x_fit, y_fit, 'o', label=f'Points for Fit ({fit_start_eV}-{fit_end_eV} eV)', 
            color='red', markersize=10, zorder=5)
    
    x_line = np.array([0, results['Energy loss, eV'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color='black', label=f'Linear Fit (y={slope:.4f}x + {intercept:.2f})')
    
    ax.plot(0, asymmetry_0, 'X', color='darkorange', markersize=12, 
            label=f'Asymmetry at 0 eV = {asymmetry_0:.2f} \u00B1 {delta_asymmetry_0:.2f}%')
    
    # Add an empty plot with a label for the polarization
    ax.plot([], [], ' ', label=f'Polarization = {polarization:.2f} \u00B1 {delta_polarization:.2f} %')
    
    ax.set_title(f'Mott Asymmetry vs. Energy Loss (Run {run_num}, {photonwavelength} nm)', fontsize=16)
    ax.set_xlabel('Energy Loss (eV)', fontsize=12)
    ax.set_ylabel('Asymmetry (%)', fontsize=12)
    
    # Set legend box with white background and black edge
    ax.legend(facecolor='white', edgecolor='black')
    ax.grid(True)
    ax.set_xlim(left=-5)
    
    plt.tight_layout()
    # Save the plot to the specified folder
    plot_filename = os.path.join(png_output_path, f'asymmetry_fit_plot_Run{run_num}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"\nPlot has been saved as '{plot_filename}'")

    return (polarization, delta_polarization, asymmetry_0, 
            asymmetry_fast_single, delta_asymmetry_fast_single, 
            asymmetry_scan_map, delta_asymmetry_scan_map)

def fast_sherman(polarization, asymmetry_fast_measure):
    """
    Calculates the Sherman function at a specific energy loss.

    Args:
        polarization (float): The calculated polarization of the beam.
        asymmetry_fast_measure (float): The measured asymmetry at the specified energy loss.

    Returns:
        float: The calculated Sherman function at the specified energy loss.
    """
    sherman_fast = asymmetry_fast_measure / polarization
    return sherman_fast

def fast_sherman_output(sherman_results, run_nums, txt_output_path, png_output_path, fast_measure_ergloss, start_average):
    """
    Summarizes results, calculates an average Sherman value with error, and exports outputs.

    Args:
        sherman_results (list): A list of dictionaries containing the results for each run.
        run_nums (np.ndarray): An array of the run numbers that were processed.
        txt_output_path (str): The path to save the summary text file.
        png_output_path (str): The path to save the summary plot.
        fast_measure_ergloss (float): The energy loss used for the fast Sherman measurement.
        start_average (int): The run number from which to start averaging the Sherman function.

    Returns:
        tuple:
            - average_fast_sherman (float): The average fast Sherman function value.
            - delta_average_fast_sherman (float): The standard error of the mean for the average sherman.
    """
    print("\n\n" + "="*25 + " Summary " + "="*25)
    summary_df = pd.DataFrame(sherman_results)
    print(summary_df.to_string(index=False))
    
    # Save the summary DataFrame to a text file
    num_start = run_nums.min()
    num_end = run_nums.max()
    # Include fast_measure_ergloss in the filename
    summary_filename = os.path.join(txt_output_path, f'fast_sherman_{fast_measure_ergloss}eV_{num_start}_{num_end + 1}.txt')
    with open(summary_filename, 'w') as f:
        f.write(summary_df.to_string(index=False))
    print(f"\nSummary results saved to '{summary_filename}'")
    
    sherman_col_name = f'Sherman @ {fast_measure_ergloss}eV'

    # Calculate the average and standard error of the mean (SEM) for the fast sherman function
    averaging_df = summary_df[summary_df['Run Number'] >= start_average]
    average_fast_sherman = averaging_df[sherman_col_name].mean()
    std_dev_fast_sherman = averaging_df[sherman_col_name].std()
    delta_average_fast_sherman = std_dev_fast_sherman / np.sqrt(len(averaging_df)) # SEM
    
    print(f"\nAverage Sherman @ {fast_measure_ergloss}eV (from run {start_average} onwards): {average_fast_sherman:.4f} +/- {delta_average_fast_sherman:.4f}")
    
    # Plot Sherman function vs. Run Number
    plt.figure(figsize=(10, 6)) # Create a new figure for the summary plot
    plt.plot(summary_df['Run Number'], summary_df[sherman_col_name], 'o-', label=sherman_col_name, color='teal')
    
    plt.title('Fast Sherman Function vs. Run Number', fontsize=16)
    plt.xlabel('Run Number', fontsize=12)
    plt.ylabel(f'Sherman Function @ {fast_measure_ergloss} eV', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    summary_plot_filename = os.path.join(png_output_path, f'sherman_vs_run_number_{num_start}_{num_end + 1}.png')
    plt.savefig(summary_plot_filename, dpi=300)
    plt.close() # Close the summary plot figure
    print(f"Summary plot saved as '{summary_plot_filename}'")

    return average_fast_sherman, delta_average_fast_sherman

def compare_fast_slow(test_run_nums, average_fast_sherman, delta_average_fast_sherman, sherman_function, folder_path, 
                      fit_start_eV, fit_end_eV, txt_output_path, png_output_path, fast_measure_ergloss):
    """
    Compares the full (slow) polarization calculation with the fast method using the averaged Sherman function.

    Args:
        test_run_nums (np.ndarray): An array of run numbers to test.
        average_fast_sherman (float): The pre-calculated average Sherman function.
        delta_average_fast_sherman (float): The uncertainty in the average Sherman function.
        sherman_function (float): The theoretical Sherman function.
        folder_path (str): The root path for data files.
        fit_start_eV (float): The starting energy for the linear fit.
        fit_end_eV (float): The ending energy for the linear fit.
        txt_output_path (str): The path to save the comparison text file.
        png_output_path (str): The path to save the comparison plot.
        fast_measure_ergloss (float): The energy loss for the fast measurement.
    """
    comparison_results = []
    plot_fast_error_bar = True # Set to True to show error bars for the fast method

    for num in test_run_nums:
        filename = f'Run{num}.csv'
        filepath = os.path.join(folder_path, filename)
        
        print(f"\n{'='*20} Testing {filename} {'='*20}")
        
        try:
            energylossmax, energyloss_step, photonwavelength, count_left, count_right = readdata(filepath)
            
            # Note: fast_measure_ergloss_scan is set to None for the test runs
            (polarization_slow, delta_polarization_slow, _, 
             asymmetry_fast, delta_asymmetry_fast, _, _) = calculate_and_plot_polarization(
                count_left, count_right, sherman_function, f"{num}_test", 
                fit_start_eV, fit_end_eV, energyloss_step, txt_output_path, png_output_path, photonwavelength,
                fast_measure_ergloss, fast_measure_ergloss_scan=None
            )
            
            # Calculate fast polarization and propagate its error
            polarization_fast = asymmetry_fast / average_fast_sherman
            delta_polarization_fast = polarization_fast * np.sqrt(
                (delta_asymmetry_fast / asymmetry_fast)**2 + 
                (delta_average_fast_sherman / average_fast_sherman)**2
            )

            comparison_results.append({
                'run_number': num,
                'photonwavelength': photonwavelength,
                'polarization_slow': polarization_slow,
                'delta_polarization_slow': delta_polarization_slow,
                'polarization_fast': polarization_fast,
                'delta_polarization_fast': delta_polarization_fast,
                'sherman_function': sherman_function,
                'average_fast_sherman': average_fast_sherman,
                'delta_average_fast_sherman': delta_average_fast_sherman
            })

        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found. Skipping test run.")
        except Exception as e:
            print(f"An error occurred during testing for {filename}: {e}")

    if not comparison_results:
        print("\nNo test runs were successfully processed. Skipping comparison output.")
        return

    # Create and save the comparison table
    slow_fast_comparison_df = pd.DataFrame(comparison_results)
    test_start = test_run_nums.min()
    test_end = test_run_nums.max()
    
    # Calculate R-squared value
    y_true = slow_fast_comparison_df['polarization_slow']
    y_pred = slow_fast_comparison_df['polarization_fast']
    
    # Using Pearson correlation coefficient (R) and squaring it to get R-squared
    correlation_matrix = np.corrcoef(y_true, y_pred)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    
    print("\n\n" + "="*20 + " Slow vs. Fast Comparison " + "="*20)
    print(slow_fast_comparison_df.to_string(index=False))
    print(f"\nR-squared between slow and fast methods: {r_squared:.4f}")

    comparison_filename = os.path.join(txt_output_path, f'slow_fast_comparison_{test_start}_{test_end + 1}.txt')
    with open(comparison_filename, 'w') as f:
        f.write(slow_fast_comparison_df.to_string(index=False))
        f.write("\n\n" + "="*20 + " Analysis " + "="*20)
        f.write(f"\nR-squared between slow and fast methods: {r_squared:.4f}\n")
    print(f"\nComparison results saved to '{comparison_filename}'")
    
    # --- Plotting ---
    # Create a figure with custom layout: Top plot spans width, bottom has 2 plots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :]) # Top row, spanning all columns
    ax2 = fig.add_subplot(gs[1, 0]) # Bottom row, left column
    ax3 = fig.add_subplot(gs[1, 1]) # Bottom row, right column

    if not slow_fast_comparison_df.empty:
        # --- Plot 1: Polarization vs Run Number (Top) ---
        ax1.errorbar(slow_fast_comparison_df['run_number'], slow_fast_comparison_df['polarization_slow'], 
                     yerr=slow_fast_comparison_df['delta_polarization_slow'], fmt='o', label='Polarization (Standard Method)', 
                     color='crimson', capsize=3, markersize=5)
        
        ax1.errorbar(slow_fast_comparison_df['run_number'], slow_fast_comparison_df['polarization_fast'], 
                     yerr=slow_fast_comparison_df['delta_polarization_fast'] if plot_fast_error_bar else None, 
                     fmt='s', label=f'Polarization Fast (@{fast_measure_ergloss}eV)', color='dodgerblue', capsize=3, markersize=5)
        
        # Add extra info to legend (R-squared removed from here)
        ax1.plot([], [], ' ', label=f'Fast Sherman Energy = {fast_measure_ergloss} eV')
        ax1.plot([], [], ' ', label=f'Avg. Fast Sherman = {average_fast_sherman:.4f} \u00B1 {delta_average_fast_sherman:.4f}')

        ax1.set_title('Comparison of Standard vs. Fast Polarization Calculation', fontsize=14)
        ax1.set_xlabel('Run Number', fontsize=12)
        ax1.set_ylabel('Polarization (%)', fontsize=12)
        ax1.grid(True)
        # Place legend outside the plot
        ax1.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), 
                   facecolor='white', edgecolor='black')

        # --- Plot 2: Slow vs. Fast with y=kx fit (Bottom Left) ---
        x_data = slow_fast_comparison_df['polarization_fast']
        y_data = slow_fast_comparison_df['polarization_slow']

        # Scatter plot (no error bars as requested)
        ax2.scatter(x_data, y_data, label='Data Points', alpha=0.7)

        # Linear fit y=kx (intercept forced to 0)
        k = 0
        if np.sum(x_data**2) > 0:
            k = np.sum(x_data * y_data) / np.sum(x_data**2)

        # Create line for plotting the fit
        fit_range = np.array([0, 100])
        ax2.plot(fit_range, k * fit_range, '--', color='green', label=f'Linear Fit (y = {k:.4f}x)')

        # Add an ideal y=x line for reference
        ax2.plot(fit_range, fit_range, ':', color='gray', label='Ideal (y = x)')
        
        # Add R-squared to legend here
        ax2.plot([], [], ' ', label=f'R-squared = {r_squared:.4f}')

        ax2.set_title('Standard vs. Fast Polarization Correlation', fontsize=14)
        ax2.set_xlabel('Polarization Fast (%)', fontsize=12)
        ax2.set_ylabel('Polarization Standard (%)', fontsize=12)
        ax2.grid(True)
        # Gray box with white background
        ax2.legend(facecolor='white', edgecolor='gray', loc='upper left')
        
        # Set axis limits and aspect ratio
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_aspect('equal', adjustable='box')

        # --- Plot 3: Error Comparison (Bottom Right) ---
        ax3.plot(slow_fast_comparison_df['run_number'], slow_fast_comparison_df['delta_polarization_slow'], 
                 'o-', label='Uncertainty Standard', color='crimson', markersize=4, alpha=0.8)
        ax3.plot(slow_fast_comparison_df['run_number'], slow_fast_comparison_df['delta_polarization_fast'], 
                 's-', label='Uncertainty Fast', color='dodgerblue', markersize=4, alpha=0.8)
        
        ax3.set_title('Polarization Uncertainty Comparison', fontsize=14)
        ax3.set_xlabel('Run Number', fontsize=12)
        ax3.set_ylabel('Uncertainty (%)', fontsize=12)
        ax3.grid(True)
        ax3.legend(facecolor='white', edgecolor='black')

    comparison_plot_filename = os.path.join(png_output_path, f'slow_fast_comparison_{test_start}_{test_end + 1}.png')
    plt.savefig(comparison_plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved as '{comparison_plot_filename}'")


def eva_train_data(run_nums, sherman_function, folder_path, txt_output_path, png_output_path, 
                   fit_start_eV, fit_end_eV, fast_measure_ergloss_single, start_average_run_num, 
                   fast_measure_ergloss_scan=None):
    """
    Evaluates the training data to generate Sherman function calibration.
    
    Checks for existing summary file first. If not found, processes all runs.
    Generates a summary for the single fast energy and a detailed scan file if a range is provided.

    Returns:
        list: sherman_results for the single fast energy.
    """
    sherman_results = []
    sherman_results_ergscan = []

    # Check if a summary file already exists to avoid re-running the calibration
    num_start = run_nums.min()
    num_end = run_nums.max()
    # Include fast_measure_ergloss_single in the filename
    summary_filename = os.path.join(txt_output_path, f'fast_sherman_{fast_measure_ergloss_single}eV_{num_start}_{num_end + 1}.txt')

    # If the summary for the single energy exists, we can skip processing.
    # We assume if this file exists, the scan file also exists if it was requested.
    # Note: This check is only valid if fast_measure_ergloss_scan is None or if the file-saving
    # logic is expanded to include the scan parameters in the filename.
    # For now, we'll only check for the single-energy summary.
    
    if os.path.exists(summary_filename):
        print(f"Found existing summary file: {summary_filename}. Loading results from file.")
        # Use sep='\s+' to address the deprecation warning and correctly parse whitespace.
        # skiprows=1 skips the header row that causes parsing issues.
        # header=None indicates that the file being read has no header.
        summary_df = pd.read_csv(summary_filename, sep='\s+', skiprows=1, header=None)
        # Explicitly assign the correct column names.
        summary_df.columns = ['Run Number', 'Photon Wavelength (nm)', f'Sherman @ {fast_measure_ergloss_single}eV']
        sherman_results = summary_df.to_dict('records')
        # We can't reconstruct the ergscan, so we'll skip it if we load from file.
        # This assumes the main goal is the eva_test_data part.
        print("Skipping erg-scan generation as summary file was loaded.")
        
    else:
        print(f"Summary file not found. Processing calibration runs {num_start} to {num_end}...")
        for num in run_nums:
            filename = f'Run{num}.csv'
            filepath = os.path.join(folder_path, filename)
            
            print(f"\n{'='*20} Processing {filename} {'='*20}")
            
            try:
                # Read data
                energylossmax, energyloss_step, photonwavelength, count_left, count_right = readdata(filepath)
                
                print("--- Data Loading ---")
                print(f"Successfully loaded data from '{filepath}'")
                print(f"Photon Wavelength: {photonwavelength} nm")
                print(f"Max Energy Loss: {energylossmax} eV")
                print(f"Energy Loss Step: {energyloss_step} eV")
                
                # Perform calculations and plotting
                (polarization, delta_polarization, asymmetry_0, 
                 asymmetry_fast_single, delta_asymmetry_fast_single, 
                 asymmetry_scan_map, delta_asymmetry_scan_map) = calculate_and_plot_polarization(
                    count_left, count_right, sherman_function, num, 
                    fit_start_eV, fit_end_eV, energyloss_step, txt_output_path, png_output_path, photonwavelength,
                    fast_measure_ergloss_single, fast_measure_ergloss_scan
                )
                
                print("\n--- Final Results ---")
                print(f"Photon Wavelength: {photonwavelength} nm")
                print(f"Extrapolated Asymmetry at 0 eV (Asymmetry_0): {asymmetry_0:.4f}%")
                print(f"Sherman Function: {sherman_function}")
                print(f"Final Calculated Polarization: {polarization:.4f} +/- {delta_polarization:.4f} %")
                
                # Calculate and store sherman_fast for the single energy
                sherman_fast_single = fast_sherman(polarization, asymmetry_fast_single)
                sherman_results.append({
                    'Run Number': num,
                    'Photon Wavelength (nm)': photonwavelength,
                    f'Sherman @ {fast_measure_ergloss_single}eV': sherman_fast_single
                })

                # Calculate and store sherman_fast for the energy scan
                if fast_measure_ergloss_scan is not None:
                    for erg, asym in asymmetry_scan_map.items():
                        # Skip if asymmetry was NaN (e.g., missing energy value)
                        if pd.isna(asym):
                            continue
                            
                        sherman_fast_scan = fast_sherman(polarization, asym)
                        delta_asym_scan = delta_asymmetry_scan_map[erg]
                        
                        # Propagate error for the scanned sherman value
                        delta_sherman_fast_scan = sherman_fast_scan * np.sqrt(
                            (delta_polarization / polarization)**2 + 
                            (delta_asym_scan / asym)**2
                        )
                        
                        sherman_results_ergscan.append({
                            'Run Number': num,
                            'Photon Wavelength (nm)': photonwavelength,
                            'fast_measure_ergloss': erg,
                            'sherman_fast': sherman_fast_scan,
                            'delta_sherman_fast': delta_sherman_fast_scan
                        })
                
            except FileNotFoundError:
                print(f"Error: The file '{filepath}' was not found. Skipping.")
            except Exception as e:
                print(f"An error occurred during processing for {filename}: {e}")

        # Save the detailed energy scan results
        if sherman_results_ergscan:
            scan_df = pd.DataFrame(sherman_results_ergscan)
            scan_filename = os.path.join(txt_output_path, f'sherman_results_ergscan_{num_start}_{num_end + 1}.txt')
            with open(scan_filename, 'w') as f:
                f.write(scan_df.to_string(index=False))
            print(f"\nDetailed energy scan results saved to '{scan_filename}'")

            # --- Generate Plot for Energy Scan ---
            if not scan_df.empty:
                print(f"Generating energy scan summary plot...")
                # Change layout to 1 row, 3 columns, and increase total width
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(38, 8))
                # Adjust spacing
                plt.subplots_adjust(wspace=0.5)

                # --- Left Subplot: Sherman vs. Run Number for different energies ---
                pivot_df = scan_df.pivot(index='Run Number', columns='fast_measure_ergloss', values='sherman_fast')
                
                # Use a color map with more distinct colors (tab20)
                num_colors = len(pivot_df.columns)
                colors = plt.cm.tab20.colors # Get the discrete color tuple
                
                for i, erg in enumerate(pivot_df.columns):
                    ax1.scatter(pivot_df.index, pivot_df[erg], label=f'{erg} eV', 
                                color=colors[i % len(colors)], s=5, alpha=0.7)
                
                ax1.set_title('Sherman Function vs. Run Number (Training Data)', fontsize=14)
                ax1.set_xlabel('Run Number', fontsize=12)
                ax1.set_ylabel('Fast Sherman Function (S_eff)', fontsize=12)
                # Move legend outside the plot and add a box
                ax1.legend(title="Energy Loss", markerscale=4, loc='center left', 
                           bbox_to_anchor=(1.02, 0.5), facecolor='white', edgecolor='black')
                ax1.grid(True)

                # --- Middle Subplot (ax2): Average Sherman vs. Energy Loss (SEM Error) ---
                # Group by energy and calculate mean, SEM, min, and max
                avg_scan_df = scan_df.groupby('fast_measure_ergloss')['sherman_fast'].agg(
                    mean='mean',
                    sem=lambda x: x.std(ddof=1) / np.sqrt(x.count()), # ddof=1 for sample std dev
                    min='min',
                    max='max'
                ).reset_index().dropna() # Dropna in case of single-point groups (std is NaN)

                # Main plot on ax2
                ax2.errorbar(avg_scan_df['fast_measure_ergloss'], avg_scan_df['mean'], yerr=avg_scan_df['sem'],
                             fmt='o', capsize=5, label='Mean S_eff (SEM)', color='C0')
                
                ax2.set_title('Average Sherman Function vs. Energy Loss (SEM)', fontsize=14)
                ax2.set_xlabel('Energy Loss (eV)', fontsize=12)
                ax2.set_ylabel('Average Fast Sherman Function (S_eff)', fontsize=12, color='C0')
                ax2.tick_params(axis='y', labelcolor='C0')
                ax2.grid(True)
                
                # Create a twin axis for ax2
                ax2_twin = ax2.twinx()
                
                # Calculate Error Bar Length (2 * SEM)
                error_bar_length_sem = 2 * avg_scan_df['sem']
                
                # Plot Error Bar Length on twin axis
                ax2_twin.plot(avg_scan_df['fast_measure_ergloss'], error_bar_length_sem, 'r--', label='Error Bar Length (2*SEM)')
                
                ax2_twin.set_ylabel('Error Bar Length (2*SEM)', color='r', fontsize=12)
                ax2_twin.tick_params(axis='y', labelcolor='r')
                ax2_twin.grid(False) # Turn off grid for the twin axis
                
                # Combine legends from both axes
                lines, labels = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='best', facecolor='white', edgecolor='black')
                
                # --- Right Subplot (ax3): Average Sherman vs. Energy Loss (Min/Max Error) ---
                
                # Calculate error bar range (distance from mean)
                y_err_lower = avg_scan_df['mean'] - avg_scan_df['min']
                y_err_upper = avg_scan_df['max'] - avg_scan_df['mean']
                y_err_minmax = [y_err_lower, y_err_upper]

                # Main plot on ax3
                ax3.errorbar(avg_scan_df['fast_measure_ergloss'], avg_scan_df['mean'], yerr=y_err_minmax,
                             fmt='o', capsize=5, label='Mean S_eff (Min/Max Range)', color='C0')
                
                ax3.set_title('Average Sherman Function vs. Energy Loss (Min/Max)', fontsize=14)
                ax3.set_xlabel('Energy Loss (eV)', fontsize=12)
                ax3.set_ylabel('Average Fast Sherman Function (S_eff)', fontsize=12, color='C0')
                ax3.tick_params(axis='y', labelcolor='C0')
                ax3.grid(True)
                
                # Create a twin axis for ax3
                ax3_twin = ax3.twinx()
                
                # Calculate Error Bar Length (Max - Min)
                error_bar_length_minmax = avg_scan_df['max'] - avg_scan_df['min']
                
                # Plot Error Bar Length on twin axis
                ax3_twin.plot(avg_scan_df['fast_measure_ergloss'], error_bar_length_minmax, 'r--', label='Error Bar Length (Max-Min)')
                
                ax3_twin.set_ylabel('Error Bar Length (Max-Min)', color='r', fontsize=12)
                ax3_twin.tick_params(axis='y', labelcolor='r')
                ax3_twin.grid(False)
                
                # Combine legends from both axes
                lines, labels = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3_twin.get_legend_handles_labels()
                ax3.legend(lines + lines2, labels + labels2, loc='best', facecolor='white', edgecolor='black')

                # Save the figure
                scan_plot_filename = os.path.join(png_output_path, f'sherman_scan_analysis_{num_start}_{num_end + 1}.png')
                plt.savefig(scan_plot_filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Energy scan summary plot saved as '{scan_plot_filename}'")
            
    return sherman_results

def eva_test_data(sherman_results, run_nums, txt_output_path, png_output_path, 
                  fast_measure_ergloss_single, start_average_run_num, 
                  sherman_function, folder_path, fit_start_eV, fit_end_eV, 
                  test_run_nums):
    """
    Evaluates the test data using the calibrated average Sherman function.
    
    Args:
        sherman_results (list): The results from the training data evaluation.
        (other args): Pass-through of parameters from main().
    """
    if sherman_results:
        # Get the calibrated average sherman function and its error
        average_fast_sherman, delta_average_fast_sherman = fast_sherman_output(
            sherman_results, run_nums, txt_output_path, png_output_path, 
            fast_measure_ergloss_single, start_average_run_num
        )

        # --- Part 2: Compare fast and slow methods on new data ---
        compare_fast_slow(
            test_run_nums, average_fast_sherman, delta_average_fast_sherman, 
            sherman_function, folder_path,
            fit_start_eV, fit_end_eV, txt_output_path, png_output_path, 
            fast_measure_ergloss_single
        )

def main():
    """
    Main function to run the Mott polarimeter analysis.
    """
    # 0. Define Sherman function and other parameters
    sherman_function = 0.235
    folder_path = '/Users/wange/Coding/Python/ai_spin_measurement/data/mini_mott/'
    
    # Define output folders
    txt_output_path = os.path.join(folder_path, 'resulttxt')
    png_output_path = os.path.join(folder_path, 'resultpng')
    
    # Create output directories if they don't exist
    os.makedirs(txt_output_path, exist_ok=True)
    os.makedirs(png_output_path, exist_ok=True)
    
    # Define the energy range (in eV) for the linear fit
    fit_start_eV = 20
    fit_end_eV = 80
    
    # --- Configuration for Training and Testing ---
    
    # 1. Define single energy for the "fast sherman" comparison
    fast_measure_ergloss_single = 40 
    
    # 2. Define a range of energies to scan for a detailed sherman report
    # Set to None to disable this feature
    fast_measure_ergloss_scan = np.arange(20, 170, 10) # Scans 20, 30, ..., 80 eV
    
    # 3. Define run numbers for calibration
    run_nums = np.arange(790, 1200)
    run_nums = np.atleast_1d(run_nums)
    start_average_run_num = 800 # Run number to start averaging from
    
    # 4. Define run numbers for testing
    test_start, test_end = 1200, 1540
    test_run_nums = np.arange(test_start, test_end)
        
    # --- Execute Analysis ---
    
    # 1. Evaluate training data (or load from file)
    # This generates the calibration data
    sherman_results = eva_train_data(
        run_nums, sherman_function, folder_path, txt_output_path, png_output_path,
        fit_start_eV, fit_end_eV, fast_measure_ergloss_single, start_average_run_num,
        fast_measure_ergloss_scan
    )

    # 2. Evaluate test data
    # This uses the calibration data to perform the fast vs. slow comparison
    eva_test_data(
        sherman_results, run_nums, txt_output_path, png_output_path,
        fast_measure_ergloss_single, start_average_run_num, sherman_function,
        folder_path, fit_start_eV, fit_end_eV, test_run_nums
    )


if __name__ == "__main__":
    main()