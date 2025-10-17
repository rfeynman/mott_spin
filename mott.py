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

def calculate_and_plot_polarization(count_left, count_right, sherman_function, run_num, fit_start_eV, fit_end_eV, energyloss_step, txt_output_path, png_output_path, photonwavelength, fast_measure_ergloss):
    """
    Calculates asymmetry and polarization, saves results to file, and plots the data.

    Args:
        count_left (pd.DataFrame): DataFrame for the left detector counts.
        count_right (pd.DataFrame): DataFrame for the right detector counts.
        sherman_function (float): The Sherman function value.
        run_num (int or str): The run number, used for output filenames.
        fit_start_eV (float): The starting energy loss for the linear fit.
        fit_end_eV (float): The ending energy loss for the linear fit.
        energyloss_step (float): The energy loss step, for defining the fit range.
        txt_output_path (str): The path to save output text files.
        png_output_path (str): The path to save output plot files.
        photonwavelength (float): The photon wavelength of the run in nm.
        fast_measure_ergloss (float): The energy loss for the fast sherman measurement.

    Returns:
        tuple: A tuple containing:
            - polarization (float): The final calculated polarization.
            - asymmetry_0 (float): The extrapolated asymmetry at 0 energy loss.
            - asymmetry_fast_measure_ergloss (float): The asymmetry at the specified fast measurement energy loss.
    """
    # 8. Create removebackground tables
    data_cols = ['X1', 'X2', 'Y1', 'Y2']
    
    # Get background counts (where energy loss is 0)
    background_left = count_left.loc[0, data_cols]
    background_right = count_right.loc[0, data_cols]
    
    removebackground_left = count_left.copy()
    removebackground_right = count_right.copy()
    
    # Subtract background from all other data points
    for col in data_cols:
        removebackground_left[col] = removebackground_left[col] - background_left[col]
        removebackground_right[col] = removebackground_right[col] - background_right[col]
        
    # We perform calculations on data where energy loss is not zero
    calc_df_left = removebackground_left.iloc[1:].reset_index(drop=True)
    calc_df_right = removebackground_right.iloc[1:].reset_index(drop=True)
    
    # Create a results DataFrame to store intermediate calculations
    results = pd.DataFrame()
    results['Energy loss, eV'] = calc_df_left['Energy loss, eV']

    # 9. Calculate asymmetry for X and Y directions
    with np.errstate(divide='ignore', invalid='ignore'):
        term_x = np.sqrt(
            calc_df_left['X1'] * calc_df_right['X2'] / 
            (calc_df_left['X2'] * calc_df_right['X1'])
        )
        results['asymmetry_x'] = (term_x - 1) / (term_x + 1)

        term_y = np.sqrt(
            calc_df_left['Y1'] * calc_df_right['Y2'] / 
            (calc_df_left['Y2'] * calc_df_right['Y1'])
        )
        results['asymmetry_y'] = (term_y - 1) / (term_y + 1)
    
    # 10. Calculate spin direction
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(results['asymmetry_x'] / results['asymmetry_y'])
        results['spindirection'] = np.arctan(1 / ratio)

    # 11. Calculate total asymmetry
    results['asymmetry'] = np.abs(results['asymmetry_y'] / np.sin(results['spindirection'])) * 100
    
    results.replace([np.inf, -np.inf], np.nan, inplace=True)
    results.dropna(inplace=True)
    
    print("\n--- Calculated Asymmetry Results ---")
    print(results.to_string())
    
    # Save the results to a txt file in the specified folder
    asymmetry_filename = os.path.join(txt_output_path, f'asymmetry_Run{run_num}.txt')
    with open(asymmetry_filename, 'w') as f:
        f.write(f"Photon Wavelength: {photonwavelength} nm\n")
        f.write("="*40 + "\n")
        f.write(results.to_string())
    print(f"Asymmetry results saved to '{asymmetry_filename}'")

    # 12. Fit and find asymmetry at energy loss = 0
    # Create the fitting range dynamically
    fit_range = np.arange(fit_start_eV, fit_end_eV + energyloss_step, energyloss_step)
    fit_points = results[results['Energy loss, eV'].isin(fit_range)]
    x_fit = fit_points['Energy loss, eV']
    y_fit = fit_points['asymmetry']
    
    # Perform linear fit (degree 1 polynomial)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    asymmetry_0 = intercept

    # 13. Calculate the polarization
    polarization = asymmetry_0 / sherman_function
    
    # Extract asymmetry at the specified fast measurement energy loss
    asymmetry_fast_measure_ergloss = results.loc[results['Energy loss, eV'] == fast_measure_ergloss, 'asymmetry'].iloc[0]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results['Energy loss, eV'], results['asymmetry'], 'o', 
            label='Calculated Asymmetry', color='royalblue', alpha=0.7)
    
    ax.plot(x_fit, y_fit, 'o', label=f'Points for Fit ({fit_start_eV}-{fit_end_eV} eV)', 
            color='red', markersize=10)
    
    x_line = np.array([0, results['Energy loss, eV'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color='black', label=f'Linear Fit (y={slope:.4f}x + {intercept:.2f})')
    
    ax.plot(0, asymmetry_0, 'X', color='darkorange', markersize=12, 
            label=f'Asymmetry at 0 eV = {asymmetry_0:.2f}%')
    
    # Add an empty plot with a label for the polarization
    ax.plot([], [], ' ', label=f'Polarization = {polarization:.2f} %')
    
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

    return polarization, asymmetry_0, asymmetry_fast_measure_ergloss

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
    Summarizes results, calculates an average Sherman value, and exports outputs.

    Args:
        sherman_results (list): A list of dictionaries containing the results for each run.
        run_nums (np.ndarray): An array of the run numbers that were processed.
        txt_output_path (str): The path to save the summary text file.
        png_output_path (str): The path to save the summary plot.
        fast_measure_ergloss (float): The energy loss used for the fast Sherman measurement.
        start_average (int): The run number from which to start averaging the Sherman function.

    Returns:
        float: The average fast Sherman function value.
    """
    print("\n\n" + "="*25 + " Summary " + "="*25)
    summary_df = pd.DataFrame(sherman_results)
    print(summary_df.to_string(index=False))
    
    # Save the summary DataFrame to a text file
    num_start = run_nums.min()
    num_end = run_nums.max()
    summary_filename = os.path.join(txt_output_path, f'fast_sherman_{num_start}_{num_end + 1}.txt')
    with open(summary_filename, 'w') as f:
        f.write(summary_df.to_string(index=False))
    print(f"\nSummary results saved to '{summary_filename}'")
    
    sherman_col_name = f'Sherman @ {fast_measure_ergloss}eV'

    # Calculate the average of the fast sherman function
    averaging_df = summary_df[summary_df['Run Number'] >= start_average]
    average_fast_sherman = averaging_df[sherman_col_name].mean()
    print(f"\nAverage Sherman @ {fast_measure_ergloss}eV (from run {start_average} onwards): {average_fast_sherman:.4f}")
    
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

    return average_fast_sherman

def compare_fast_slow(test_run_nums, average_fast_sherman, sherman_function, folder_path, fit_start_eV, fit_end_eV, txt_output_path, png_output_path, fast_measure_ergloss):
    """
    Compares the full (slow) polarization calculation with the fast method using the averaged Sherman function.

    Args:
        test_run_nums (np.ndarray): An array of run numbers to test.
        average_fast_sherman (float): The pre-calculated average Sherman function.
        sherman_function (float): The theoretical Sherman function.
        folder_path (str): The root path for data files.
        fit_start_eV (float): The starting energy for the linear fit.
        fit_end_eV (float): The ending energy for the linear fit.
        txt_output_path (str): The path to save the comparison text file.
        png_output_path (str): The path to save the comparison plot.
        fast_measure_ergloss (float): The energy loss for the fast measurement.
    """
    comparison_results = []

    for num in test_run_nums:
        filename = f'Run{num}.csv'
        filepath = os.path.join(folder_path, filename)
        
        print(f"\n{'='*20} Testing {filename} {'='*20}")
        
        try:
            energylossmax, energyloss_step, photonwavelength, count_left, count_right = readdata(filepath)
            
            polarization_slow, _, asymmetry_fast = calculate_and_plot_polarization(
                count_left, count_right, sherman_function, f"{num}_test", 
                fit_start_eV, fit_end_eV, energyloss_step, txt_output_path, png_output_path, photonwavelength,
                fast_measure_ergloss
            )
            
            polarization_fast = asymmetry_fast / average_fast_sherman
            
            comparison_results.append({
                'run_number': num,
                'photonwavelength': photonwavelength,
                'polarization_slow': polarization_slow,
                'polarization_fast': polarization_fast,
                'sherman_function': sherman_function,
                'average_fast_sherman': average_fast_sherman
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
    # Create a figure with two subplots, side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Plot 1: Polarization vs Run Number ---
    ax1.plot(slow_fast_comparison_df['run_number'], slow_fast_comparison_df['polarization_slow'], 'o', label='Polarization Slow (Full Fit)', color='crimson')
    ax1.plot(slow_fast_comparison_df['run_number'], slow_fast_comparison_df['polarization_fast'], 's', label=f'Polarization Fast (@{fast_measure_ergloss}eV)', color='dodgerblue')
    
    # Add extra info to legend
    ax1.plot([], [], ' ', label=f'Fast Sherman Energy = {fast_measure_ergloss} eV')
    ax1.plot([], [], ' ', label=f'Avg. Fast Sherman = {average_fast_sherman:.4f}')
    ax1.plot([], [], ' ', label=f'R-squared = {r_squared:.4f}')

    ax1.set_title('Comparison of Slow vs. Fast Polarization Calculation', fontsize=14)
    ax1.set_xlabel('Run Number', fontsize=12)
    ax1.set_ylabel('Polarization (%)', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    # --- Plot 2: Slow vs. Fast with y=kx fit ---
    x_data = slow_fast_comparison_df['polarization_fast']
    y_data = slow_fast_comparison_df['polarization_slow']

    # Scatter plot
    ax2.scatter(x_data, y_data, label='Data Points', alpha=0.7)

    # Linear fit y=kx (intercept forced to 0)
    # k = sum(x*y) / sum(x^2)
    k = np.sum(x_data * y_data) / np.sum(x_data**2)

    # Create line for plotting the fit
    fit_range = np.array([0, 100])
    ax2.plot(fit_range, k * fit_range, '--', color='green', label=f'Linear Fit (y = {k:.4f}x)')

    # Add an ideal y=x line for reference
    ax2.plot(fit_range, fit_range, ':', color='gray', label='Ideal (y = x)')

    ax2.set_title('Slow vs. Fast Polarization Correlation', fontsize=14)
    ax2.set_xlabel('Polarization Fast (%)', fontsize=12)
    ax2.set_ylabel('Polarization Slow (%)', fontsize=12)
    ax2.grid(True)
    ax2.legend()
    
    # Set axis limits and aspect ratio
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_aspect('equal', adjustable='box')


    # Adjust overall layout to make room for the legend
    plt.subplots_adjust(wspace=0.5)

    comparison_plot_filename = os.path.join(png_output_path, f'slow_fast_comparison_{test_start}_{test_end + 1}.png')
    plt.savefig(comparison_plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved as '{comparison_plot_filename}'")


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
    fast_measure_ergloss = 60 # Define the energy for the fast Sherman measurement
    
    # 1. Define run numbers for calibration
    run_nums = np.arange(10, 1200) # Range of values (e.g., 1010 to 1014)
    run_nums = np.atleast_1d(run_nums)
    start_average_run_num = 800 # Run number to start averaging from
    
    sherman_results = []

    # Check if a summary file already exists to avoid re-running the calibration
    num_start = run_nums.min()
    num_end = run_nums.max()
    summary_filename = os.path.join(txt_output_path, f'fast_sherman_{num_start}_{num_end + 1}.txt')

    if os.path.exists(summary_filename):
        print(f"Found existing summary file: {summary_filename}. Loading results from file.")
        # Use sep='\s+' to address the deprecation warning and correctly parse whitespace.
        # skiprows=1 skips the header row that causes parsing issues.
        # header=None indicates that the file being read has no header.
        summary_df = pd.read_csv(summary_filename, sep='\s+', skiprows=1, header=None)
        # Explicitly assign the correct column names.
        summary_df.columns = ['Run Number', 'Photon Wavelength (nm)', f'Sherman @ {fast_measure_ergloss}eV']
        sherman_results = summary_df.to_dict('records')
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
                polarization, asymmetry_0, asymmetry_fast = calculate_and_plot_polarization(
                    count_left, count_right, sherman_function, num, 
                    fit_start_eV, fit_end_eV, energyloss_step, txt_output_path, png_output_path, photonwavelength,
                    fast_measure_ergloss
                )
                
                print("\n--- Final Results ---")
                print(f"Photon Wavelength: {photonwavelength} nm")
                print(f"Extrapolated Asymmetry at 0 eV (Asymmetry_0): {asymmetry_0:.4f}%")
                print(f"Sherman Function: {sherman_function}")
                print(f"Final Calculated Polarization: {polarization:.4f} %")
                
                # Calculate and store sherman_fast
                sherman_fast = fast_sherman(polarization, asymmetry_fast)
                sherman_results.append({
                    'Run Number': num,
                    'Photon Wavelength (nm)': photonwavelength,
                    f'Sherman @ {fast_measure_ergloss}eV': sherman_fast
                })
                
            except FileNotFoundError:
                print(f"Error: The file '{filepath}' was not found. Skipping.")
            except Exception as e:
                print(f"An error occurred during processing for {filename}: {e}")

    # After the loop, process and save the summary results
    if sherman_results:
        average_fast_sherman = fast_sherman_output(
            sherman_results, run_nums, txt_output_path, png_output_path, fast_measure_ergloss, start_average_run_num
        )

        # --- Part 2: Compare fast and slow methods on new data ---
        test_start, test_end = 1200,1540
        test_run_nums = np.arange(test_start, test_end)
        
        compare_fast_slow(
            test_run_nums, average_fast_sherman, sherman_function, folder_path,
            fit_start_eV, fit_end_eV, txt_output_path, png_output_path, fast_measure_ergloss
        )


if __name__ == "__main__":
    main()

