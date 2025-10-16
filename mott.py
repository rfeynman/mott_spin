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

def calculate_and_plot_polarization(count_left, count_right, sherman_function, run_num, fit_start_eV, fit_end_eV, energyloss_step, folder_path, photonwavelength):
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
        folder_path (str): The path to save output files.
        photonwavelength (float): The photon wavelength of the run in nm.

    Returns:
        tuple: A tuple containing:
            - polarization (float): The final calculated polarization.
            - asymmetry_0 (float): The extrapolated asymmetry at 0 energy loss.
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
    asymmetry_filename = os.path.join(folder_path, f'asymmetry_Run{run_num}.txt')
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
    plot_filename = os.path.join(folder_path, f'asymmetry_fit_plot_Run{run_num}.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"\nPlot has been saved as '{plot_filename}'")

    return polarization, asymmetry_0


def main():
    """
    Main function to run the Mott polarimeter analysis.
    """
    # 0. Define Sherman function and other parameters
    sherman_function = 0.27
    folder_path = '/Users/wange/Coding/Python/ai_spin_measurement/data/mini_mott/'
    
    # Define the energy range (in eV) for the linear fit
    fit_start_eV = 10
    fit_end_eV = 70
    
    # 1. Define run numbers to process
    # Examples:
    run_nums = 1242                  # Single value
    # run_nums = [1010, 1011, 1012]    # List of values
    #run_nums = np.arange(1010, 1012) # Range of values (e.g., 1010, 1011)
    
    # Ensure run_nums is always iterable
    run_nums = np.atleast_1d(run_nums)
    
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
            polarization, asymmetry_0 = calculate_and_plot_polarization(
                count_left, count_right, sherman_function, num, 
                fit_start_eV, fit_end_eV, energyloss_step, folder_path, photonwavelength
            )
            
            print("\n--- Final Results ---")
            print(f"Photon Wavelength: {photonwavelength} nm")
            print(f"Extrapolated Asymmetry at 0 eV (Asymmetry_0): {asymmetry_0:.4f}%")
            print(f"Sherman Function: {sherman_function}")
            print(f"Final Calculated Polarization: {polarization:.4f} %")
            
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found. Skipping.")
        except Exception as e:
            print(f"An error occurred during processing for {filename}: {e}")

if __name__ == "__main__":
    main()

