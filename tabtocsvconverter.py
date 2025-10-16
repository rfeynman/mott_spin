import pandas as pd
import os

def convert_txt_to_csv(input_txt_file, output_csv_file):
    """
    Reads a tab-separated text file and converts it to a CSV file.

    Args:
        input_txt_file (str): The path to the input text file.
        output_csv_file (str): The path for the output CSV file.
    """
    try:
        # Read the tab-delimited file into a pandas DataFrame.
        # The `sep='\t'` argument specifies that the file is separated by tabs.
        df = pd.read_csv(input_txt_file, sep='\t')

        # Write the DataFrame to a CSV file.
        # `index=False` prevents writing the DataFrame's row numbers into the file.
        df.to_csv(output_csv_file, index=False)

        print(f"Successfully converted '{input_txt_file}' to '{output_csv_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_txt_file}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing {input_txt_file}: {e}")

if __name__ == '__main__':
    # Define the folder containing your .xls files and the folder for output.
    # '.' means the script will look in the directory it is located in.
    input_folder = '.'
    output_folder = 'converted_csv_files'

    # Create the output folder if it doesn't already exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Loop through every file in the input folder.
    for filename in os.listdir(input_folder):
        # Check if the file has the .xls extension.
        if filename.endswith('.xls'):
            input_file_path = os.path.join(input_folder, filename)
            
            # Create the new filename by replacing .xls with .csv.
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.csv"
            output_file_path = os.path.join(output_folder, output_filename)
            
            # Call the conversion function for each matching file.
            convert_txt_to_csv(input_file_path, output_file_path)

    print("\nBatch conversion complete.")


