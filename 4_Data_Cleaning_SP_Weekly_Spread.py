import pandas as pd
import os
import glob

# Constant base file name and sheet name
base_file_name = "SP spread time series"
sheet_name = "Master Table"

cwd = os.getcwd()

# Input folder path
folder_path = os.path.join(cwd, 'Input')

# Use glob to find files matching the pattern
file_pattern = os.path.join(folder_path, f"{base_file_name}*.xlsx")
list_of_files = glob.glob(file_pattern)
print(list_of_files)
if not list_of_files:
    raise FileNotFoundError(f"No files found matching pattern {file_pattern}")

# Select the most recent file by creation time
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
# Read the "Master Table" sheet from the most recent Excel file
df = pd.read_excel(latest_file, sheet_name=sheet_name, skiprows=9)

# Output file path
output_file_path = os.path.join(cwd, "Intermediate_results/Cleaned_BoFA_Master_Table.xlsx")

# Save the DataFrame to an Excel file
df.to_excel(output_file_path, index=False)

# Display the output file path
print(f"The cleaned data has been saved to: {output_file_path}")
