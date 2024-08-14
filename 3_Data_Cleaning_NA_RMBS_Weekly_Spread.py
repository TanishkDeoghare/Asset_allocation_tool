import pandas as pd
import os
import glob

# Constant base file name and sheet name
base_file_name = "Non-Agency RMBS - Weekly Spreads Report"
sheet_name = "Non-Agency RMBS Spreads"
cwd = os.getcwd()

# Input folder path
folder_path = os.path.join(cwd, 'Input')

# Use glob to find the latest file matching the pattern
file_pattern = os.path.join(folder_path, f"{base_file_name}*.xlsx")
list_of_files = glob.glob(file_pattern)
print(list_of_files)
if not list_of_files:
    raise FileNotFoundError(f"No files found matching pattern {file_pattern}")

# Select the most recent file
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
# Define the columns to read by their indices
columns_to_read = [1] + list(range(105, 126)) + list(range(169, 176))  # 0-indexed for pandas

# Read the first four rows to concatenate their text
first_four_rows = pd.read_excel(latest_file, sheet_name=sheet_name, usecols=columns_to_read, skiprows=2, nrows=3, header=None)

# Concatenate the text from the first four rows
concatenated_row = [first_four_rows.iloc[0, 0]] + [' '.join(first_four_rows[col].dropna().astype(str)) for col in first_four_rows.columns[1:]]

# Read the remaining data, skipping the first 9 rows
df = pd.read_excel(latest_file, sheet_name=sheet_name, usecols=columns_to_read, skiprows=1, header=None)

# Create a new DataFrame with the concatenated row as the first row
new_df = pd.DataFrame([concatenated_row], columns=df.columns)
new_df = pd.concat([new_df, df], ignore_index=True)
new_df = new_df.drop([1, 2, 3, 4])
new_df.columns = new_df.iloc[0]

new_columns = ['Date'] + [
    f"Prime 2.0 {col}" if col in new_df.columns[1:len(range(105, 126))+1] else f"Non-QM {col}" if col in new_df.columns[len(range(105, 126))+1:] else col
    for col in new_df.columns[1:]
]
new_df.columns = new_columns

new_df = new_df.drop([0])
new_df = new_df.reset_index(drop=True)

columns_to_keep = ['Date'] + [col for col in new_df.columns if 'Spread' in col or 'Non-QM' in col]
filtered_df = new_df[columns_to_keep]

# Output file path
output_file_path = os.path.join(cwd, "Intermediate_results/Filtered_WF_Spreads_Report.xlsx")

# Save the new DataFrame to an Excel file
filtered_df.to_excel(output_file_path, index=False)

# Display the output file path
print(f"The cleaned data has been saved to: {output_file_path}")
