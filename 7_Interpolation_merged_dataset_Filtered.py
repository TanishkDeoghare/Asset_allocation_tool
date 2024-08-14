import pandas as pd
import os

cwd = os.getcwd()

# Define file paths
# spreads_file_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Filled_Merged_Spreads_Report.xlsx"
spreads_file_path = os.path.join(cwd,'Intermediate_results/Filled_Merged_Spreads_Report.xlsx')
# input_controls_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Input_controls.xlsx"
input_controls_path = os.path.join(cwd,'Input/Input_controls.xlsx')

# Load the data into a pandas DataFrame
df_spreads = pd.read_excel(spreads_file_path)

# Load the mappings and fund sector data
df_mappings = pd.read_excel(input_controls_path, sheet_name='Mappings')
df_fund_sector = pd.read_excel(input_controls_path, sheet_name='Fund_Sector')

# Extract unique sectors from the Fund_Sector sheet
unique_sectors = df_fund_sector['Sector'].unique()

# Filter the mappings to include only instruments in the specified sectors
df_filtered_mappings = df_mappings[df_mappings['Sector'].isin(unique_sectors)]

# Extract the instrument names
instruments_to_keep = df_filtered_mappings['Instrument'].tolist()

# Keep only the relevant columns from the spreads DataFrame
columns_to_keep = ['Date'] + instruments_to_keep  # Include 'Date' if it is needed
df_filtered_spreads = df_spreads[columns_to_keep]

# Save the filtered DataFrame to a new Excel file
# output_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Filtered_Merged_Spreads_Report.xlsx"
output_path = os.path.join(cwd,'Intermediate_results/Filtered_Merged_Spreads_Report.xlsx')
df_filtered_spreads.to_excel(output_path, index=False)

print(f"Filtered DataFrame saved at {output_path}")
