import pandas as pd
import os

cwd = os.getcwd()
# mapped_spreads_stress_loss_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Mapped_Spreads_And_Stress_Loss_Report.xlsx"
mapped_spreads_stress_loss_path = os.path.join(cwd, 'Intermediate_results/Mapped_Spreads_And_Stress_Loss_Report.xlsx')
# input_controls_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Input_controls.xlsx"
input_controls_path = os.path.join(cwd, 'Input/Input_controls.xlsx')

# Load the mappings and fund sector data
df_mappings = pd.read_excel(input_controls_path, sheet_name='Mappings')
df_fund_sector = pd.read_excel(input_controls_path, sheet_name='Fund_Sector')

# Extract unique sectors from the Fund_Sector sheet
unique_sectors = df_fund_sector['Sector'].unique()

# Filter the mappings to include only instruments in the specified sectors
df_filtered_mappings = df_mappings[df_mappings['Sector'].isin(unique_sectors)]

# Extract the instrument names
instruments_to_keep = df_filtered_mappings['Instrument'].tolist()

# Load the Mapped Spreads and Stress Loss Report data
df_spreads_stress_loss = pd.read_excel(mapped_spreads_stress_loss_path)

# Filter the data to keep only the instruments from the filtered mappings
df_filtered_spreads_stress_loss = df_spreads_stress_loss[df_spreads_stress_loss['Instrument'].isin(instruments_to_keep)]

# Save the filtered data back to a new Excel file or overwrite the existing file
# output_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Filtered_Mapped_Spreads_And_Stress_Loss_Report.xlsx"
output_path = os.path.join(cwd, 'Intermediate_results/Filtered_Mapped_Spreads_And_Stress_Loss_Report.xlsx')
df_filtered_spreads_stress_loss.to_excel(output_path, index=False)

print("Filtered data has been saved to", output_path)