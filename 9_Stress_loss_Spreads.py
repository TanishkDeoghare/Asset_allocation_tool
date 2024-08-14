import pandas as pd
import numpy as np
import os


cwd = os.getcwd()
# Load the Excel file
# file_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 1/Merged_Spreads_Report.xlsx"
# file_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 2/Filled_Merged_Spreads_Report.xlsx"
# file_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Filtered_Merged_Spreads_Report.xlsx"
file_path = os.path.join(cwd, 'Intermediate_results/Filtered_Merged_Spreads_Report.xlsx')
df = pd.read_excel(file_path)

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Define the function to calculate the percentile

def calculate_percentile(series, value):
    return np.sum(series < value) / len(series) * 100

# Extract the necessary data for each calculation
max_2020 = df[df['Date'].dt.year == 2020].max()
max_2022 = df[df['Date'].dt.year == 2022].max()
min_2021 = df[df['Date'].dt.year == 2021].min()
avg_2022_H2 = df[(df['Date'].dt.year == 2022) & (df['Date'].dt.month > 6)].mean()
avg_2021 = df[df['Date'].dt.year == 2021].mean()

# Get the latest spreads
latest_spreads = df.iloc[0]

# Calculate percentiles
percentiles = df.apply(lambda x: calculate_percentile(x, latest_spreads[x.name]))

# Create the resulting DataFrame
result = pd.DataFrame({
    'Instrument': df.columns[1:],  # The first column is 'Date'
    'Latest Spreads': latest_spreads[1:],  # Exclude 'Date'
    'Covid wides': max_2020[1:],  # Exclude 'Date'
    '2022 wides': max_2022[1:],  # Exclude 'Date'
    '2021 tights': min_2021[1:],  # Exclude 'Date'
    'Avg 2022 H2': avg_2022_H2[1:],  # Exclude 'Date'
    'Avg 2021': avg_2021[1:],  # Exclude 'Date'
    'Percentile': percentiles[1:]  # Exclude 'Date'
})

# Reset the index to have a column for instruments
result.reset_index(inplace=True, drop=True)

# Calculate the target spreads
targets = pd.DataFrame({
    'Instrument': result['Instrument'],
    'Unchanged': result['Latest Spreads'],
    '10% to 21': result['Latest Spreads'] + 0.1 * result['Avg 2021'],
    '20% to 21': result['Latest Spreads'] + 0.2 * result['Avg 2021'],
    '50% to 21': result['Latest Spreads'] + 0.5 * result['Avg 2021'],
    '10% to 22': result['Latest Spreads'] + 0.1 * result['Avg 2022 H2'],
    '20% to 22': result['Latest Spreads'] + 0.2 * result['Avg 2022 H2'],
    '50% to 22': result['Latest Spreads'] + 0.5 * result['Avg 2022 H2'],
    'Custom': result['Latest Spreads'] + 0.5 * result['Avg 2021']  # Placeholder for custom calculation
})

# Calculate the stress loss scenarios
stress_loss = pd.DataFrame({
    'Instrument': result['Instrument'],
    '25% to 22': result['Latest Spreads'] + 0.25 * result['Avg 2022 H2'],
    '50% to 22': result['Latest Spreads'] + 0.5 * result['Avg 2022 H2'],
    '100% to 22': result['Latest Spreads'] + result['Avg 2022 H2'],
    'Custom': result['Latest Spreads'] + 0.5 * result['Avg 2021'],  # Placeholder for custom calculation
    'Custom 2': result['Latest Spreads'] + 0.25*result['Avg 2022 H2']  # Placeholder for custom calculation
})

# Save the results to an Excel file
with pd.ExcelWriter(os.path.join(cwd,'Intermediate_results/Percentile_Targets_Stress_Loss_Spreads_Report.xlsx')) as writer:
    result.to_excel(writer, sheet_name='Summary', index=False)
    targets.to_excel(writer, sheet_name='Targets', index=False)
    stress_loss.to_excel(writer, sheet_name='Stress Loss', index=False)
