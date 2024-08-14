import pandas as pd
import numpy as np
import os


cwd = os.getcwd()
# Load the Excel files
# spreads_file_path = 'C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 1/Merged_Spreads_Report.xlsx'
# spreads_file_path = 'C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 2/Filled_Merged_Spreads_Report.xlsx'
# spreads_file_path = 'C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Filtered_Merged_Spreads_Report.xlsx'
spreads_file_path = os.path.join(cwd, 'Intermediate_results/Filtered_Merged_Spreads_Report.xlsx')
# spread_duration_file_path = 'C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Spread_duration.xlsx'
spread_duration_file_path = os.path.join(cwd,'Input/Spread_duration.xlsx')

df_spreads = pd.read_excel(spreads_file_path)
df_duration = pd.read_excel(spread_duration_file_path)

# Ensure 'Date' column is in datetime format
df_spreads['Date'] = pd.to_datetime(df_spreads['Date'])

# Define the function to calculate the percentile
def calculate_percentile(series, value):
    return np.sum(series < value) / len(series) * 100

# Extract the necessary data for each calculation
avg_2022_H2 = df_spreads[(df_spreads['Date'].dt.year == 2022) & (df_spreads['Date'].dt.month > 6)].mean()
avg_2021 = df_spreads[df_spreads['Date'].dt.year == 2021].mean()

# Get the latest spreads
latest_spreads = df_spreads.iloc[0]

# Calculate the target spreads
target_spreads = pd.DataFrame({
    'Instrument': df_spreads.columns[1:],  #  the first column is 'Date'
    'Unchanged': latest_spreads[1:],  # Exclude 'Date'
    '10% to 21': latest_spreads[1:] + 0.1 * (avg_2021[1:] - latest_spreads[1:]),
    '20% to 21': latest_spreads[1:] + 0.2 * (avg_2021[1:] - latest_spreads[1:]),
    '50% to 21': latest_spreads[1:] + 0.5 * (avg_2021[1:] - latest_spreads[1:]),
    '10% to 22': latest_spreads[1:] + 0.1 * (avg_2022_H2[1:] - latest_spreads[1:]),
    '20% to 22': latest_spreads[1:] + 0.2 * (avg_2022_H2[1:] - latest_spreads[1:]),
    '50% to 22': latest_spreads[1:] + 0.5 * (avg_2022_H2[1:] - latest_spreads[1:])
})

# Calculate the stress loss scenarios
stress_loss_scenarios = pd.DataFrame({
    'Instrument': df_spreads.columns[1:],  # the first column is 'Date'
    '25% to 22': latest_spreads[1:] + 0.25 * (avg_2022_H2[1:] - latest_spreads[1:]),
    '50% to 22': latest_spreads[1:] + 0.5 * (avg_2022_H2[1:] - latest_spreads[1:]),
    '100% to 22': avg_2022_H2[1:]
})

# Create the result DataFrame
results = []

for instrument in df_spreads.columns[1:]:  # Exclude 'Date'
    latest_spread = latest_spreads[instrument]
    expected_level = target_spreads.loc[target_spreads['Instrument'] == instrument, '50% to 21'].values[0]  # Example target
    spread_duration = df_duration.loc[df_duration['Instrument'] == instrument, 'Spread_duration'].values[0]
    carry = latest_spread / 100
    px_appr = - spread_duration * (expected_level - latest_spread) / 100
    total_return = carry + px_appr
    stress_spread = stress_loss_scenarios.loc[stress_loss_scenarios['Instrument'] == instrument, '50% to 22'].values[0]  # Example stress scenario
    stress_loss = spread_duration * (expected_level - stress_spread) / 100

    results.append({
        'Instrument': instrument,
        'Spread': latest_spread,
        'Expected level in 1yr': expected_level,
        'Spread duration': spread_duration,
        'Carry': carry,
        'Px appr': px_appr,
        'Total return': total_return,
        'Stress spread': stress_spread,
        'Stress loss': stress_loss
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save the results to an Excel file
results_df.to_excel(os.path.join(cwd,'Intermediate_results/Calculated_Spreads_And_Stress_Loss_Report.xlsx'), index=False)

print(results_df)
