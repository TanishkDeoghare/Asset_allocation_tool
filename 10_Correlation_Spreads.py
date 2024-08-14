import pandas as pd
import os 

cwd = os.getcwd()
# Load your data into a pandas DataFrame
# file_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Filtered_Merged_Spreads_Report.xlsx"
file_path = os.path.join(cwd, 'Intermediate_results/Filtered_Merged_Spreads_Report.xlsx')
df = pd.read_excel(file_path)

# Exclude the date column (assuming it's named 'Date')
df_no_date = df.drop(columns=['Date'])

# Form the correlation matrix
correlation_matrix = df_no_date.corr()

# Save the correlation matrix to a new Excel file

# correlation_output_path = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 4/Correlation_Matrix_Filtered.xlsx"
correlation_output_path = os.path.join(cwd,'Intermediate_results/Correlation_Matrix_Filtered.xlsx')
correlation_matrix.to_excel(correlation_output_path)

print(f"Correlation matrix saved at {correlation_output_path}")
