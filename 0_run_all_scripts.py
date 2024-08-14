import subprocess
import time
start_time = time.time()


def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(result.stdout)

if __name__ == "__main__":
    run_script('1_file_import_NA_RMBS_Weekly_Spread.py')
    run_script('2_file_import_SP_Weekly_Spread.py')
    run_script('3_Data_Cleaning_NA_RMBS_Weekly_Spread.py')
    run_script('4_Data_Cleaning_SP_Weekly_Spread.py')
    run_script('5_Merged_Spreads_Report.py')
    run_script('6_Interpolation_merged_dataset.py')
    run_script('7_Interpolation_merged_dataset_Filtered.py')
    run_script('8_Estimated_Spread_Stress_Spread_Scenario.py')
    run_script('9_Stress_loss_Spreads.py')
    run_script('10_Correlation_Spreads.py')
    run_script('11_Mapping.py')
    run_script('12_Filtered_Mapped_Spreads_and_Stress_Loss_Instruments.py')
    run_script('13_Portfolio_construction.py')

end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time} minutes")