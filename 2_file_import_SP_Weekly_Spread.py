import os
import shutil
from datetime import datetime

def get_latest_file(folder_path, base_name):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter files that match the base name pattern
    matching_files = [f for f in files if f.startswith(base_name) and f.endswith('.xlsx')]
    
    if not matching_files:
        raise FileNotFoundError(f"No files found with base name: {base_name}")

    # Extract dates from filenames and find the latest one
    latest_file = None
    latest_date = None
    date_format = "%m.%d.%y"
    
    for file in matching_files:
        try:
            # Extract the date part from the filename
            date_str = file[len(base_name)+1:-5]  # Removing base name and extension
            file_date = datetime.strptime(date_str, date_format)
            
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_file = file
        except ValueError:
            # Skip files that do not match the date format
            continue

    if latest_file is None:
        raise FileNotFoundError(f"No files with valid dates found for base name: {base_name}")

    return os.path.join(folder_path, latest_file)

def copy_latest_file(source_folder, base_name, destination_folder):
    try:
        latest_file_path = get_latest_file(source_folder, base_name)
        destination_path = os.path.join(destination_folder, os.path.basename(latest_file_path))
        
        # Copy the file to the destination folder
        shutil.copy(latest_file_path, destination_path)
        print(f"Copied {latest_file_path} to {destination_path}")
    except Exception as e:
        print(f"Error: {e}")

cwd = os.getcwd()
source_folder = r"//aocaazurefiles.file.core.windows.net/assetmgt/PM - Structured Credit/Relative Value/Historical Spreads - BofA"
base_name = 'SP spread time series'
# destination_folder = r"C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Portfolio Construction/Dry Run 8/Input"
destination_folder = os.path.join(cwd,'Input')
# os.mkdir(os.path.join(cwd,'Output'))
# os.mkdir(os.path.join(cwd,'Intermediate_results'))
copy_latest_file(source_folder, base_name, destination_folder)
