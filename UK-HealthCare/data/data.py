import pandas as pd
import os

# 1. Set your data folder
data_folder = 'data'

# 2. List all Excel files in the folder
excel_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

# 3. Month map to convert 'Jun24' → '2024-06'
month_map = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
}

merged_data = []

for file in excel_files:
    file_path = os.path.join(data_folder, file)

    # Example: "Incomplete-Provider-Jun24-XLSX-8M-revised.xlsx"
    parts = file.split('-')
    month_year_part = parts[2].strip().lower()  # 'jun24'

    month_abbr = month_year_part[:3]
    year_suffix = month_year_part[3:]

    # Convert '24' to '2024', '25' to '2025'
    year_full = '20' + year_suffix
    month_number = month_map[month_abbr]

    reporting_period = f"{year_full}-{month_number}"

    # Read Excel file
    df = pd.read_excel(file_path)

    # Add reporting period column
    df['Reporting_Period'] = reporting_period

    # Append to list
    merged_data.append(df)

# 4. Concatenate all into one DataFrame
final_df = pd.concat(merged_data, ignore_index=True)

# 5. Save the final merged file
output_file = 'incomplete_pathways_2024_2025.xlsx'
final_df.to_excel(output_file, index=False)

print(f"✅ Merged dataset saved as {output_file}")
