import pandas as pd
import os
import matplotlib.pyplot as plt

# Folder where NHS monthly Excel files are stored
folder_path = "data"

months = [
    "April-2024.xlsx", "May-2024.xlsx", "June-2024.xlsx", "July-2024.xlsx",
    "August-2024.xlsx", "September-2024.xlsx", "October-2024.xlsx", "November-2024.xlsx",
    "December-2024.xlsx", "January-2025.xlsx", "February-2025.xlsx", "March-2025.xlsx"
]

all_data = []

# Read and tag each file with month
for file in months:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    month_name = file.replace(".xlsx", "")
    df["Month"] = month_name
    all_data.append(df)

# Merge all data
merged_df = pd.concat(all_data, ignore_index=True)
merged_df.columns = merged_df.columns.str.strip()

# Ensure numeric values
merged_df["Average (median) waiting time (in weeks)"] = pd.to_numeric(
    merged_df["Average (median) waiting time (in weeks)"], errors="coerce"
)

# Save merged CSV
merged_df.to_csv("data/NHS_Trusts_Merged_2024_2025.csv", index=False)

# Summary outputs
print("Rows, Columns:", merged_df.shape)
print("Unique Providers:", merged_df['Provider Name'].nunique())
print("Unique Regions:", merged_df['Region Code'].nunique())

# Monthly trend
monthly_avg = merged_df.groupby("Month")["Average (median) waiting time (in weeks)"].mean().reset_index()
monthly_avg["Month"] = pd.to_datetime(monthly_avg["Month"], format="%B-%Y")
monthly_avg = monthly_avg.sort_values("Month")

# Grouped by Month & Region
region_month_avg = (
    merged_df.groupby(["Month", "Region Code"])["Average (median) waiting time (in weeks)"]
    .mean()
    .reset_index()
)
region_month_avg["Month"] = pd.to_datetime(region_month_avg["Month"], format="%B-%Y")
region_month_avg = region_month_avg.sort_values(["Region Code", "Month"])

# Save region-wise monthly average CSV
region_month_avg.to_csv("data/Region_Monthly_Average_Waiting.csv", index=False)

# Plot monthly overall trend
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg["Month"], monthly_avg["Average (median) waiting time (in weeks)"], marker='o')
plt.title("Overall NHS Average Waiting Time by Month")
plt.xlabel("Month")
plt.ylabel("Waiting Time (Weeks)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
