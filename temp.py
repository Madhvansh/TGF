import pandas as pd
import numpy as np

# Read the Excel file
file_path = 'DCM.xlsx'

# Read makeup water data
makeup_df = pd.read_excel(file_path, sheet_name='MakeUp', skiprows=3)

# Read cooling tower data
ct_df = pd.read_excel(file_path, sheet_name='OLD CT PH Process', skiprows=3)

# Clean makeup water data
# Remove summary rows at the end
makeup_df = makeup_df.iloc[:-5]

# Replace non-numeric values with 0
makeup_df = makeup_df.replace(['NIL', '-', 'STOP', 'shut down', ''], 0)

# Fill remaining NaN values with 0
makeup_df = makeup_df.fillna(0)

# Clean cooling tower data
# Remove summary rows at the end
ct_df = ct_df.iloc[:-7]

# Replace non-numeric values with 0
ct_df = ct_df.replace(['NIL', '-', 'STOP', 'Shut Down', ''], 0)

# Fill remaining NaN values with 0
ct_df = ct_df.fillna(0)

# Save to CSV files
makeup_df.to_csv('makeup_water_parameters.csv', index=False)
ct_df.to_csv('cooling_tower_parameters.csv', index=False)

print("CSV files created successfully!")
print(f"Makeup water data shape: {makeup_df.shape}")
print(f"Cooling tower data shape: {ct_df.shape}")