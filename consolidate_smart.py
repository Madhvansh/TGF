import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import re
warnings.filterwarnings('ignore')

# Define the standardized column set
STANDARD_COLUMNS = [
    'pH',
    'Turbidity_NTU',
    'Free_Residual_Chlorine_ppm',
    'TDS_ppm',
    'Total_Hardness_ppm',
    'Calcium_Hardness_ppm',
    'Magnesium_Hardness_ppm',
    'Chlorides_ppm',
    'Phosphate_ppm',
    'Total_Alkalinity_ppm',
    'Sulphates_ppm',
    'Silica_ppm',
    'Source_Sheet',
    'Date',
    'Iron_ppm',
    'Suspended_Solids_ppm',
    'Conductivity_uS_cm',
    'Cycles_of_Concentration'
]

# Column mapping - maps various column names to standard names
COLUMN_MAPPER = {
    # Date variations
    'DATE': 'Date',
    'Date': 'Date',
    'date': 'Date',
    'DATE ': 'Date',
    
    # pH variations
    'pH': 'pH',
    'ph': 'pH',
    'PH': 'pH',
    'Ph': 'pH',
    
    # Turbidity variations
    'Turbidity': 'Turbidity_NTU',
    'turbidity': 'Turbidity_NTU',
    'TURBIDITY': 'Turbidity_NTU',
    'Turbidity (NTU)': 'Turbidity_NTU',
    
    # FRC variations
    'FRC': 'Free_Residual_Chlorine_ppm',
    'Free Residual Chlorine': 'Free_Residual_Chlorine_ppm',
    'Free Chlorine': 'Free_Residual_Chlorine_ppm',
    
    # TDS variations
    'TDS': 'TDS_ppm',
    'tds': 'TDS_ppm',
    'Total Dissolved Solids': 'TDS_ppm',
    'TDS (ppm)': 'TDS_ppm',
    
    # Total Hardness variations
    'TH': 'Total_Hardness_ppm',
    'Total Hardness': 'Total_Hardness_ppm',
    'TOTAL HARDNESS': 'Total_Hardness_ppm',
    'T.H.': 'Total_Hardness_ppm',
    
    # Calcium Hardness variations
    'CaH': 'Calcium_Hardness_ppm',
    'Calcium Hardness': 'Calcium_Hardness_ppm',
    'Ca Hardness': 'Calcium_Hardness_ppm',
    'Ca H': 'Calcium_Hardness_ppm',
    
    # Magnesium Hardness variations
    'MgH': 'Magnesium_Hardness_ppm',
    'Magnesium Hardness': 'Magnesium_Hardness_ppm',
    'Mg Hardness': 'Magnesium_Hardness_ppm',
    'Mg H': 'Magnesium_Hardness_ppm',
    
    # Chloride variations
    'Cl': 'Chlorides_ppm',
    'Chloride': 'Chlorides_ppm',
    'CHLORIDE': 'Chlorides_ppm',
    'Chlorides': 'Chlorides_ppm',
    
    # Phosphate variations  
    'ORTHO PO4': 'Phosphate_ppm',
    'Ortho PO4': 'Phosphate_ppm',
    'Phosphate': 'Phosphate_ppm',
    'PO4': 'Phosphate_ppm',
    'Total PO4': 'Phosphate_ppm',
    'ORTHO-PO4': 'Phosphate_ppm',
    
    # Total Alkalinity variations
    'T. Alk.': 'Total_Alkalinity_ppm',
    'Total alk': 'Total_Alkalinity_ppm',
    'Total Alk': 'Total_Alkalinity_ppm',
    'Total Alkalinity': 'Total_Alkalinity_ppm',
    'T Alk': 'Total_Alkalinity_ppm',
    'T.Alk.': 'Total_Alkalinity_ppm',
    
    # Sulphate variations
    'Sulphate': 'Sulphates_ppm',
    'SULFATE': 'Sulphates_ppm',
    'Sulphates': 'Sulphates_ppm',
    'SO4': 'Sulphates_ppm',
    'Sulfate': 'Sulphates_ppm',
    
    # Silica variations
    'SiO2': 'Silica_ppm',
    'Silica': 'Silica_ppm',
    'SILICA': 'Silica_ppm',
    'SIO2': 'Silica_ppm',
    
    # Iron variations
    'Total Iron': 'Iron_ppm',
    'Iron': 'Iron_ppm',
    'Fe': 'Iron_ppm',
    'TOTAL IRON': 'Iron_ppm',
    
    # Suspended Solids variations
    'SS': 'Suspended_Solids_ppm',
    'TSS': 'Suspended_Solids_ppm',
    'Suspended Solids': 'Suspended_Solids_ppm',
    'Total Suspended Solids': 'Suspended_Solids_ppm',
    'S.S.': 'Suspended_Solids_ppm',
    
    # Conductivity variations
    'COND': 'Conductivity_uS_cm',
    'Conductivity': 'Conductivity_uS_cm',
    'CONDUCTIVITY': 'Conductivity_uS_cm',
    'Cond': 'Conductivity_uS_cm',
    
    # Cycles of Concentration variations
    'COC': 'Cycles_of_Concentration',
    'Cycles': 'Cycles_of_Concentration',
    'Cycles of Concentration': 'Cycles_of_Concentration',
    'C.O.C.': 'Cycles_of_Concentration',
    
    # P. Alk (not used but present in data)
    'P. Alk.': None,
    'P Alk': None,
}

def clean_value(val):
    """Clean individual cell values"""
    if pd.isna(val):
        return np.nan
    
    # Convert to string for processing
    val_str = str(val).strip()
    
    # Handle common non-numeric values
    if val_str.upper() in ['NIL', '-', 'NA', 'N/A', 'NAN', '']:
        return np.nan
    
    # Try to extract numeric value
    try:
        # Remove any non-numeric characters except decimal point and minus
        cleaned = re.sub(r'[^\d\.\-]', '', val_str)
        if cleaned and cleaned != '-':
            return float(cleaned)
        else:
            return np.nan
    except:
        return np.nan

def load_and_map_file(csv_file, column_mapper, standard_columns):
    """Load a CSV file, map columns to standard names, and clean data"""
    try:
        # Read the file without any header processing first
        df_raw = pd.read_csv(csv_file, header=None, low_memory=False)
        
        if len(df_raw) < 3:
            return None, 0, "File too short (< 3 rows)"
        
        # Find which row contains the column names
        # Look for rows containing "DATE" or "pH" (case insensitive)
        header_row_idx = None
        for idx in range(min(5, len(df_raw))):  # Check first 5 rows
            row_values = df_raw.iloc[idx].astype(str).str.upper()
            if 'DATE' in row_values.values or 'PH' in row_values.values:
                header_row_idx = idx
                break
        
        if header_row_idx is None:
            return None, 0, "Could not find column header row"
        
        # Extract column names from the identified row
        column_names = df_raw.iloc[header_row_idx].tolist()
        
        # Data starts 2 rows after column names (skip units row)
        data_start_row = header_row_idx + 2
        
        if data_start_row >= len(df_raw):
            return None, 0, "No data rows after headers"
        
        df = df_raw.iloc[data_start_row:].copy()
        df.columns = column_names
        df = df.reset_index(drop=True)
        
        if len(df) == 0:
            return None, 0, "No data rows after header/units removal"
        
        # Clean column names - convert NaN to empty string and strip whitespace
        df.columns = [str(c).strip() if pd.notna(c) else '' for c in df.columns]
        
        # Remove columns with empty names (were NaN)
        df = df[[c for c in df.columns if c != '' and c != 'nan']]
        
        if len(df.columns) == 0:
            return None, 0, "No valid columns found"
        
        # Map columns to standard names
        mapped_df = pd.DataFrame()
        
        for standard_col in standard_columns:
            if standard_col == 'Source_Sheet':
                # Add source file name
                mapped_df['Source_Sheet'] = csv_file.stem
                continue
            
            # Find matching column in original data
            found = False
            for orig_col in df.columns:
                if orig_col in column_mapper:
                    # Skip columns mapped to None
                    if column_mapper[orig_col] is None:
                        continue
                    if column_mapper[orig_col] == standard_col:
                        mapped_df[standard_col] = df[orig_col].apply(clean_value)
                        found = True
                        break
            
            if not found:
                # Column doesn't exist - fill with NaN
                mapped_df[standard_col] = np.nan
        
        # Additional date parsing
        if 'Date' in mapped_df.columns:
            mapped_df['Date'] = pd.to_datetime(mapped_df['Date'], errors='coerce')
        
        # Remove completely empty rows
        data_cols = [c for c in mapped_df.columns if c not in ['Date', 'Source_Sheet']]
        valid_rows = mapped_df[data_cols].notna().any(axis=1)
        mapped_df = mapped_df[valid_rows].reset_index(drop=True)
        
        return mapped_df, len(mapped_df), None
        
    except Exception as e:
        return None, 0, str(e)

def find_csv_files(directory):
    """Find all CSV files in the directory"""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    csv_files = list(path.glob('*.csv'))
    csv_files = [f for f in csv_files if 'consolidated' not in f.name.lower() 
                 and 'sample' not in f.name.lower()]
    return sorted(csv_files)

def consolidate_files(csv_files, column_mapper, standard_columns):
    """Consolidate all CSV files"""
    print("\n" + "=" * 100)
    print("CONSOLIDATING FILES WITH SMART COLUMN MAPPING")
    print("=" * 100)
    
    all_data = []
    success_count = 0
    error_count = 0
    total_rows = 0
    errors_list = []
    
    # For debugging - track which columns we're finding
    found_columns = set()
    
    for i, csv_file in enumerate(csv_files, 1):
        mapped_df, rows, error = load_and_map_file(csv_file, column_mapper, standard_columns)
        
        if error:
            print(f"⚠️  [{i}/{len(csv_files)}] {csv_file.name}: {error}")
            errors_list.append((csv_file.name, error))
            error_count += 1
        elif mapped_df is None or rows == 0:
            print(f"⚠️  [{i}/{len(csv_files)}] {csv_file.name}: No valid data rows")
            error_count += 1
        else:
            all_data.append(mapped_df)
            total_rows += rows
            success_count += 1
            
            # Track columns found for debugging
            found_columns.update([col for col in mapped_df.columns if mapped_df[col].notna().any()])
            
            if i % 10 == 0 or i == len(csv_files):
                print(f"✓ [{i}/{len(csv_files)}] {csv_file.name}: {rows} rows")
    
    print(f"\n" + "=" * 100)
    print(f"CONSOLIDATION SUMMARY")
    print(f"=" * 100)
    print(f"✓ Successfully processed: {success_count}/{len(csv_files)} files")
    print(f"❌ Errors: {error_count}/{len(csv_files)} files")
    print(f"📊 Total data rows: {total_rows}")
    
    if success_count > 0:
        print(f"\n✓ Columns with data found: {len(found_columns)}")
        print(f"  {', '.join(sorted(found_columns))}")
    
    if errors_list and len(errors_list) <= 10:
        print(f"\n⚠️  Files with errors:")
        for fname, err in errors_list:
            print(f"  - {fname}: {err}")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Final consolidated dataset: {final_df.shape[0]} rows × {final_df.shape[1]} columns")
        return final_df
    else:
        return None

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    print("\n" + "=" * 100)
    print("MISSING DATA ANALYSIS (NaN Masking)")
    print("=" * 100)
    
    total_rows = len(df)
    analysis_cols = [c for c in df.columns if c not in ['Date', 'Source_Sheet']]
    
    print(f"\nTotal rows: {total_rows}")
    print(f"\nMissing data per parameter:")
    
    missing_stats = []
    for col in analysis_cols:
        present_count = df[col].notna().sum()
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        missing_stats.append({
            'Parameter': col,
            'Present': present_count,
            'Missing': missing_count,
            'Missing %': missing_pct
        })
    
    missing_df = pd.DataFrame(missing_stats).sort_values('Missing %', ascending=False)
    
    for _, row in missing_df.iterrows():
        bar_length = int(row['Missing %'] / 2) if pd.notna(row['Missing %']) else 0
        bar = '█' * bar_length
        print(f"  {row['Parameter']:35s} | Present: {row['Present']:5d} | Missing: {row['Missing']:5d} ({row['Missing %']:5.1f}%) {bar}")
    
    # Completeness statistics
    complete_rows = df[analysis_cols].notna().all(axis=1).sum()
    partial_rows = total_rows - complete_rows
    
    print(f"\n✓ Rows with ALL parameters: {complete_rows} ({complete_rows/total_rows*100:.1f}%)")
    print(f"⚠️  Rows with SOME missing: {partial_rows} ({partial_rows/total_rows*100:.1f}%)")

def show_preview(df):
    """Show data preview"""
    print("\n" + "=" * 100)
    print("DATA PREVIEW")
    print("=" * 100)
    
    if len(df) > 0:
        print("\nFirst 5 rows:")
        print(df.head(5)[['Date', 'Source_Sheet', 'pH', 'TDS_ppm', 'Total_Hardness_ppm', 'Conductivity_uS_cm']].to_string())
        
        # Show a row with good data coverage
        non_null_counts = df.drop(['Date', 'Source_Sheet'], axis=1, errors='ignore').notna().sum(axis=1)
        if len(non_null_counts) > 0 and non_null_counts.max() > 0:
            best_idx = non_null_counts.idxmax()
            print(f"\nBest coverage row (index {best_idx}, {non_null_counts[best_idx]} parameters):")
            print(df.iloc[[best_idx]].T.to_string())
    else:
        print("⚠️  No data to preview")

def save_dataset(df, output_dir):
    """Save the consolidated dataset"""
    print("\n" + "=" * 100)
    print("SAVING DATASET")
    print("=" * 100)
    
    output_path = Path(output_dir) / 'consolidated_dataset_FINAL.csv'
    
    try:
        df.to_csv(output_path, index=False)
        file_size = output_path.stat().st_size / 1024 / 1024
        print(f"✓ Dataset saved to: {output_path}")
        print(f"✓ File size: {file_size:.2f} MB")
        print(f"✓ Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return True
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def main():
    print("\n" + "=" * 100)
    print("COOLING TOWER DATA CONSOLIDATION")
    print("Smart Column Mapping + NaN Masking for MOMENT/PatchTST/RRCF")
    print("=" * 100)
    
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        print("\n❌ Please provide directory path")
        print("\nUsage: python consolidate_smart.py <path_to_csv_directory>")
        return None
    
    print(f"\n📁 Directory: {data_directory}")
    
    # Find files
    try:
        csv_files = find_csv_files(data_directory)
        print(f"✓ Found {len(csv_files)} CSV files")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    if not csv_files:
        print("❌ No CSV files found!")
        return None
    
    # Consolidate
    consolidated_df = consolidate_files(csv_files, COLUMN_MAPPER, STANDARD_COLUMNS)
    
    if consolidated_df is None or len(consolidated_df) == 0:
        print("❌ No data extracted!")
        return None
    
    # Analyze
    analyze_missing_data(consolidated_df)
    
    # Preview
    show_preview(consolidated_df)
    
    # Save
    if save_dataset(consolidated_df, data_directory):
        print("\n" + "=" * 100)
        print("✓✓✓ CONSOLIDATION COMPLETE! ✓✓✓")
        print("=" * 100)
        print(f"✓ {consolidated_df.shape[0]} rows with NaN masking")
        print(f"✓ Ready for MOMENT, PatchTST, RRCF training")
        print("=" * 100)
    
    return consolidated_df

if __name__ == "__main__":
    df = main()