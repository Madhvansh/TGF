"""
Cooling Tower Parameter Data Extraction and Cleaning Pipeline
==============================================================

This script automatically:
1. Scans all Excel files in a folder
2. Identifies cooling tower parameter sheets using heuristic scoring
3. Cleans and standardizes the data
4. Concatenates all sheets into a unified dataset
5. Exports as CSV for anomaly detection

Author: Production-Ready ML Pipeline
Version: 1.0
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import openpyxl

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cooling tower parameter keywords for scoring
CTP_KEYWORDS = {
    'primary': ['ph', 'tds', 'conductivity', 'ec', 'chloride', 'chlorides'],
    'secondary': ['hardness', 'alkalinity', 'turbidity', 'calcium', 'magnesium',
                  'silica', 'phosphate', 'sulfate', 'sulphate', 'iron', 'frc'],
    'date_indicators': ['date', 'day', 'time', 'datetime']
}

# Keywords that indicate non-CTP sheets (penalize these)
EXCLUSION_KEYWORDS = [
    'report', 'summary', 'chart', 'graph', 'makeup', 'make up', 'make-up',
    'analysis', 'total', 'average', 'monthly', 'yearly', 'annual',
    'specification', 'spec', 'dashboard', 'index', 'contents', 'overview'
]

# Valid parameter ranges for validation
VALID_RANGES = {
    'ph': (0, 14),
    'tds': (0, 10000),
    'conductivity': (0, 10000),
    'turbidity': (0, 1000),
    'temperature': (0, 100),
    'chloride': (0, 2000),
    'hardness': (0, 2000)
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_text(text: any) -> str:
    """Clean and normalize text for comparison."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def standardize_column_name(col: str) -> Optional[str]:
    """
    Standardize column names to consistent format.
    
    Args:
        col: Original column name
        
    Returns:
        Standardized column name or None if column should be excluded
    """
    col_lower = clean_text(col)
    
    # Date columns
    if any(kw in col_lower for kw in ['date', 'day', 'time']):
        return 'Date'
    
    # pH
    elif col_lower in ['ph', 'ph_', 'p.h.', 'p.h']:
        return 'pH'
    
    # Turbidity
    elif 'turb' in col_lower:
        return 'Turbidity'
    
    # TSS (Total Suspended Solids)
    elif 'tss' in col_lower:
        return 'TSS'
    
    # FRC (Free Residual Chlorine)
    elif 'frc' in col_lower:
        return 'FRC'
    
    # Conductivity
    elif ('cond' in col_lower or 'ec' in col_lower) and 'ppm' not in col_lower:
        return 'Conductivity'
    
    # TDS (Total Dissolved Solids)
    elif 'tds' in col_lower or 't.d.s' in col_lower or 't_d_s' in col_lower:
        return 'TDS'
    
    # Total Hardness
    elif col_lower in ['th', 't_h', 't.h.', 'total_hardness', 'total(th)', 'total hardness']:
        return 'Total_Hardness'
    
    # Calcium Hardness
    elif col_lower in ['cah', 'ca_h', 'ca.h.', 'cah_', 'calcium_hardness', 
                       'calcium(cah)', 'ca', 'calcium', 'calcium hardness']:
        return 'Calcium_Hardness'
    
    # Magnesium Hardness
    elif col_lower in ['mgh', 'mg_h', 'mg.h.', 'magnesium_hardness', 
                       'magnesium', 'mg', 'magnesium hardness']:
        return 'Magnesium_Hardness'
    
    # Chlorides
    elif col_lower in ['cl', 'chloride', 'chlorides']:
        return 'Chlorides'
    
    # Silica
    elif 'silica' in col_lower or 'sio2' in col_lower:
        return 'Silica'
    
    # Total Alkalinity
    elif any(kw in col_lower for kw in ['t_alk', 'total(talk)', 't.alk', 
                                         'total alkalinity', 'total_alkalinity']):
        return 'Total_Alkalinity'
    
    # P Alkalinity
    elif any(kw in col_lower for kw in ['p_alk', 'p-alk', 'p-alkalinity', 
                                         'p.alk', 'p alkalinity']):
        return 'P_Alkalinity'
    
    # Phosphate (various types)
    elif 'po4' in col_lower or 'phosphate' in col_lower:
        if 'ortho' in col_lower:
            return 'Ortho_PO4'
        elif 'total' in col_lower:
            return 'Total_PO4'
        else:
            return 'PO4'
    
    # Iron
    elif 'iron' in col_lower or col_lower == 'fe':
        return 'Total_Iron'
    
    # Sulfate/Sulphate
    elif 'sulphate' in col_lower or 'sulfate' in col_lower or 'so4' in col_lower:
        return 'Sulphate'
    
    # SS (Suspended Solids)
    elif col_lower == 'ss':
        return 'SS'
    
    # Temperature
    elif 'temp' in col_lower:
        return 'Temperature'
    
    # Tower identifier
    elif col_lower in ['tower', 'tower_id', 'tower id']:
        return 'Tower'
    
    else:
        # Unknown column - skip it
        return None


# ============================================================================
# SHEET SCORING FUNCTIONS
# ============================================================================

def score_sheet_header(header_row: pd.Series) -> float:
    """
    Score a potential header row based on cooling tower parameter keywords.
    
    Args:
        header_row: A pandas Series representing a potential header row
        
    Returns:
        Score indicating likelihood this is a CTP sheet (higher = more likely)
    """
    score = 0.0
    penalty = 0.0
    
    # Convert all values to lowercase strings for comparison
    row_text = ' '.join([clean_text(x) for x in header_row if not pd.isna(x)])
    
    # Score primary CTP keywords highly
    for keyword in CTP_KEYWORDS['primary']:
        if keyword in row_text:
            score += 10.0
    
    # Score secondary CTP keywords moderately
    for keyword in CTP_KEYWORDS['secondary']:
        if keyword in row_text:
            score += 5.0
    
    # Score date indicators
    for keyword in CTP_KEYWORDS['date_indicators']:
        if keyword in row_text:
            score += 3.0
    
    # Penalize exclusion keywords heavily
    for keyword in EXCLUSION_KEYWORDS:
        if keyword in row_text:
            penalty += 15.0
    
    # Bonus for having many non-null columns (CTP sheets have many parameters)
    non_null_count = header_row.notna().sum()
    if non_null_count >= 8:
        score += non_null_count * 0.5
    
    return max(0.0, score - penalty)


def find_header_row(df: pd.DataFrame) -> Optional[int]:
    """
    Find the most likely header row in a dataframe.
    
    Args:
        df: DataFrame to search
        
    Returns:
        Index of header row, or None if not found
    """
    best_score = 0.0
    best_row = None
    
    # Search first 20 rows for header
    search_limit = min(20, len(df))
    
    for i in range(search_limit):
        row = df.iloc[i]
        score = score_sheet_header(row)
        
        if score > best_score and score > 20:  # Minimum threshold
            best_score = score
            best_row = i
    
    return best_row


def score_sheet_data(df: pd.DataFrame, header_row_idx: int) -> float:
    """
    Score the data quality and characteristics of a sheet.
    
    Args:
        df: DataFrame to score
        header_row_idx: Index of the header row
        
    Returns:
        Data quality score
    """
    score = 0.0
    
    # Check if there's actual data after the header
    data_rows = len(df) - header_row_idx - 1
    if data_rows < 5:
        return 0.0
    
    # Score based on amount of data
    score += min(data_rows * 0.1, 50.0)
    
    # Check for numeric columns (CTP data is mostly numeric)
    data_df = df.iloc[header_row_idx + 1:, :]
    numeric_cols = 0
    for col in data_df.columns:
        try:
            pd.to_numeric(data_df[col], errors='coerce')
            numeric_ratio = data_df[col].notna().sum() / len(data_df)
            if numeric_ratio > 0.5:
                numeric_cols += 1
        except:
            pass
    
    score += numeric_cols * 2.0
    
    return score


def score_sheet(df: pd.DataFrame, sheet_name: str) -> Tuple[float, Optional[int]]:
    """
    Comprehensive scoring of a sheet to determine if it's a CTP sheet.
    
    Args:
        df: DataFrame to score
        sheet_name: Name of the sheet
        
    Returns:
        Tuple of (total_score, header_row_index)
    """
    if df is None or df.empty:
        return 0.0, None
    
    # Penalize based on sheet name
    sheet_penalty = 0.0
    sheet_lower = clean_text(sheet_name)
    for keyword in EXCLUSION_KEYWORDS:
        if keyword in sheet_lower:
            sheet_penalty += 20.0
    
    # Find header row
    header_row_idx = find_header_row(df)
    if header_row_idx is None:
        return 0.0, None
    
    # Score header
    header_score = score_sheet_header(df.iloc[header_row_idx])
    
    # Score data quality
    data_score = score_sheet_data(df, header_row_idx)
    
    # Calculate total score
    total_score = header_score + data_score - sheet_penalty
    
    return max(0.0, total_score), header_row_idx


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def parse_date_value(date_val: any) -> Optional[pd.Timestamp]:
    """
    Parse various date formats into pandas Timestamp.
    
    Args:
        date_val: Value to parse as date
        
    Returns:
        Parsed timestamp or None
    """
    if pd.isna(date_val):
        return None
    
    date_str = str(date_val).strip()
    
    # Try Excel serial date (number)
    try:
        num = float(date_str)
        if 40000 < num < 60000:  # Reasonable Excel date range
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=num)
    except:
        pass
    
    # Try common date formats
    date_formats = [
        '%d.%m.%y', '%d.%m.%Y',
        '%d/%m/%y', '%d/%m/%Y',
        '%Y-%m-%d', '%d-%m-%Y',
        '%m/%d/%y', '%m/%d/%Y',
        '%Y/%m/%d', '%d-%b-%Y',
        '%d %b %Y', '%d-%b-%y',
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # Try pandas auto-parser as last resort
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def clean_numeric_value(value: any) -> any:
    """
    Clean and convert numeric values, handling commas and other formats.
    
    Args:
        value: Value to clean
        
    Returns:
        Cleaned numeric value or NaN
    """
    if pd.isna(value):
        return np.nan
    
    # If already numeric, return as-is
    if isinstance(value, (int, float)):
        return value
    
    # Convert to string and clean
    str_val = str(value).strip()
    
    # Remove commas
    str_val = str_val.replace(',', '')
    
    # Remove any non-numeric characters except decimal point and minus
    str_val = re.sub(r'[^\d.-]', '', str_val)
    
    # Try to convert to float
    try:
        return float(str_val)
    except:
        return np.nan


def validate_parameter_value(param_name: str, value: float) -> bool:
    """
    Validate if a parameter value is within acceptable range.
    
    Args:
        param_name: Name of parameter
        value: Value to validate
        
    Returns:
        True if valid, False otherwise
    """
    if pd.isna(value):
        return True  # Missing values are okay
    
    param_lower = param_name.lower()
    
    # Check pH range
    if 'ph' in param_lower:
        return 0 <= value <= 14
    
    # Check for negative values in parameters that should be positive
    if any(kw in param_lower for kw in ['tds', 'conductivity', 'hardness', 
                                          'chloride', 'turbidity', 'alkalinity',
                                          'silica', 'iron', 'sulphate']):
        return value >= 0
    
    return True


def clean_dataframe(df: pd.DataFrame, header_row_idx: int) -> pd.DataFrame:
    """
    Clean and standardize a cooling tower parameter dataframe.
    
    Args:
        df: Raw dataframe
        header_row_idx: Index of header row
        
    Returns:
        Cleaned dataframe
    """
    # Extract header and data
    header = df.iloc[header_row_idx].tolist()
    
    # Skip unit row if present (next row after header often contains units)
    data_start_idx = header_row_idx + 1
    if data_start_idx < len(df):
        next_row = df.iloc[data_start_idx]
        next_text = ' '.join([clean_text(x) for x in next_row if not pd.isna(x)])
        if any(unit in next_text for unit in ['unit', 'ppm', 'ntu', 'us/cm', 
                                               'µs/cm', '--', 'units']):
            data_start_idx += 1
    
    # Extract data
    data = df.iloc[data_start_idx:].copy()
    
    # Standardize column names
    standardized_cols = []
    valid_col_indices = []
    
    for idx, col in enumerate(header):
        std_name = standardize_column_name(col)
        if std_name is not None:
            standardized_cols.append(std_name)
            valid_col_indices.append(idx)
    
    if len(standardized_cols) == 0:
        return pd.DataFrame()
    
    # Select only valid columns
    data = data.iloc[:, valid_col_indices].copy()
    data.columns = standardized_cols
    
    # Handle duplicate column names
    if len(standardized_cols) != len(set(standardized_cols)):
        cols = pd.Series(standardized_cols)
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index.values.tolist()
            cols.iloc[dup_indices] = [f"{dup}_{i}" if i != 0 else dup 
                                      for i in range(len(dup_indices))]
        data.columns = cols.tolist()
    
    # Remove completely empty rows
    data = data.dropna(how='all')
    
    if len(data) == 0:
        return pd.DataFrame()
    
    # Process Date column
    if 'Date' in data.columns:
        data['Date'] = data['Date'].apply(parse_date_value)
        # Remove rows where date parsing failed
        data = data[data['Date'].notna()]
    
    # Convert numeric columns
    numeric_cols = [col for col in data.columns if col != 'Date' and col != 'Tower']
    
    for col in numeric_cols:
        # Clean and convert to numeric
        data[col] = data[col].apply(clean_numeric_value)
        
        # Validate values
        valid_mask = data[col].apply(lambda x: validate_parameter_value(col, x))
        data.loc[~valid_mask, col] = np.nan
    
    # Remove duplicate rows
    data = data.drop_duplicates()
    
    # Reset index
    data = data.reset_index(drop=True)
    
    return data


def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate columns (e.g., PO4, PO4_1, PO4_2) by taking first non-null value.
    
    Args:
        df: DataFrame with potential duplicate columns
        
    Returns:
        DataFrame with merged columns
    """
    # Find duplicate column groups
    duplicate_groups = {}
    for col in df.columns:
        # Extract base column name (before underscore and number)
        base_col = re.sub(r'_\d+$', '', col)
        if base_col not in duplicate_groups:
            duplicate_groups[base_col] = []
        duplicate_groups[base_col].append(col)
    
    # Merge duplicate columns
    merged_data = {}
    for base_col, col_list in duplicate_groups.items():
        if len(col_list) == 1:
            merged_data[base_col] = df[col_list[0]]
        else:
            # Take first non-null value across duplicates
            merged_data[base_col] = df[col_list].bfill(axis=1).iloc[:, 0]
    
    return pd.DataFrame(merged_data)


def impute_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing categorical columns using mode.
    
    Args:
        df: DataFrame to impute
        
    Returns:
        DataFrame with imputed values
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Date']
    
    for col in categorical_cols:
        if df[col].notna().sum() > 0:  # Only impute if there's at least one value
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col].fillna(mode_value[0], inplace=True)
    
    return df


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_excel_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Process a single Excel file: identify CTP sheet and clean it.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        Cleaned DataFrame or None
    """
    print(f"\nProcessing: {os.path.basename(file_path)}")
    
    try:
        # Load all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"  Found {len(sheet_names)} sheets")
        
        # Score each sheet
        best_score = 0.0
        best_sheet_name = None
        best_header_idx = None
        best_df = None
        
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                score, header_idx = score_sheet(df, sheet_name)
                
                if score > best_score:
                    best_score = score
                    best_sheet_name = sheet_name
                    best_header_idx = header_idx
                    best_df = df
                    
            except Exception as e:
                print(f"    Warning: Could not read sheet '{sheet_name}': {e}")
                continue
        
        if best_sheet_name is None or best_score < 30:
            print(f"  ❌ No suitable CTP sheet found (best score: {best_score:.1f})")
            return None
        
        print(f"  ✓ Selected: '{best_sheet_name}' (score: {best_score:.1f})")
        
        # Clean the selected sheet
        cleaned_df = clean_dataframe(best_df, best_header_idx)
        
        if len(cleaned_df) == 0:
            print(f"  ❌ No valid data after cleaning")
            return None
        
        print(f"  ✓ Cleaned: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
        
        return cleaned_df
        
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        return None


def process_all_excel_files(folder_path: str) -> pd.DataFrame:
    """
    Process all Excel files in a folder and concatenate results.
    
    Args:
        folder_path: Path to folder containing Excel files
        
    Returns:
        Concatenated DataFrame
    """
    print("="*80)
    print("COOLING TOWER PARAMETER DATA EXTRACTION PIPELINE")
    print("="*80)
    
    # Find all Excel files
    folder = Path(folder_path)
    excel_files = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    
    print(f"\nFound {len(excel_files)} Excel files in: {folder_path}")
    
    if len(excel_files) == 0:
        print("❌ No Excel files found!")
        return pd.DataFrame()
    
    # Process each file
    all_dataframes = []
    
    for file_path in excel_files:
        df = process_excel_file(str(file_path))
        if df is not None and len(df) > 0:
            all_dataframes.append(df)
    
    if len(all_dataframes) == 0:
        print("\n❌ No valid data extracted from any file!")
        return pd.DataFrame()
    
    print(f"\n{'='*80}")
    print(f"Successfully processed {len(all_dataframes)} files")
    print(f"{'='*80}")
    
    # Concatenate all dataframes
    print("\nConcatenating all sheets...")
    df_combined = pd.concat(all_dataframes, axis=0, ignore_index=True, sort=False)
    
    print(f"  Combined shape: {df_combined.shape}")
    
    # Merge duplicate columns
    print("\nMerging duplicate columns...")
    df_final = merge_duplicate_columns(df_combined)
    
    print(f"  Final shape: {df_final.shape}")
    print(f"  Columns: {list(df_final.columns)}")
    
    # Remove duplicate rows
    initial_rows = len(df_final)
    df_final = df_final.drop_duplicates()
    duplicates_removed = initial_rows - len(df_final)
    print(f"\n  Removed {duplicates_removed} duplicate rows")
    
    # Sort by date if available
    if 'Date' in df_final.columns:
        df_final = df_final.sort_values('Date').reset_index(drop=True)
        print(f"  Sorted by date: {df_final['Date'].min()} to {df_final['Date'].max()}")
    
    # Impute categorical columns
    print("\nImputing missing categorical values...")
    df_final = impute_categorical_columns(df_final)
    
    return df_final


def generate_summary_report(df: pd.DataFrame):
    """
    Generate and print a summary report of the final dataset.
    
    Args:
        df: Final cleaned dataset
    """
    print("\n" + "="*80)
    print("FINAL DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal Records: {len(df):,}")
    
    if 'Date' in df.columns and df['Date'].notna().sum() > 0:
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        time_span = (df['Date'].max() - df['Date'].min()).days / 365.25
        print(f"Time Span: {time_span:.1f} years")
    
    print(f"\nParameters: {len(df.columns)}")
    
    print("\nColumn Coverage:")
    print(f"{'Column':<25} {'Non-Null':>10} {'Coverage':>10}")
    print("-"*50)
    
    for col in df.columns:
        non_null = df[col].notna().sum()
        coverage = (non_null / len(df)) * 100
        print(f"{col:<25} {non_null:>10,} {coverage:>9.1f}%")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    # Configuration
    INPUT_FOLDER = "./excel_files"  # Folder containing Excel files
    OUTPUT_FILE = "./cooling_tower_dataset_cleaned.csv"
    
    # Process all Excel files
    df_final = process_all_excel_files(INPUT_FOLDER)
    
    if df_final.empty:
        print("\n❌ No data to export. Exiting.")
        return
    
    # Generate summary report
    generate_summary_report(df_final)
    
    # Export to CSV
    print(f"\nExporting to: {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Successfully exported {len(df_final):,} rows to CSV")
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    # Display sample
    print("\nSample data (first 5 rows):")
    print(df_final.head().to_string())


if __name__ == '__main__':
    main()