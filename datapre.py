"""
merge_and_clean_excels.py

Usage:
    - Put all your Excel files into INPUT_FOLDER (default: ./excel_files)
    - Run: python merge_and_clean_excels.py
Outputs:
    - merged_cleaned.csv         : full merged + cleaned dataset
    - merge_metadata.json        : metadata (column mapping, inferred dtypes, cleaning notes)
"""

import os
import glob
import json
from collections import Counter, defaultdict
from datetime import datetime
import re

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from dateutil import parser as dateparser
from sklearn.impute import SimpleImputer

# ---------------- User parameters ----------------
INPUT_FOLDER = "./excel_files"   # folder with Excel files
OUTPUT_CSV = "merged_cleaned.csv"
METADATA_JSON = "merge_metadata.json"

# Whether to attempt fuzzy mapping of column names to a canonical list.
# If CANONICAL_COLS is empty, script will just normalize names to lowercase_underscored.
CANONICAL_COLS = [
    # Example: "customer_id", "age", "gender", "purchase_amount", "purchase_date", "label"
    # If you have known target column names, list them here to enable fuzzy mapping.
]

FUZZY_THRESHOLD = 85  # 0-100, higher => stricter mapping

# Imputation: 'none' -> do not impute, 'median' -> numeric medians, 'mode' -> categorical modes
IMPUTE_STRATEGY = "median"  # choose: "none", "median", "mode"

# Sentinels to treat as missing (common sensor placeholders)
SENTINEL_VALUES = set(["", "nan", "null", "n/a", "na", "none", "-", "--", "9999", "-9999", -999, -9999, 999999999])

# Whether to create missingness indicator columns (True recommended)
CREATE_MISSING_INDICATORS = True

# Whether to drop exact duplicate rows after cleaning
DROP_DUPLICATES = True

# Max rows to preview when detecting types (keeps speed reasonable)
SAMPLE_ROWS_FOR_TYPE_DETECT = 1000

# --------------------------------------------------

def list_excel_files(folder):
    patterns = ["*.xlsx", "*.xls", "*.xlsm"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = sorted(files)
    return files

def slugify_colname(s):
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w_]", "", s)
    s = s.lower()
    return s

def canonicalize_columns(cols, canonical_list, threshold=FUZZY_THRESHOLD):
    """
    Map each column in cols to canonical_list if close enough. Otherwise fallback to slugify.
    Returns mapping original -> mapped.
    """
    mapping = {}
    canonical_set = set(canonical_list)
    for c in cols:
        if not isinstance(c, str):
            c_str = str(c)
        else:
            c_str = c
        if canonical_list:
            match = process.extractOne(c_str, canonical_list, scorer=fuzz.token_sort_ratio)
            if match:
                matched_name, score, _ = match
                if score >= threshold:
                    mapping[c_str] = matched_name
                    continue
        # fallback
        mapping[c_str] = slugify_colname(c_str)
    return mapping

def read_all_excels(folder):
    files = list_excel_files(folder)
    dfs = []
    if not files:
        print(f"No Excel files found in folder {folder}.")
    for f in files:
        try:
            xls = pd.ExcelFile(f)
            for sheet in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl" if f.lower().endswith("xlsx") else None)
                except Exception:
                    # fallback without engine; let pandas decide
                    df = pd.read_excel(f, sheet_name=sheet)
                # attach provenance
                df.attrs["__source_file__"] = os.path.basename(f)
                df.attrs["__source_sheet__"] = sheet
                dfs.append(df)
                print(f"Read {os.path.basename(f)} :: sheet '{sheet}' -> shape {df.shape}")
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    return dfs

def detect_column_types(df_sample):
    """
    Infer probable column types based on sample values.
    Returns dict col -> one of ('numeric','datetime','boolean','categorical','text')
    """
    types = {}
    for col in df_sample.columns:
        ser = df_sample[col].dropna().astype(str).head(SAMPLE_ROWS_FOR_TYPE_DETECT)
        if ser.empty:
            types[col] = "categorical"
            continue
        num_count = 0
        date_count = 0
        bool_count = 0
        text_count = 0
        for v in ser:
            s = str(v).strip()
            if s == "":
                continue
            # numeric check (allow commas)
            s_num = s.replace(",", "")
            try:
                float(s_num)
                num_count += 1
                continue
            except Exception:
                pass
            # datetime check
            try:
                _ = dateparser.parse(s, fuzzy=False)
                date_count += 1
                continue
            except Exception:
                pass
            # boolean check
            if s.lower() in {"true", "false", "yes", "no", "y", "n", "0", "1"}:
                bool_count += 1
                continue
            text_count += 1
        counts = {"numeric": num_count, "datetime": date_count, "boolean": bool_count, "text": text_count}
        inferred = max(counts, key=counts.get)
        if inferred == "text":
            types[col] = "categorical"
        else:
            types[col] = inferred
    return types

def clean_dataframe(df, col_mapping, global_options):
    """
    Apply cleaning steps on a single dataframe and return cleaned df.
    - rename columns via col_mapping (original->mapped)
    - strip whitespace, normalize common categorical synonyms
    - convert numeric-like strings to numbers
    - parse dates
    - replace sentinel values with NaN
    - create provenance columns if not present
    """
    df = df.copy()
    # rename
    rename_map = {orig: col_mapping.get(orig, col_mapping.get(str(orig), slugify_colname(orig))) for orig in df.columns}
    df = df.rename(columns=rename_map)
    # ensure provenance columns
    if "__source_file__" in df.attrs and "__source_file__" not in df.columns:
        df["source_file"] = df.attrs.get("__source_file__", "")
    if "__source_sheet__" in df.attrs and "__source_sheet__" not in df.columns:
        df["source_sheet"] = df.attrs.get("__source_sheet__", "")
    # Strip whitespace in string columns and normalize empty-like tokens
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            # convert bytes to str if needed
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
            df[col] = df[col].astype(str).str.strip()
            # common normalization: empty-like to NaN
            df[col] = df[col].replace(list(global_options["sentinels"]), np.nan)
            # normalize boolean-like strings
            df[col] = df[col].replace({"TRUE":"True","True":"True","true":"True","FALSE":"False","False":"False","false":"False"})
    # Replace numeric sentinel integers in numeric columns (best-effort) - we will handle type conv next
    # Convert columns that look numeric or datetime
    inferred = detect_column_types(df.sample(n=min(len(df), global_options["sample_for_type"]), random_state=0) if len(df)>0 else df)
    for col, itype in inferred.items():
        try:
            if itype == "numeric":
                # clean up thousands separators, currency symbols, unit suffixes
                series = df[col].astype(str).replace({"nan": np.nan})
                # remove common currency symbols and whitespace
                series = series.str.replace(r"[\$,£€\s]", "", regex=True)
                # handle unit suffixes like 'mV', 'V', 'km', 'm' by removing letters (keeping sign and decimal)
                series = series.str.replace(r"[^\d\.\-eE,]", "", regex=True)
                # remove commas left
                series = series.str.replace(",", "", regex=True)
                num = pd.to_numeric(series, errors="coerce")
                # treat sentinel numeric patterns as NaN
                num = num.where(~num.isin(global_options["sentinel_numeric_values"]), np.nan)
                df[col] = num
            elif itype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            elif itype == "boolean":
                ser = df[col].astype(str).str.lower().map(
                    {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False, "1": True, "0": False}
                )
                df[col] = ser
            else:
                # categorical/text: normalize common variants (e.g., YES/Yes/yes -> yes)
                df[col] = df[col].astype(str).replace({"nan": np.nan})
                df[col] = df[col].where(~df[col].isin([np.nan, "nan", "None", "NoneType"]), np.nan)
                # try lowercasing where it makes sense
                # but don't lowercase possible IDs that look numeric or mixed with underscores
                sample_vals = df[col].dropna().astype(str).head(10).tolist()
                if sample_vals and all(re.match(r"^[A-Za-z\s]+$", v) for v in sample_vals):
                    df[col] = df[col].str.lower().str.strip()
        except Exception as e:
            print(f"Warning: could not process column '{col}': {e}")
    # Replace any remaining sentinel strings
    df = df.replace(list(global_options["sentinels"]), np.nan)
    # Create missingness indicators
    if global_options["create_missing_indicators"]:
        for col in list(df.columns):
            miss_col = f"{col}__missing"
            df[miss_col] = df[col].isna().astype(int)
    # Drop columns that are completely empty
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        print(f"Dropping entirely-empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)
    return df

def unify_and_merge(dfs, col_mapping, options):
    cleaned = []
    for df in dfs:
        c = clean_dataframe(df, col_mapping, options)
        cleaned.append(c)
    if not cleaned:
        return pd.DataFrame()
    merged = pd.concat(cleaned, ignore_index=True, sort=False)
    # Drop exact duplicates if requested
    if options["drop_duplicates"]:
        before = len(merged)
        merged = merged.drop_duplicates(ignore_index=True)
        after = len(merged)
        print(f"Dropped {before-after} duplicate rows")
    return merged

def impute_missing(df, strategy):
    """
    Impute numeric columns with median (if strategy='median') and categorical with mode (if 'mode').
    If strategy == 'none', do nothing.
    """
    if strategy == "none":
        return df, {}
    impute_info = {}
    # numeric impute
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        if strategy == "median":
            medians = df[num_cols].median()
            df[num_cols] = df[num_cols].fillna(medians)
            impute_info["numeric"] = medians.to_dict()
        elif strategy == "mean":
            means = df[num_cols].mean()
            df[num_cols] = df[num_cols].fillna(means)
            impute_info["numeric"] = means.to_dict()
    # categorical impute (object / category / bool)
    cat_cols = [c for c in df.columns if c not in num_cols and not pd.api.types.is_datetime64_any_dtype(df[c])]
    if cat_cols and strategy in ("mode", "median"):  # using 'mode' as the fallback
        modes = {}
        for c in cat_cols:
            try:
                m = df[c].mode(dropna=True)
                val = m.iloc[0] if not m.empty else None
            except Exception:
                val = None
            if val is not None and pd.isna(val) is False:
                df[c] = df[c].fillna(val)
                modes[c] = val
        if modes:
            impute_info["categorical"] = modes
    return df, impute_info

def generate_metadata(mapping, inferred_types, impute_info, options):
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "column_mapping": mapping,
        "inferred_types_sample": inferred_types,
        "imputation": impute_info,
        "options": options
    }
    return meta

def main():
    # Prepare global options
    options = {
        "sentinels": set([s for s in SENTINEL_VALUES]),
        "create_missing_indicators": CREATE_MISSING_INDICATORS,
        "drop_duplicates": DROP_DUPLICATES,
        "sample_for_type": SAMPLE_ROWS_FOR_TYPE_DETECT,
        "sentinel_numeric_values": { -999, -9999, 9999, 999999999 }
    }

    print("Listing Excel files...")
    dfs = read_all_excels(INPUT_FOLDER)
    if not dfs:
        print("No dataframes read. Exiting.")
        return

    # Build global column set
    all_cols = []
    for df in dfs:
        all_cols.extend([str(c) for c in df.columns])
    all_cols_unique = sorted(set(all_cols))

    # Generate mapping
    if CANONICAL_COLS:
        mapping = canonicalize_columns(all_cols_unique, CANONICAL_COLS, threshold=FUZZY_THRESHOLD)
    else:
        mapping = {c: slugify_colname(c) for c in all_cols_unique}
    print("Column mapping preview (first 20):")
    for k, v in list(mapping.items())[:20]:
        print(f"  {k} -> {v}")

    # Merge & clean
    merged = unify_and_merge(dfs, mapping, options)
    if merged.empty:
        print("Merged dataframe is empty after cleaning. Exiting.")
        return

    # Infer types on merged head
    inferred_types = detect_column_types(merged.head(SAMPLE_ROWS_FOR_TYPE_DETECT))

    # Impute if requested
    impute_info = {}
    if IMPUTE_STRATEGY != "none":
        print(f"Imputing missing values using strategy: {IMPUTE_STRATEGY}")
        merged, impute_info = impute_missing(merged, IMPUTE_STRATEGY)

    # Final housekeeping: reorder columns to put provenance at end
    prov_cols = [c for c in ["source_file", "source_sheet"] if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in prov_cols]
    merged = merged[other_cols + prov_cols]

    # Save outputs
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved merged cleaned CSV: {OUTPUT_CSV} (shape: {merged.shape})")

    metadata = generate_metadata(mapping, inferred_types, impute_info, {
        "impute_strategy": IMPUTE_STRATEGY,
        "create_missing_indicators": CREATE_MISSING_INDICATORS,
        "drop_duplicates": DROP_DUPLICATES,
        "canonical_cols_given": bool(CANONICAL_COLS),
        "fuzzy_threshold": FUZZY_THRESHOLD
    })
    with open(METADATA_JSON, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"Saved metadata: {METADATA_JSON}")

if __name__ == "__main__":
    main()
