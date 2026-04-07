"""
select_and_merge_cooling_sheets.py

Purpose:
 - For every Excel workbook in INPUT_FOLDER, automatically pick the sheet most likely
   to contain cooling-tower parameters using heuristic scoring, then merge those selected
   sheets (raw, no preprocessing) into one CSV.

Outputs:
 - merged_selected_sheets.csv   : concatenated raw rows from selected sheets
 - flagged_candidates.csv      : per-workbook top candidate when score < SELECTION_THRESHOLD (for review)
 - selection_report.json       : summary of decisions per workbook

Configuration:
 - Change INPUT_FOLDER, OUTPUT_CSV, and SELECTION_THRESHOLD below as required.
"""

import os
import glob
import json
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

# ---------------- USER CONFIG ----------------
INPUT_FOLDER = "./excel_files"
OUTPUT_CSV = "merged_selected_sheets.csv"
FLAGGED_CSV = "flagged_candidates.csv"
REPORT_JSON = "selection_report.json"

# Selection behavior:
SELECTION_THRESHOLD = 40   # minimum score to auto-accept the top-scoring sheet per workbook
INCLUDE_FLAGGED_IN_OUTPUT = False  # if True, include flagged top candidate sheets into merged output anyway

# How many data rows to sample when checking numeric columns or inspecting first N values
DATA_SAMPLE_ROWS = 5

# Heuristics lists (customize as needed)
REQUIRED_KEYWORDS = ["tds", "conductivity", "cond", "ec", "flow", "temperature", "temp", "inlet", "outlet", "return", "supply"]
HELPFUL_KEYWORDS = ["ph", "chloride", "hardness", "alkalinity", "ppm", "mg/l", "cycles", "bleed", "blowdown", "gpm", "m3/h"]
COOLING_PHRASES = ["cooling", "tower", "coolant", "circulation", "cooling_tower", "cooling tower"]
BAD_TOKENS = ["summary", "report", "chart", "figure", "dashboard", "notes", "comments", "makeup", "make-up", "make up"]  # 'makeup' often to exclude
UNITS_PATTERNS = [r"°c", r"degc", r"c\b", r"ppm\b", r"mg/?l", r"gpm\b", r"m3/h", r"l/s", r"m3h", r"psig", r"kpa"]

FUZZY_THRESHOLD = 80  # fuzzy match threshold (0-100) for header / token matching

# scoring weights (as in plan)
SCORES = {
    "required_present": 50,
    "helpful_each": 20,
    "header_keyword": 15,
    "numeric_columns": 10,
    "units_bonus": 25,
    "report_penalty": -30,
    "few_columns_penalty": -30
}
# ----------------------------------------------

def list_excel_files(folder):
    patterns = ["*.xlsx", "*.xls", "*.xlsm"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    return sorted(files)

def normalize_token(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^\w\s/%°\.]", " ", s)  # keep percent, degree, dot etc
    s = s.strip()
    return s

def fuzz_contains_any(text, keywords, threshold=FUZZY_THRESHOLD):
    """
    Return list of keywords that fuzzy-match parts of text above threshold.
    """
    found = []
    if not text:
        return found
    for k in keywords:
        # use token_set_ratio style approximation
        score = fuzz.token_sort_ratio(k, text)
        if score >= threshold:
            found.append((k, int(score)))
    return found

def detect_units_in_tokens(tokens):
    joined = " ".join(tokens)
    j = joined.lower()
    for pat in UNITS_PATTERNS:
        if re.search(pat, j):
            return True
    return False

def count_numeric_columns(df, sample_rows=DATA_SAMPLE_ROWS):
    """
    Count columns that appear numeric in the first sample_rows (no dtype conversions are done globally).
    Uses pandas.to_numeric on sampled strings with errors='coerce' to test numeric-ness.
    """
    if df.shape[1] == 0:
        return 0
    nrows = min(sample_rows, len(df))
    if nrows == 0:
        return 0
    sample = df.head(nrows)
    numeric_count = 0
    for col in df.columns:
        # ignore provenance columns if present
        try:
            ser = sample[col].astype(str).str.strip().replace({"nan": ""})
        except Exception:
            ser = sample[col].astype(str)
        # check if at least 2 non-empty sample values convert to number
        converted = pd.to_numeric(ser.replace("", np.nan), errors="coerce")
        num_non_na = converted.notna().sum()
        if num_non_na >= 2:
            numeric_count += 1
    return numeric_count

def score_sheet(sheet_df, sheet_name):
    """
    Compute heuristic score for a sheet DataFrame.
    We look at headers (sheet_df.columns) and first DATA_SAMPLE_ROWS rows.
    """
    score_breakdown = defaultdict(int)
    # gather header tokens and first rows tokens
    headers = [str(c) for c in sheet_df.columns]
    header_text = " ".join([normalize_token(h) for h in headers])
    # sample cell texts
    sample_texts = []
    for row in sheet_df.head(DATA_SAMPLE_ROWS).itertuples(index=False, name=None):
        for cell in row:
            sample_texts.append(normalize_token(cell))
    # tokens to search
    tokens = headers + sample_texts

    # few columns penalty (likely report / chart)
    if len(headers) <= 1:
        score_breakdown["few_columns_penalty"] += SCORES["few_columns_penalty"]

    # required keywords presence in headers or sample
    required_found = False
    for req in REQUIRED_KEYWORDS:
        # check header_text fuzzy and sample tokens
        if fuzz.token_sort_ratio(req, header_text) >= FUZZY_THRESHOLD:
            required_found = True
            break
        # check sample tokens
        for t in tokens:
            if not t:
                continue
            if fuzz.token_sort_ratio(req, t) >= FUZZY_THRESHOLD:
                required_found = True
                break
        if required_found:
            break
    if required_found:
        score_breakdown["required_present"] += SCORES["required_present"]

    # helpful keywords count
    helpful_count = 0
    for helpkw in HELPFUL_KEYWORDS:
        matched = False
        if fuzz.token_sort_ratio(helpkw, header_text) >= FUZZY_THRESHOLD:
            matched = True
        else:
            for t in tokens:
                if not t:
                    continue
                if fuzz.token_sort_ratio(helpkw, t) >= FUZZY_THRESHOLD:
                    matched = True
                    break
        if matched:
            helpful_count += 1
    score_breakdown["helpful_each"] += helpful_count * SCORES["helpful_each"]

    # header contains explicit cooling phrase
    header_phrase_found = False
    for phrase in COOLING_PHRASES:
        if fuzz.token_sort_ratio(phrase, header_text) >= FUZZY_THRESHOLD:
            header_phrase_found = True
            break
    if header_phrase_found:
        score_breakdown["header_keyword"] += SCORES["header_keyword"]

    # numeric columns presence
    numeric_cols = count_numeric_columns(sheet_df, sample_rows=DATA_SAMPLE_ROWS)
    if numeric_cols >= 2:
        score_breakdown["numeric_columns"] += SCORES["numeric_columns"]

    # units detection in headers or sample tokens
    if detect_units_in_tokens(tokens):
        score_breakdown["units_bonus"] += SCORES["units_bonus"]

    # report/chart penalty if bad tokens present in header or first rows (and few numeric columns)
    bad_found = False
    combined_text = header_text + " " + " ".join(sample_texts)
    for bad in BAD_TOKENS:
        if fuzz.token_sort_ratio(bad, combined_text) >= FUZZY_THRESHOLD:
            bad_found = True
            break
    if bad_found and numeric_cols < 2:
        score_breakdown["report_penalty"] += SCORES["report_penalty"]

    total_score = sum(score_breakdown.values())
    return int(total_score), dict(score_breakdown)

def read_sheet_raw(file_path, sheet_name):
    """Read a sheet with pandas as-is (no header inference beyond default)."""
    try:
        # default header=0 to use first row as header — this keeps raw naming
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        # fallback: try reading with header=None (in case file is strange)
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            # if header=None, set generic headers so concatenation is easier
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            return df
        except Exception as e2:
            print(f"ERROR reading {file_path} :: sheet {sheet_name}: {e2}")
            return None

def process_workbook(file_path):
    xls = pd.ExcelFile(file_path)
    workbook_report = {"file": os.path.basename(file_path), "candidates": []}
    for sheet in xls.sheet_names:
        # read small sample for scoring (we don't want heavy reads for all sheets if they are huge)
        try:
            sample_df = pd.read_excel(file_path, sheet_name=sheet, nrows=DATA_SAMPLE_ROWS)
        except Exception:
            # fallback to reading full sheet if nrows fails
            try:
                sample_df = pd.read_excel(file_path, sheet_name=sheet)
            except Exception:
                sample_df = pd.DataFrame()
        score, breakdown = score_sheet(sample_df, sheet)
        workbook_report["candidates"].append({
            "sheet": sheet,
            "score": int(score),
            "score_breakdown": breakdown,
            "ncols": int(sample_df.shape[1]) if sample_df is not None else 0,
            "nrows_sampled": int(min(DATA_SAMPLE_ROWS, len(sample_df))) if sample_df is not None else 0
        })
    # pick top candidate
    if workbook_report["candidates"]:
        sorted_c = sorted(workbook_report["candidates"], key=lambda x: x["score"], reverse=True)
        top = sorted_c[0]
        workbook_report["selected_sheet"] = top["sheet"]
        workbook_report["selected_score"] = top["score"]
        workbook_report["selected_breakdown"] = top["score_breakdown"]
    else:
        workbook_report["selected_sheet"] = None
        workbook_report["selected_score"] = -999
        workbook_report["selected_breakdown"] = {}
    return workbook_report

def main():
    files = list_excel_files(INPUT_FOLDER)
    if not files:
        print(f"No Excel files found in {INPUT_FOLDER}")
        return

    merged_parts = []
    flagged_rows = []
    selection_summary = []

    for f in files:
        print("Processing:", os.path.basename(f))
        wb_report = process_workbook(f)
        sel_sheet = wb_report["selected_sheet"]
        sel_score = wb_report["selected_score"]
        sel_breakdown = wb_report["selected_breakdown"]

        # Decide whether to accept
        flagged = False
        if sel_score >= SELECTION_THRESHOLD and sel_sheet:
            accept = True
        else:
            accept = False
            flagged = True

        if accept or INCLUDE_FLAGGED_IN_OUTPUT:
            # read full sheet raw
            df_full = read_sheet_raw(f, sel_sheet) if sel_sheet else None
            if df_full is not None:
                # add provenance + selection info **without altering data**
                df_full["__source_file__"] = os.path.basename(f)
                df_full["__source_sheet__"] = sel_sheet
                df_full["__selection_score__"] = sel_score
                df_full["__selection_flagged__"] = int(flagged)
                merged_parts.append(df_full)
                print(f"  -> Selected sheet '{sel_sheet}' (score={sel_score}) -> included")
            else:
                print(f"  -> Selected sheet '{sel_sheet}' could not be read; skipping.")
        else:
            print(f"  -> Selected sheet '{sel_sheet}' (score={sel_score}) flagged (below threshold) -> not included")

        # Add to flagged rows summary for review when flagged
        if flagged:
            # find top candidate details
            candidate = {
                "file": os.path.basename(f),
                "selected_sheet": sel_sheet,
                "selected_score": sel_score,
                "selected_breakdown": sel_breakdown,
                "candidates": wb_report["candidates"]
            }
            flagged_rows.append(candidate)
        selection_summary.append(wb_report)

    # Concatenate selected raw sheets
    if merged_parts:
        combined = pd.concat(merged_parts, ignore_index=True, sort=False)
        combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\nSaved merged selected sheets to: {OUTPUT_CSV} (rows={combined.shape[0]}, cols={combined.shape[1]})")
    else:
        print("\nNo sheets were included in output (no accepted sheets).")

    # Save flagged candidates CSV (for human review)
    if flagged_rows:
        # Normalize flagged rows into a DataFrame for easy review
        rows_for_df = []
        for fr in flagged_rows:
            rows_for_df.append({
                "file": fr["file"],
                "selected_sheet": fr["selected_sheet"],
                "selected_score": fr["selected_score"],
                "selected_breakdown": json.dumps(fr["selected_breakdown"])
            })
        df_flagged = pd.DataFrame(rows_for_df)
        df_flagged.to_csv(FLAGGED_CSV, index=False, encoding="utf-8")
        print(f"Saved flagged candidates to: {FLAGGED_CSV} ({len(df_flagged)} workbooks flagged)")
    else:
        print("No flagged workbooks (all selected auto-accepted).")

    # Save selection report (full)
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(selection_summary, fh, indent=2, default=str)
    print(f"Saved selection report to: {REPORT_JSON}")

if __name__ == "__main__":
    main()
