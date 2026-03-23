#!/usr/bin/env python3
"""
select_ct_sheets.py

Two-layer selection:
  Layer 1: sheet-name must contain any of ["cooling","tower","parameter","ct"]
           and must NOT contain "make up" anywhere (sheet name or header).
  Layer 2: prefer sheets that contain "ph" in header / first rows.

Usage:
  python select_ct_sheets.py --input_dir "/path/to/excels" --output_dir "./chosen_csvs" --log "./selection_log.csv"

Produces:
  - one CSV per workbook with the chosen sheet
  - selection_log.csv describing choice, scores and flags
"""

import os
import argparse
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# -------------- Config ----------------
LAYER1_NAME_WORDS = ["cooling", "tower", "parameter", "ct"]
EXCLUDE_PHRASE = "make up"   # exclude if this phrase appears in sheet name or header (case-insensitive)
PH_KEYWORD = "ph"

SAMPLE_HEADER_ROWS = 5   # how many top rows to scan for 'ph' / textual clues
MIN_ROWS_FOR_SERIES = 8  # helpful but optional

# -------------- Helpers --------------
def contains_any_token(text, tokens):
    if text is None:
        return False
    s = str(text).lower()
    return any(tok in s for tok in tokens)

def fuzzy_contains(text, token, thresh=0.85):
    if text is None:
        return False
    return SequenceMatcher(None, str(text).lower(), token.lower()).ratio() >= thresh

def sheet_name_layer1_pass(sheet_name):
    s = str(sheet_name).lower()
    # must contain any of the allowed words
    has_allow = any(w in s for w in LAYER1_NAME_WORDS)
    # must NOT contain exclude phrase
    has_exclude = EXCLUDE_PHRASE in s
    return has_allow and (not has_exclude)

def header_contains_exclude(df):
    # check if any header text or first few rows contain the exclude phrase
    try:
        # check pandas column names
        cols = [str(c) for c in df.columns]
        if any(EXCLUDE_PHRASE in c.lower() for c in cols):
            return True
        # check first SAMPLE_HEADER_ROWS rows (cells)
        top = df.head(SAMPLE_HEADER_ROWS).astype(str).fillna("").apply(lambda col: " ".join(col.values), axis=0)
        if any(EXCLUDE_PHRASE in v.lower() for v in top.values):
            return True
    except Exception:
        return False
    return False

def find_ph_in_headers_or_top_rows(df):
    # return count of occurrences of 'ph' (case-insensitive) in column names or first SAMPLE_HEADER_ROWS rows
    count = 0
    try:
        # check columns
        for c in df.columns:
            if PH_KEYWORD in str(c).lower():
                count += 2  # weight header hits higher
            else:
                # fuzzy match "ph" variants like "pH" vs "ph"
                if fuzzy_contains(str(c).lower(), PH_KEYWORD, thresh=0.9):
                    count += 1
        # check cell values in top rows (textual)
        top = df.head(SAMPLE_HEADER_ROWS).astype(str).fillna("")
        for col in top.columns:
            for val in top[col].values:
                if PH_KEYWORD in str(val).lower():
                    count += 1
    except Exception:
        pass
    return count

def numeric_series_signal(df, sample_rows=20):
    # crude numeric density in first sample_rows rows (helpful to prefer time-series)
    try:
        df_sample = df.head(sample_rows)
        total = 0
        numeric = 0
        for col in df_sample.columns:
            for v in df_sample[col].values:
                total += 1
                try:
                    if pd.isna(v):
                        continue
                    float(v)
                    numeric += 1
                except Exception:
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        numeric += 1
        return (numeric / total) if total>0 else 0.0
    except Exception:
        return 0.0

# -------------- Core logic --------------
def choose_sheet_for_workbook(path):
    workbook_name = os.path.basename(path)
    result = {
        "workbook": workbook_name,
        "status": "error",
        "chosen_sheet": None,
        "reason": None,
        "ph_count": 0,
        "numeric_density": 0.0,
        "layer1_candidates": [],
    }
    try:
        all_sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        result["status"] = "error"
        result["reason"] = f"read_error: {e}"
        return result

    # 1) filter sheets by layer1 name test & exclude phrase in sheet name
    layer1_candidates = []
    for sname, df in all_sheets.items():
        # skip obviously empty sheets quickly
        try:
            df2 = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        except Exception:
            df2 = df
        if not sheet_name_layer1_pass(sname):
            continue
        # also ensure exclude phrase not present in headers/first rows
        if header_contains_exclude(df2):
            continue
        # sheet passes layer1
        layer1_candidates.append((sname, df2))
    result["layer1_candidates"] = [s for s,_ in layer1_candidates]

    if not layer1_candidates:
        # fallback: no sheet matched layer1 -> mark and return best heuristic sheet (highest numeric density)
        best = None
        best_score = -1.0
        for sname, df in all_sheets.items():
            df2 = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
            nd = numeric_series_signal(df2)
            if nd > best_score:
                best = (sname, df2)
                best_score = nd
        if best:
            chosen_name = best[0]
            out = {
                "workbook": workbook_name,
                "status": "no_layer1_match",
                "chosen_sheet": chosen_name,
                "reason": "fallback_numeric_best",
                "ph_count": find_ph_in_headers_or_top_rows(best[1]),
                "numeric_density": best_score,
                "layer1_candidates": []
            }
            return out
        else:
            result["status"] = "no_sheets_readable"
            result["reason"] = "no readable sheets"
            return result

    # 2) among layer1 candidates, compute PH-count and numeric_density; choose best by ph_count then numeric_density
    scored = []
    for sname, df in layer1_candidates:
        phc = find_ph_in_headers_or_top_rows(df)
        nd = numeric_series_signal(df)
        rows = df.shape[0]
        cols = df.shape[1]
        scored.append({"sheet": sname, "df": df, "ph_count": phc, "numeric_density": nd, "rows": rows, "cols": cols,
                       "score": (phc * 2.0) + (nd * 1.0) + (min(rows,200)/200.0)})
    scored_sorted = sorted(scored, key=lambda x: (x["score"], x["ph_count"], x["numeric_density"]), reverse=True)
    best = scored_sorted[0]

    # choose best
    chosen = best["sheet"]
    phc = best["ph_count"]
    nd = best["numeric_density"]
    reason = []
    reason.append("layer1_name_match")
    if phc > 0:
        reason.append("ph_found")
    else:
        reason.append("ph_missing")
    if nd >= 0.3:
        reason.append("numeric_series")
    reason_str = ";".join(reason)

    return {"workbook": workbook_name, "status": "ok", "chosen_sheet": chosen,
            "reason": reason_str, "ph_count": phc, "numeric_density": nd,
            "layer1_candidates": [s["sheet"] for s in scored_sorted]}

# -------------- Batch driver --------------
def process_folder(input_dir, output_dir, log_csv):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.xls', '.xlsx', '.xlsm'))]
    rows = []
    for f in files:
        path = os.path.join(input_dir, f)
        res = choose_sheet_for_workbook(path)
        if res["status"] in ("ok", "no_layer1_match"):
            # attempt to write chosen sheet to CSV
            try:
                # read chosen sheet again and save
                df = pd.read_excel(path, sheet_name=res["chosen_sheet"])
                out_name = os.path.splitext(f)[0].replace(" ", "_") + "__chosen.xlsx"
                out_path = os.path.join(output_dir, out_name)
                df.to_excel(out_path, index=False)
            except Exception as e:
                res["status"] = "error_writing"
                res["error"] = str(e)
                out_path = ""
        else:
            out_path = ""
        log_row = {
            "workbook": res["workbook"],
            "status": res["status"],
            "chosen_sheet": res.get("chosen_sheet",""),
            "reason": res.get("reason",""),
            "ph_count": res.get("ph_count",0),
            "numeric_density": res.get("numeric_density",0.0),
            "layer1_candidates": "|".join(res.get("layer1_candidates",[])),
            "out_csv": out_path
        }
        print(f"[{log_row['status']}] {f} -> {log_row['chosen_sheet']}  ({log_row['reason']}) ph_count={log_row['ph_count']} nd={log_row['numeric_density']:.2f}")
        rows.append(log_row)
    pd.DataFrame(rows).to_csv(log_csv, index=False)
    print("Done. Log saved to:", log_csv)

# -------------- CLI --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder with Excel workbooks")
    parser.add_argument("--output_dir", default="./chosen_csvs", help="Where to write chosen CSVs")
    parser.add_argument("--log", default="./selection_log.csv", help="Selection log CSV path")
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.log)
