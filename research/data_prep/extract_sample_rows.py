import pandas as pd
from pathlib import Path
import sys

def extract_samples(directory):
    """Extract first 10 rows from each CSV file to analyze structure"""
    
    path = Path(directory)
    if not path.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find all CSV files
    csv_files = list(path.glob('*.csv'))
    csv_files = [f for f in csv_files if 'consolidated' not in f.name.lower() 
                 and 'sample' not in f.name.lower()]
    csv_files = sorted(csv_files)
    
    print(f"Found {len(csv_files)} CSV files\n")
    print("=" * 120)
    
    # Create output file
    output_path = path / 'sample_rows_analysis.csv  '
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("SAMPLE ROWS FROM ALL CSV FILES\n")
        f.write("Showing first 10 rows from each file to understand column structure\n")
        f.write("=" * 120 + "\n\n")
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"Processing [{i}/{len(csv_files)}]: {csv_file.name}")
            
            try:
                # Try to read the file
                df = pd.read_csv(csv_file)
                
                f.write("\n" + "=" * 120 + "\n")
                f.write(f"FILE {i}: {csv_file.name}\n")
                f.write("=" * 120 + "\n")
                f.write(f"Total rows: {len(df)}\n")
                f.write(f"Total columns: {len(df.columns)}\n\n")
                
                # Show column names
                f.write("COLUMN NAMES:\n")
                for col_idx, col in enumerate(df.columns, 1):
                    f.write(f"  {col_idx}. {col}\n")
                
                f.write("\n" + "-" * 120 + "\n")
                f.write("FIRST 10 ROWS:\n")
                f.write("-" * 120 + "\n")
                
                # Show first 10 rows
                sample = df.head(10)
                f.write(sample.to_string(index=True))
                f.write("\n\n")
                
            except Exception as e:
                f.write("\n" + "=" * 120 + "\n")
                f.write(f"FILE {i}: {csv_file.name}\n")
                f.write("=" * 120 + "\n")
                f.write(f"❌ ERROR reading file: {e}\n\n")
                print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 120)
    print(f"✓ Analysis complete!")
    print(f"✓ Output saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.2f} KB")
    print("=" * 120)
    
    return output_path

def main():
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
    else:
        print("\n❌ ERROR: Please provide the directory path containing your CSV files")
        print("\nUsage:")
        print("  python extract_sample_rows.py <path_to_csv_directory>")
        print("\nExample:")
        print("  python extract_sample_rows.py ./chosen_csvs")
        print("  python extract_sample_rows.py C:\\Users\\YourName\\Documents\\CoolingTowerData")
        return
    
    print("\n" + "=" * 120)
    print("CSV SAMPLE ROW EXTRACTOR")
    print("Extracting first 10 rows from each file to analyze column structure")
    print("=" * 120)
    print(f"\n📁 Directory: {data_directory}\n")
    
    extract_samples(data_directory)

if __name__ == "__main__":
    main()