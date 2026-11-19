from pathlib import Path
from collections import Counter
from ctdam import parse

DATA_DIR = Path(r"C:\Users\Manasa\OneDrive\Desktop\Anamoly Detection in CTD data\CTD_DATA_CORRECT")
files = list(DATA_DIR.rglob("*.cnv"))
print("CNV files found:", len(files))

# Check A: does any file name appear twice (in different folders)?
dupes = [n for n, c in Counter(f.name for f in files).items() if c > 1]
print("Duplicate file names:", len(dupes), dupes[:5])

# Check B: rows in the dataframe vs scans in the file
mismatch = 0
for f in files:
    ds = parse(f)
    rows = len(ds[["pressure", "temperature", "conductivity"]].to_dataframe())
    scans = ds.sizes["scan"]
    if rows != scans:
        mismatch += 1
        if mismatch <= 5:                       # show only the first 5
            print(f.name, "| scans:", scans, "| rows:", rows, "| dims:", ds["temperature"].dims)
print("Files where rows != scans:", mismatch, "of", len(files))