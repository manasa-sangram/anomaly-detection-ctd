from pathlib import Path
from ctdam.parser import PARSEABLE_FILE_FORMATS
from ctdam.parser.read_ctd_data import parse

# On Windows, code that may start processes must live under this guard
if __name__ == "__main__":
    DATA_DIR = Path(r"C:\Users\Manasa\OneDrive\Desktop\Anamoly Detection in CTD data\emb_data")

    # ---- Stage 1: does ctdam accept any of your files? ----
    print("Accepted suffixes:", PARSEABLE_FILE_FORMATS)

    all_files = [f for f in DATA_DIR.rglob("*") if f.is_file()]
    matching = [f for f in all_files if f.suffix in PARSEABLE_FILE_FORMATS]
    print("Files in folder:      ", len(all_files))
    print("Files ctdam accepts:  ", len(matching))
    print("Your suffixes:        ", sorted({f.suffix for f in all_files}))

    # ---- Stages 2 and 3: try parsing a few files, showing the real error ----
    for f in matching[:5]:
        try:
            ds = parse(f)   # same call ctdam makes internally
            # .access.size and .access.binned are what the size check uses
            print(f.name, "| binned:", ds.access.binned, "| size:", ds.access.size)
        except Exception as e:
            print(f.name, "| FAILED:", repr(e))   


          # this is the hidden error