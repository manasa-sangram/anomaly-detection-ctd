from pathlib import Path
from ctdam.entry.casts import Casts
from ctdam.parser.read_ctd_data import parse

# Everything that starts work goes under this guard (required on Windows)
if __name__ == "__main__":
    DATA_DIR = Path(r"C:\Users\Manasa\OneDrive\Desktop\Anamoly Detection in CTD data\emb_data")

    # ---- Part A: try ALL .cnv files one by one, count problems ----
    files = sorted(DATA_DIR.rglob("*.cnv"))
    ok, failed, too_small = 0, [], []

    for f in files:
        try:
            ds = parse(f)
            # same thresholds ctdam's size check uses: 10 if binned, else 500
            limit = 10 if ds.access.binned else 500
            if ds.access.size < limit:
                too_small.append((f.name, ds.access.size))
            else:
                ok += 1
        except Exception as e:
            failed.append((f.name, repr(e)))

    print(f"OK: {ok} | too small: {len(too_small)} | failed: {len(failed)}")
    print("First failures:", failed[:3])
    print("First too-small:", too_small[:3])

    # ---- Part B: the real Casts call, single process ----
    casts = Casts(
        path_to_data=DATA_DIR,
        use_multiprocessing=False,   # no worker processes, errors stay visible
    )
    print("Casts loaded:", len(casts.data))