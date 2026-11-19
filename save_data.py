from visualize_data import dataframe_access
from visualize_data import read_idx_ranges
from pathlib import Path
from collections import Counter



if __name__ == "__main__":
    idx_files ={}
    df = dataframe_access()
    print("Rows, columns:", df.shape)
    print("Casts loaded:", df["file"].nunique())

    # .pkl ("pickle") stores the table exactly as it is,
    # including column types, so nothing changes when we read it back
    df.to_pickle("ctd_data.pkl")

    print("Saved to ctd_data.pkl")

# ============================================================
# DATA VALIDATION
# ============================================================

    cnv_files = set(df["file"].unique())
    idx_names = set(idx_files.keys())

    cnv_names = {
    Path(name).stem for name in cnv_files
}

    matched = cnv_names & idx_names
    cnv_without_idx = cnv_names - idx_names
    idx_without_cnv = idx_names - cnv_names

    print("\n========== IDX COVERAGE ==========")
    print("Unique CNV files:", len(cnv_names))
    print("IDX files:", len(idx_names))
    print("Matched files:", len(matched))
    print("CNV files without IDX:", len(cnv_without_idx))
    print("IDX files without CNV:", len(idx_without_cnv))


    print("\nExamples of CNV files without IDX:")
    print(list(sorted(cnv_without_idx))[:10])

    print("\nExamples of IDX files without CNV:")
    print(list(sorted(idx_without_cnv))[:10])

# ============================================================
# CHECK THE DATA AVAILABLE FOR TRAINING AND EVALUATION
# ============================================================

    annotation_counts = Counter()

    for idx_path in idx_files.values():
        ranges = read_idx_ranges(idx_path)

        for item in ranges:
            annotation_counts[
            (item["delete"], item["interpolate"])
            ] += 1

    print("\nAnnotation combinations found:\n")

    for (delete, interpolate), count in sorted(annotation_counts.items()):
        print(
            f"delete={delete}, interpolate={interpolate}: "
            f"{count} annotated ranges"
    )
