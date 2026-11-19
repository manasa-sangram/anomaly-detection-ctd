from pathlib import Path
from ctdam.entry.casts import Casts
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
import warnings
from ctdam.conv.cast_borders import get_cast_borders
from ctdam.proc.modules.geomar_wildedit import wildedit_geomar
from visualize_data import read_idx_ranges

if __name__=="__main__":
    DATA_FILE = Path("ctd_data.pkl")

    if not DATA_FILE.exists():
        raise FileNotFoundError("ctd_data.pkl not found. Run save_data.py first!")

    df = pd.read_pickle(DATA_FILE)
    print("\nDataframe Loaded:", df.shape)
   

    # df = dataframe_access()
    # print("\nDataframe Loaded")

    DATA_DIR = Path(r"C:\Users\Manasa\OneDrive\Desktop\Anamoly Detection in CTD data\CTD_DATA_CORRECT")

    idx_files = {}

    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.lower().endswith(".idx"):
                idx_path = Path(root) / filename
                idx_files[idx_path.stem] = idx_path

    def read_idx_ranges(idx_path):
            ranges = []
            after_end = False

            with open(idx_path, "r", encoding="utf-8",errors="replace") as f:
                for line in f:
                    line = line.strip()

                    if line == "*END*":
                        after_end = True
                        continue

                    if not after_end or not line:
                        continue

                    parts = line.split()

                    if len(parts) >= 4:
                        start = int(parts[0])
                        end = int(parts[1])
                        delete = int(parts[2])
                        interpolate = int(parts[3])

                        ranges.append({
                    "start": start,
                    "end": end,
                    "delete": delete,
                    "interpolate": interpolate
                    })
            return ranges
    for r in read_idx_ranges(idx_files["EMB311_019-01_CTD_0024"]):
            print(r)
    



#===========================================================
# Creating idx ground truth labels for every matched CTD cast
#==========================================================
    
    df["y_true"] =-1   #-1 - not labelled by a matched idx file

    cast_groups = {
        Path(filename).stem: indices
        for filename, indices in df.groupby("file", sort=False).groups.items()
}
    matched_count = 0

    for i, (cast_name, idx_path) in enumerate(idx_files.items(), start=1):
        indices = cast_groups.get(cast_name)

        if indices is None:
            continue

        ranges = read_idx_ranges(idx_path)

        labels = np.zeros(len(indices), dtype=int)

        for item in ranges:
            if item["delete"] == 1:
                start = max(0, item["start"] - 1)
                end = min(len(labels), item["end"])

                if start < end:
                    labels[start:end] = 1

        df.loc[indices, "y_true"] = labels
        matched_count += 1

        if i % 100 == 0:
            print(f"Processed {i}/{len(idx_files)} IDX files")
    
    print(f"\nMatched casts labelled: {matched_count}")
    print("\n Ground truth labels counts")
    print(df["y_true"].value_counts().sort_index())


#===========================================================
# NEW: Label alignment check
#===========================================================

    labelled_rows =df[df["y_true"].isin([0,1])]
    scans_per_cast = labelled_rows.groupby("file", sort=False)["scan"]

    first_scan = scans_per_cast.min()
    casts_not_starting_at_1 = (first_scan != 1).sum()

    casts_with_gaps = 0
    for cast_name, scans in scans_per_cast:
        steps = scans.diff().dropna()      # difference between neighbouring scan numbers
        if (steps != 1).any():             # every step should be exactly 1
            casts_with_gaps += 1

    print("\nLabel alignment check")
    print("Casts whose first scan is not 1:", casts_not_starting_at_1)
    print("Casts with missing scans:", casts_with_gaps)



#===========================================================
# Feature Engineering
#===========================================================

    sensor_cols =["pressure" , "temperature" , "conductivity"]

    # print("\nSensor Features")
    # print(sensor_cols)

    # print("\nMissing value in sensors")
    # print(df[sensor_cols].isna().sum())



# ========================================================
# Features used by Isolation Forest
# ========================================================

    sensor_cols = ["temperature" , "conductivity" , "pressure"]

    for col in sensor_cols:
        diff_col = f"{col}_diff"

        df[diff_col] = (df.groupby("file", sort=False)[col].diff())
    df["descent_rate"] = (df.groupby("file", sort=False)["pressure_diff"].transform(lambda s: s.rolling(window=192, center=True, min_periods=1).mean()))

    features =["temperature" , "conductivity" , "pressure" , "temperature_diff" , "conductivity_diff" , "pressure_diff"]

    # print("Features used by Isolation Forest :" , features)

    # print("\nMissing values in engineered features:")
    # print(df[features].isna().sum())

# ===========================================================
# EXPERIMENT 2: ROLLING STANDARD DEVIATION
# ===========================================================

    window_size =11

    for col in sensor_cols:
        std_col = f"{col}_rolling_std"

        df[std_col] =(df.groupby("file", sort=False)[col].transform(
            lambda s: s.rolling(window= window_size , 
                                center = True , min_periods=3).std()
        ))

 #===========================================================
# NEW - EXPERIMENT 3: QC TEST FEATURES (Castelao 2015)
# ===========================================================
# These give the model "context": each scan is compared with its neighbours.
# All of them are calculated inside one cast only (groupby "file").

    for col in ["temperature" , "conductivity"]:
        previous_value = df.groupby("file" , sort = False)[col].shift(1)
        next_value = df.groupby("file" , sort = False)[col].shift(-1)

        # 1) Gradient test: distance from the average of the two neighbours
        #    Example: 10.0, 15.3, 9.9  ->  |15.3 - (10.0 + 9.9) / 2| = 5.35  (big = suspicious)
        df[f"{col}_gradient"] = (df[col] - (previous_value + next_value) / 2).abs()

        # 2) Spike test: gradient MINUS the normal slope between the neighbours
        #    Smooth real change (10 -> 9 -> 8)  gives about 0
        #    Sudden jump       (10 -> 15 -> 10) stays big
        df[f"{col}_spike"] = df[f"{col}_gradient"] - ((next_value - previous_value) / 2).abs()

        
        # 3) Distance from the rolling median (simpler version of the Tukey 53H test)
        #    The median ignores single spikes, so a spike sticks out clearly
        rolling_median = df.groupby("file", sort=False)[col].transform(
            lambda s: s.rolling(window=window_size, center=True, min_periods=3).median()
        )
        df[f"{col}_median_resid"] = (df[col] - rolling_median).abs()

      

# ===========================================================
# Train , test split
# ============================================================

    labelled_files = df.loc[df["y_true"].isin([0,1]) , "file"].unique()
    rng = np.random.default_rng(42)
    shuffled_files = rng.permutation(labelled_files)

    train_end = int(0.70 * len(shuffled_files))
    val_end   = int(0.85 * len(shuffled_files))

    train_files = shuffled_files[:train_end]
    val_files   = shuffled_files[train_end:val_end]   # for choosing settings
    test_files  = shuffled_files[val_end:]            # use only ONCE at the end

    print("Training casts:", len(train_files))
    print("Validation casts:", len(val_files))
    print("Testing casts:", len(test_files))

    # split_index = int(0.8 *len(shuffled_files))

    # train_files = shuffled_files[:split_index]
    # test_files = shuffled_files[split_index:]

    # print("Training casts:", len(train_files))
    # print("Testing casts:", len(test_files))



# # All six features
    all_features = [
    "temperature", "conductivity", "pressure",
    "temperature_diff", "conductivity_diff", "pressure_diff" , "pressure_rolling_std" , "descent_rate" , "temperature_gradient" , "conductivity_gradient",
    "temperature_spike", "conductivity_spike","temperature_median_resid", "conductivity_median_resid",
]

# ===========================================================
# CHOOSE THE APPROACH
# ===========================================================
    APPROACH = "supervised"     # change to "unsupervised" for the second experiment
    

    FEATURE_SET = "baseline"         #"baseline", "qc_tests", "qc_tests_no_raw",
    EVALUATE_ON = "validation" 
    SMOOTH_WINDOW = 1
    print("\nApproach:", APPROACH)
    print("Feature set:", FEATURE_SET)
    print("Evaluate on:", EVALUATE_ON)
    print("Smooth window:", SMOOTH_WINDOW)   

    if EVALUATE_ON == "validation":
        eval_files = val_files
    elif EVALUATE_ON == "test":
        eval_files = test_files
    else:
        raise ValueError("EVALUATE_ON must be 'validation' or 'test'")
 

# ===========================================================
# Model Training : GOOD ROWS only
# ============================================================

    if FEATURE_SET == "baseline":

        selected_features =["temperature" , "conductivity" , "pressure" , "temperature_diff" , "pressure_diff" , "pressure_rolling_std" , "descent_rate"]
    elif FEATURE_SET == "qc_tests":
        selected_features =["temperature" , "conductivity" , "pressure" ,
                            "temperature_diff" , "conductivity_diff" , "pressure_diff" ,
                            "pressure_rolling_std" , "descent_rate" ,
                            "temperature_gradient" , "conductivity_gradient" ,
                            "temperature_spike" , "conductivity_spike" ,
                            "temperature_median_resid" , "conductivity_median_resid"]
    elif FEATURE_SET == "qc_tests_no_raw":
        selected_features =["pressure" ,
                            "temperature_diff" , "conductivity_diff" , "pressure_diff" ,
                            "pressure_rolling_std" , "descent_rate" ,
                            "temperature_gradient" , "conductivity_gradient" ,
                            "temperature_spike" , "conductivity_spike" ,
                            "temperature_median_resid" , "conductivity_median_resid"]

    else:
        raise ValueError("Unknown FEATURE_SET")



    valid= df[all_features].notna().all(axis=1)

    if APPROACH == "supervised":
        # Train on GOOD rows of the training casts only,
        # evaluate on good + bad rows of OTHER casts (disjoint sets)
        train_mask = (df["file"].isin(train_files) & (df["y_true"] == 0) & valid)
        test_mask  = (df["file"].isin(test_files) & df["y_true"].isin([0, 1]) & valid)

    elif APPROACH == "unsupervised":
        # No split, no label filter: train AND evaluate on ALL labelled rows.
        # The model never sees the labels; we only use them at the end to score it.
        all_rows = df["y_true"].isin([0, 1]) & valid
        train_mask = all_rows
        test_mask  = all_rows

    else:
        raise ValueError("APPROACH must be 'supervised' or 'unsupervised'")

    # train_mask =(df["file"].isin(train_files) & (df["y_true"] == 0) & valid)

    X_train = df.loc[train_mask , selected_features].copy()

# ===========================================================
# Testing : GOOD and BAD ROWS only
# ============================================================

    # test_mask = (df["file"].isin(test_files) & df["y_true"].isin([0,1]) & valid)

    X_test = df.loc[test_mask,selected_features].copy()
    y_test = df.loc[test_mask , "y_true"].copy()

    print("\nTraining data:", X_train.shape)
    print("Testing data:", X_test.shape)

    print("\nTraining data contains:")
    print(df.loc[train_mask, "y_true"].value_counts())
    # print("Good rows:", len(X_train))

    print("\nTesting ground truth:")
    print(y_test.value_counts())

# ===========================================================
# ISOLATION FOREST: GOOD-ONLY TRAINING
# ===========================================================

    print("\n====================ISOLATION FOREST=============")

    # max_samples_values = [256, 512, 1024, 2048]

    # results = []
    # for sample_size in max_samples_values:
    #     print(f"\nTesting max_samples = {sample_size}")

    model = IsolationForest(n_estimators=700 , max_samples= 8192 , contamination="auto" , random_state=42,
                        n_jobs=-1)

    model.fit(X_train)
    anomaly_scores = -model.decision_function(X_test)

    #     pr_auc = average_precision_score(
    #     y_test, anomaly_scores
    # )

    #     roc_auc = roc_auc_score(
    #     y_test, anomaly_scores
    # )

    #     results.append({
    #         "max_samples": sample_size,
    #         "PR_AUC": pr_auc,
    #         "ROC_AUC": roc_auc
    #     })
    #     results_df = pd.DataFrame(results)

    #     print("\n========== MAX_SAMPLES RESULTS ==========")
    #     print(results_df.to_string(index=False))
    #     print("\n Training completed !")

# ===========================================================
# ANOMALY SCORES
# ===========================================================
# Higher score = more anomalous

 # ===========================================================
# NEW: SCORE SMOOTHING (optional, controlled by SMOOTH_WINDOW)
# ===========================================================
# Humans delete whole RANGES of scans. So we average each score with its
# neighbouring scans: one noisy score inside a bad range gets help from
# its neighbours. Done inside each cast only. SMOOTH_WINDOW = 1 skips this.

    if SMOOTH_WINDOW > 1:
        score_table = df.loc[test_mask, ["file"]].copy()   # same row order as X_test
        score_table["score"] = anomaly_scores

        smoothed = score_table.groupby("file", sort=False)["score"].transform(
            lambda s: s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
        )
        anomaly_scores = smoothed.to_numpy()
   
# ===========================================================
# EVALUATION
# ===========================================================

    pr_auc = average_precision_score(
    y_test,
    anomaly_scores
)

    roc_auc = roc_auc_score(
    y_test,
    anomaly_scores
)

    # print("\n========== GOOD-ONLY ISOLATION FOREST RESULTS ==========")

    print("PR-AUC:", round(pr_auc, 6))
    print("ROC-AUC:", round(roc_auc, 6))

    print("\nEvaluated rows:", len(y_test))
    print("Good rows:", (y_test == 0).sum())
    print("Bad rows:", (y_test == 1).sum())

    precision, recall, thresholds = precision_recall_curve(
    y_test,
    anomaly_scores
)

    plt.figure(figsize=(8, 6))

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Good-Only Isolation Forest")

    plt.grid()
    plt.tight_layout()
    plt.show()

    # ===========================================================
# WHAT DOES THE MODEL STILL GET WRONG? (on validation casts)
# ===========================================================

    # Take the evaluated rows and put the model's anomaly score next to each one.
    # anomaly_scores is in the same order as X_test, so they line up row by row.
    test_df = df.loc[test_mask, ["file", "scan", "pressure", "y_true"]].copy()
    test_df["score"] = anomaly_scores

        # The model gives a score, not a yes/no. We pick the cut-off with the best F1
    # (best balance between precision and recall).
    f1 = 2 * precision * recall / (precision + recall + 1e-12)   # +1e-12 avoids dividing by 0
    best = np.argmax(f1[:-1])          # last point has no threshold, so skip it
    threshold = thresholds[best]

    # Score above the cut-off → model says "bad" (1)
    test_df["y_pred"] = (test_df["score"] >= threshold).astype(int)

    print("\nBest-F1 threshold:", round(threshold, 4))
    print("Precision:", round(precision[best], 3),
          "| Recall:", round(recall[best], 3),
          "| F1:", round(f1[best], 3))

        # Everyone starts as "Correct good", then we overwrite the other 3 groups
    test_df["outcome"] = "Correct good"
    test_df.loc[(test_df["y_true"] == 1) & (test_df["y_pred"] == 1), "outcome"] = "Caught bad"
    test_df.loc[(test_df["y_true"] == 1) & (test_df["y_pred"] == 0), "outcome"] = "Missed bad"
    test_df.loc[(test_df["y_true"] == 0) & (test_df["y_pred"] == 1), "outcome"] = "False alarm"

    print("\nOutcome counts:")
    print(test_df["outcome"].value_counts())

        # For each cast: what share of its bad rows did we catch? (recall per cast)
    per_cast = (test_df[test_df["y_true"] == 1]
                .groupby("file")["y_pred"].mean()
                .sort_values())

    print("\nRecall per cast (0 = caught nothing, 1 = caught everything):")
    print(per_cast.describe().round(3))
    print("\nWorst 5 casts:")
    print(per_cast.head().round(3))

    colors = {"Correct good": "lightgrey", "Caught bad": "tab:blue",
              "Missed bad": "tab:red", "False alarm": "tab:orange"}

    casts_to_plot = list(per_cast.index[:2]) + list(per_cast.index[-1:])

    for cast in casts_to_plot:
        one = test_df[test_df["file"] == cast]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # --- Top: pressure over time, coloured by outcome ---
        for name, color in colors.items():
            part = one[one["outcome"] == name]
            ax1.scatter(part["scan"], part["pressure"], s=3, color=color, label=name)
        ax1.invert_yaxis()                     # deeper = lower, like the real ocean
        ax1.set_ylabel("Pressure (dbar)")
        ax1.set_title(f"{cast}  |  recall = {per_cast[cast]:.2f}")
        ax1.legend(markerscale=4)

        # --- Bottom: anomaly score over time ---
        ax2.plot(one["scan"], one["score"], linewidth=0.8, color="black")
        ax2.axhline(threshold, color="tab:orange", linestyle="--", label="cut-off")
        # light red background = where IDX says "bad"
        ax2.fill_between(one["scan"], one["score"].min(), one["score"].max(),
                         where=(one["y_true"] == 1), color="tab:red", alpha=0.15,
                         label="IDX bad")
        ax2.set_ylabel("Anomaly score")
        ax2.set_xlabel("Scan number")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Cruise name = first 6 letters of the file name (e.g. "EMB311")
    test_df["cruise"] = test_df["file"].str[:6]

    print("\nPR-AUC per cruise:")
    for cruise, group in test_df.groupby("cruise"):
        # PR-AUC only works if the cruise has both good and bad rows
        if group["y_true"].nunique() == 2:
            pr = average_precision_score(group["y_true"], group["score"])
            print(cruise, "| PR-AUC:", round(pr, 3), "| casts:", group["file"].nunique())

    
