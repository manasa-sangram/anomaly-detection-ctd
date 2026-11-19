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

import sklearn
from visualize_data import dataframe_access
from ctdam.entry.casts import Casts
from visualize_data import read_idx_ranges

# print(sklearn.__version__)

if __name__ == "__main__":
    idx_files ={}
    # DATA_FILE = Path("ctd_data.pkl")
    
    # if not DATA_FILE.exists():
    #     raise FileNotFoundError("ctd_data.pkl not found. Run save_data.py first!")
    
    # df = pd.read_pickle(DATA_FILE)
    df = dataframe_access()
    print("\nDataframe Loaded:", df.shape)

    DATA_DIR = Path(r"C:\Users\Manasa\OneDrive\Desktop\Anamoly Detection in CTD data\CTD_DATA_CORRECT")


    for root, dirs, files in os.walk(DATA_DIR):
            for filename in files:
                if filename.lower().endswith(".idx"):
                    idx_path = Path(root) / filename
                    idx_files[idx_path.stem] = idx_path

    # def read_idx_ranges(idx_path):
    #     ranges = []
    #     after_end = False

    #     with open(idx_path, "r", encoding="utf-8",errors="replace") as f:
    #         for line in f:
    #             line = line.strip()

    #             if line == "*END*":
    #                 after_end = True
    #                 continue

    #             if not after_end or not line:
    #                 continue

    #             parts = line.split()

    #             if len(parts) >= 4:
    #                 start = int(parts[0])
    #                 end = int(parts[1])
    #                 delete = int(parts[2])
    #                 interpolate = int(parts[3])

    #                 ranges.append({
    #                 "start": start,
    #                 "end": end,
    #                 "delete": delete,
    #                 "interpolate": interpolate
    #             })
    #     return ranges


# Test using one IDX file
    # example_idx = idx_files["EMB311_001-01_CTD_0001"]
    # ranges = read_idx_ranges(example_idx)
    # print("\n-------------IDX annotations from 1 IDX file---------------" , ranges)


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

    
#========================================================
# Features for Isolation Forest
#=========================================================

    # df = df.replace(-9.99e-29 , np.nan)
    # df.to_parquet("all_casts.parquet", index=False)

    sensor_candidates = ["pressure","temperature","conductivity"]

    sensor_cols = [
    col for col in sensor_candidates
    if col in df.columns
]
    print("\nAvailable sensor columns:", sensor_cols)

    # for col in sensor_cols:
    #     col_features ={col}
    #     print("Features : " , col)
        


# ========================================================
# Features used by Isolation Forest
# ========================================================



    temperature_features = ["temperature" ]

    conductivity_features =["conductivity" ]

    pressure_features =["pressure" ]

    print("\nTemperature Features ", temperature_features)
    print("\nConductivity Features used:", conductivity_features)
    print("\nPressure Features used:", pressure_features)
    


# ===========================================================
# Train / validation /test split by complete CTD cast
# ============================================================

    files = df["file"].unique()

    rng = np.random.default_rng(42)
    shuffled_files = rng.permutation(files)

    n=len(shuffled_files)

    n_train = int(0.80 *n)

  
    train_files = shuffled_files[:n_train]
    test_files = shuffled_files[n_train:]

    # Select all rows belonging to each file
    train = df[df["file"].isin(train_files)].copy()
    test = df[df["file"].isin(test_files)].copy()

    # Check that no casts overlap
    assert set(train["file"]).isdisjoint(set(test["file"]))

    print("\nTraining casts:", train["file"].nunique())
    print("Test casts:", test["file"].nunique())

    # print("\nTraining rows:", len(train))
    # print("Test rows:", len(test))


# ===========================================================
# Model Training : SUPERVISED AND UNSUPERVISED
# ============================================================
    # valid= df[features].notna().all(axis=1)
    APPROACH =["supervised" , "unsupervised"]
    if APPROACH == "supervised":
        train_mask=(df["file"].isin(train_files) & (df["y_true"] == 0) )
        test_mask  = (df["file"].isin(test_files) & df["y_true"].isin([0, 1]) )
    elif APPROACH == "unsupervised":
        all_rows = df["y_true"].isin([0, 1])
        train_mask = all_rows
        test_mask  = all_rows
    else:
        raise ValueError("Choose an approach : Supervised or Unsupervised")

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

# train 3 isolation models
    models ={}

    features_dict ={
        "temperature" : temperature_features , 
        "conductivity": conductivity_features , 
        "pressure" : pressure_features
    }
    modes = ["supervised", "unsupervised"]

    for mode in modes:
        for sensor, features in features_dict.items():

            print(f"\nTraining {mode} {sensor} model")

            if mode =="supervised":
                train_rows = train[train["y_true"]==0]
            else:
                train_rows =df

        X_train = train_rows[features]
        print("Training rows :" , len(X_train))


        model = IsolationForest(
        n_estimators=700,
        max_samples=2048,
        contamination="auto",
        random_state=42,
        n_jobs=-1 )

        model.fit(X_train)
        models[(mode ,sensor)] = model

        print(f"{mode}{sensor} model trained successfully")

    
    print("\nIsolation Forest training complete!")
    

    
 # ============================================================
 # Calculate Anomaly scores
# ============================================================

    results = []

    curves ={}

    test_eval_all = test[test["y_true"].isin([0, 1])].copy()

    if test_eval_all.empty:
        raise ValueError("No test rows have valid IDX labels")

    for (mode ,sensor), model in models.items():
        print(f"\n========== {mode.upper()} -{sensor.upper()} RESULTS ==========")

        features = features_dict[sensor]

        X_test = test[features]

        # Keep only rows with valid IDX labels
        test_eval = test_eval_all

        if test_eval.empty:
            raise ValueError("No labelled test rows with this sensor.")
            continue

        X_eval = test_eval[features]
        y_test = test_eval["y_true"].to_numpy()

        
    # Higher score = more anomalous
        anomaly_scores = -model.decision_function(X_eval)

        if len(np.unique(y_test)) == 2:

            pr_auc = average_precision_score(
            y_test, anomaly_scores
        )

            roc_auc = roc_auc_score(
            y_test, anomaly_scores
        )

        else:
            pr_auc = np.nan
            roc_auc = np.nan

        print("Test casts:", test_eval["file"].nunique())
        print("Labelled test rows:", len(test_eval))
        print("Actual bad-row share:", y_test.mean())
        print("PR-AUC:", pr_auc)
        print("ROC-AUC:", roc_auc)

        results.append({
            "mode" : mode ,
        "sensor": sensor,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "labelled_rows": len(test_eval),
        "bad_row_share": y_test.mean()
    })

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========")
    
    print(results_df.to_string(index=False))

    print("\n========== PR-AUC SUMMARY ==========")
    print(results_df.pivot(index="sensor", columns="mode", values="PR_AUC"))

    
# ============================================================
# Plot Precision-Recall curves for all 3 models
# ============================================================

    colors = {
        "temperature": "tab:red",
        "conductivity": "tab:blue",
        "pressure": "tab:green"
    }

    fig , axes = plt.subplots(1 , 2 , figsize =(14 , 6) , sharey=True)

    for ax , mode in zip(axes , modes):
        for sensor in features_dict:

            if (mode , sensor) not in curves:
                continue

            y_test , anomaly_scores , pr_auc = curves[(mode, sensor)]

            precision , recall , thresholds = precision_recall_curve(y_test , anomaly_scores)

            ax.plot(
                recall,
                precision,
                color=colors[sensor],
                label=f"{sensor.capitalize()} (PR_AUC = {pr_auc:.3f})"
            )

            baseline = y_test.mean()
            ax.axhline(
                y=baseline,
                color=colors[sensor],
                linestyle="--",
                alpha=0.6,
                label=f"{sensor.capitalize()} random baseline ({baseline:.3f})"
            )
        ax.set_xlabel("Recall")
        ax.set_title(f"{mode.capitalize()} individual Isolation Forests")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Precision")
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(9, 6))

#     for sensor, model in models.items():

#         features = features_dict[sensor]

#     # Keep only rows with valid IDX labels
#         test_eval = test[test["y_true"].isin([0, 1])].copy()

#         X_eval = test_eval[features]
#         y_test = test_eval["y_true"].to_numpy()

#     # Higher score means more anomalous
#         anomaly_scores = -model.decision_function(X_eval)

#     # Calculate precision and recall at different thresholds
#         precision, recall, thresholds = precision_recall_curve(
#         y_test, anomaly_scores
#         )

#     # Calculate Average Precision
#         pr_auc = average_precision_score(
#         y_test, anomaly_scores
#     )

#     # Plot the curve
#         plt.plot(
#         recall,
#         precision, color ="red" ,
#         label=f"{sensor.capitalize()} (PR_AUC = {pr_auc:.3f})"
#         )

# # Random-ranking baseline
#     baseline = test_eval["y_true"].mean()
#     plt.axhline(
#     y=baseline,
#     linestyle="--",
#     label=f"Random baseline ({baseline:.3f})"
# )

#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curves: Sensor-Specific Isolation Forests")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


