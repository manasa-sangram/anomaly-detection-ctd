# ========================================================
# FAIR COMPARISON: 3 FEATURES vs 6 FEATURES
# ========================================================

#     base_features = [
#     "temperature",
#     "conductivity",
#     "pressure"
# ]
#     engineered_features = base_features + [
#     "temperature_diff",
#     "conductivity_diff",
#     "pressure_diff"
# ]
#     common_valid = df[engineered_features].notna().all(axis=1)
#     train_mask = (
#     df["file"].isin(train_files)
#     & (df["y_true"] == 0)
#     & common_valid
# )
#     test_mask = (
#     df["file"].isin(test_files)
#     & df["y_true"].isin([0, 1])
#     & common_valid
# )

#     y_test = df.loc[test_mask, "y_true"]
#     print("Common training rows:", train_mask.sum())
#     print("Common testing rows:", test_mask.sum())
#     print("Testing labels:")
#     print(y_test.value_counts())

#     # Prepare both datasets
#     X_train_base = df.loc[train_mask, base_features]
#     X_test_base = df.loc[test_mask, base_features]

#     X_train_eng = df.loc[train_mask, engineered_features]
#     X_test_eng = df.loc[test_mask, engineered_features]

#     model_base = IsolationForest(
#     n_estimators=700,
#     max_samples=2048,
#     contamination="auto",
#     random_state=42,
#     n_jobs=-1
# )
#     model_base.fit(X_train_base)
#     scores_base = -model_base.decision_function(X_test_base)

#     model_eng = IsolationForest(
#     n_estimators=700,
#     max_samples=2048,
#     contamination="auto",
#     random_state=42,
#     n_jobs=-1
# )
#     model_eng.fit(X_train_eng)
#     scores_eng = -model_eng.decision_function(X_test_eng)

#     print("\n===== FAIR MODEL COMPARISON =====")

#     print(
#     "Baseline PR-AUC:",
#     round(average_precision_score(y_test, scores_base), 6)
# )
#     print(
#     "Engineered PR-AUC:",
#     round(average_precision_score(y_test, scores_eng), 6)
# )

#     print(
#     "Baseline ROC-AUC:",
#     round(roc_auc_score(y_test, scores_base), 6)
# )

#     print(
#     "Engineered ROC-AUC:",
#     round(roc_auc_score(y_test, scores_eng), 6)
# )



# # Use the same valid rows for every experiment
#     common_valid = df[all_features].notna().all(axis=1)

#     train_mask = (
#     df["file"].isin(train_files)
#     & (df["y_true"] == 0)
#     & common_valid
# )

#     test_mask = (
#     df["file"].isin(test_files)
#     & df["y_true"].isin([0, 1])
#     & common_valid
# )

# # Same training and testing data for all models
#     y_test = df.loc[test_mask, "y_true"]

# # Define feature combinations
#     experiments = {
#     "Full model (6 features)": all_features,

#     "Without temperature_diff": [
#         "temperature", "conductivity", "pressure",
#         "conductivity_diff", "pressure_diff"
#     ],

#     "Without conductivity_diff": [
#         "temperature", "conductivity", "pressure",
#         "temperature_diff", "pressure_diff"
#     ],

#     "Without pressure_diff": [
#         "temperature", "conductivity", "pressure",
#         "temperature_diff", "conductivity_diff"
#     ]
# }

# ==========================================
# EXPERIMENT 1: FEATURE ABLATION
# ==========================================

#     results = []

#     for experiment_name, features in experiments.items():

#         X_train = df.loc[train_mask, features]
#         X_test = df.loc[test_mask, features]

#         model = IsolationForest(
#         n_estimators=700,
#         max_samples=2048,
#         contamination="auto",
#         random_state=42,
#         n_jobs=-1
#     )

#         model.fit(X_train)

#         anomaly_scores = -model.decision_function(X_test)

#         pr_auc = average_precision_score(y_test, anomaly_scores)
#         roc_auc = roc_auc_score(y_test, anomaly_scores)

#         results.append({
#         "Experiment": experiment_name,
#         "Features": len(features),
#         "PR_AUC": pr_auc,
#         "ROC_AUC": roc_auc
#     })

#     results_df = pd.DataFrame(results)

#     print("\n========== FEATURE ABLATION RESULTS ==========")
#     print(results_df.to_string(index=False))


 # ===========================================================
# EXPERIMENT 2: ROLLING FEATURE ABLATION
# ===========================================================

#     base_features = [
#     "temperature",
#     "conductivity",
#     "pressure",
#     "temperature_diff",
#     "pressure_diff"
# ]

#     rolling_features = [
#     "temperature_local_median",
#     "conductivity_local_median",
#     "pressure_local_median"
# ]

# # All eight features must be valid.
# # This ensures every model uses exactly the same rows.

#     all_features = base_features + rolling_features

#     common_valid = df[all_features].notna().all(axis=1)

#     train_mask = (
#     df["file"].isin(train_files)
#     & (df["y_true"] == 0)
#     & common_valid
# )

#     test_mask = (
#     df["file"].isin(test_files)
#     & df["y_true"].isin([0, 1])
#     & common_valid
# )

#     y_test = df.loc[test_mask, "y_true"]

# # Define experiments

#     experiments = {
#     "Baseline (5 features)": base_features,

#     "Baseline + pressure_local_median":
#         base_features + ["pressure_local_median"] , 

#     "Without pressure diff": ["temperature", "conductivity", "pressure",
#         "temperature_diff", "conductivity_diff", "pressure_local_median"],

#     "Pressure features only":["pressure", "pressure_diff", "pressure_local_median"]
# }

#     results = []

#     for experiment_name, selected_features in experiments.items():

#         X_train = df.loc[train_mask, selected_features]
#         X_test = df.loc[test_mask, selected_features]

#         model = IsolationForest(
#         n_estimators=700,
#         max_samples=2048,
#         contamination="auto",
#         random_state=42,
#         n_jobs=-1
#     )

#         model.fit(X_train)

#         anomaly_scores = -model.decision_function(X_test)

#         pr_auc = average_precision_score(
#         y_test, anomaly_scores
#     )

#         roc_auc = roc_auc_score(
#         y_test, anomaly_scores
#     )

#         results.append({
#         "Experiment": experiment_name,
#         "PR_AUC": pr_auc,
#         "ROC_AUC": roc_auc
#     })

#     results_df = pd.DataFrame(results)

#     print("\n========== ROLLING FEATURE ABLATION ==========")
#     # print(results_df.to_string(index=False)



# ========================================================
# Sliding-window median imputation and Linear Interpolation
# ========================================================
#     IMPUTATION_METHOD = "linear_interpolation"

#     if IMPUTATION_METHOD == "local_median":

#         print("\nExperiment 1: Sliding-window local median")
#         for col in sensor_cols:

#             df[col] = (
#                 df.groupby("file")[col].transform(
#                 lambda s: s.fillna(
#                     s.rolling(
#                       window=11,
#                       center=True,
#                       min_periods=3
#                   ).median())))

            
#     elif IMPUTATION_METHOD == "linear_interpolation":
#         print("\nExperiment 2: Linear interpolation")

#         for col in sensor_cols:
#             df[col] = (
#                 df.groupby("file", sort=False)[col]
#                 .transform(
#                     lambda s: s.interpolate(
#                         method="linear",
#                         limit_area="inside"
#                 )))

#     else:
#             raise ValueError("Unknown imputation method")

# # ========================================================
# # Check missing values after imputation
# # ========================================================

#     print("\nMissing values after imputation:")

#     print(df[sensor_cols].isna().sum())


#     g= df.groupby("file" , sort= False)
