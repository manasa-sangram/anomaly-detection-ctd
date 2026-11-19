from importlib.resources import path
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ctdam.entry.casts import Casts
from ctdam import parse
import plotly.express as px

DATA_DIR = Path(r"C:\Users\Manasa\OneDrive\Desktop\Anamoly Detection in CTD data\CTD_DATA_CORRECT")

sensor_cols = ["pressure", "temperature", "conductivity"]

def dataframe_access():
         
        casts = Casts(path_to_data=DATA_DIR,pattern=".cnv",)
        frames =[]

        for file_path in DATA_DIR.rglob("*.cnv"):
            try:
                ds = parse(file_path)
                data = {}
                for col in sensor_cols:
                    values = ds[col]
                    if "sensor" in values.dims:                # cast has 2 sensors
                        values = values.sel(sensor="primary")  # keep primary only
                    data[col] = values.values

                    qc_name = f"{col}_qc"
                    if qc_name in ds:
                        qc = ds[qc_name]
                        if "sensor" in qc.dims:
                            qc = qc.sel(sensor="primary")
                        data[qc_name] = qc.values

                df_cast = pd.DataFrame(data)
                df_cast["scan"] = range(len(df_cast))

                # print("OK" , file_path.name)
                # df_cast = ds[sensor_cols].to_dataframe().reset_index()
                # df_cast[sensor_cols] = df_cast[sensor_cols].replace([-9.99e-29, -9.99e-28],float("nan"))

                # Safety check: stops the run if rows and scans ever differ again
                assert len(df_cast) == ds.sizes["scan"], f"{file_path.name}: rows != scans"

                for col in sensor_cols:
                    tiny =(df_cast[col] !=0) & (df_cast[col].abs() < 1e-20)
                    df_cast.loc[tiny , col] = float("nan")

                # df_cast[sensor_cols] = df_cast[sensor_cols].replace(
                # [-9.99e-29, -9.99e-28], float("nan"))

            except Exception as e :
                print("FAILED " , file_path.name)
                print("Error" , e)
                continue

            df_cast["file"] = file_path.name
            frames.append(df_cast)
          
        if not frames:
             raise ValueError("No CTD dataframes were created")
        
     #Combine the DataFrames
        df= pd.concat(frames , ignore_index= True)
        return df

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

def add_idx_labels(df , cast_name = None):
    #   file_name ="EMB323_006-01_CTD_0009"
      df = df.copy()
      df["y_true"] =0

      idx_files = {}

      for path in DATA_DIR.rglob("*.idx"):

        # print("Current IDX file:", path)
        # print("Filename without extension:", path.stem)

        idx_files[path.stem] = path

    #   idx_files = {path.stem: path
    #     for path in DATA_DIR.rglob("*.idx")
    # }

      cast_groups = {
        Path(filename).stem: indices
        for filename, indices in df.groupby(
            "file", sort=False
        ).groups.items()
    }
      matched_count = 0

      for name, idx_path in idx_files.items():
        if cast_name is not None and name != cast_name:
            continue
        indices = cast_groups.get(name)

        if indices is None:
            continue

        ranges = read_idx_ranges(idx_path)
        labels = [0] * len(indices)

        for item in ranges:
            if item["delete"] == 1:
                start = max(0, item["start"] - 1)
                end = min(len(labels), item["end"])

                if start < end:
                    labels[start:end] = [1] * (end - start)

                print(
                    f"IDX range: {item['start']} - {item['end']}"
                    f" | Python: {start} - {end - 1}"
                    f" |No of rows {start} - {end - 1} = {end - start}"
                )


        df.loc[indices, "y_true"] = labels
        matched_count += 1

      print("\nMatched IDX casts in 1 file", matched_count)
      print("\nIDX Ground Truth in 1 file")
      print(df["y_true"].value_counts().sort_index())

      return df


def plot_idx_labels(df):

    # Keep only labelled observations
    plot_df = df[df["y_true"].isin([0, 1])].copy()

    # Sample equal numbers of good and bad observations
    good = plot_df[plot_df["y_true"] == 0].sample(
        n=min(10000, (plot_df["y_true"] == 0).sum()),
        random_state=42
    )

    bad = plot_df[plot_df["y_true"] == 1].sample(
        n=min(10000, (plot_df["y_true"] == 1).sum()),
        random_state=42
    )

    plot_df = pd.concat([good, bad])

    plot_df["label"] = plot_df["y_true"].map({
        0: "Good",
        1: "Bad"
    })

    # Three sensor combinations
    feature_pairs = [
        ("temperature", "conductivity"),
        ("temperature", "pressure"),
        ("conductivity", "pressure")
    ]

    for x_col, y_col in feature_pairs:

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="label",
            symbol="label",
            symbol_map={
                "Good": "circle",
                "Bad": "triangle-up"
            },
            render_mode="webgl",
            hover_data=["file", "y_true"],
            title=f"IDX Ground Truth: {x_col} vs {y_col}"
        )

        fig.update_layout(
            width=1000,
            height=700,
            legend_title="IDX Label"
        )

        fig.show()


                 

     

if __name__ == "__main__":
    df = dataframe_access()
    rows_per_file = df.groupby("file").size()
    print("EMB311 cast rows:", rows_per_file["EMB311_001-01_CTD_0001.cnv"], "(expected 1617)")
    print("Unique file names:", df["file"].nunique(), "of 1215 files")
    all_files = list(DATA_DIR.rglob("*.cnv"))
    print("cnv files on disk:", len(all_files))
    print("unique names on disk:", len({f.name for f in all_files}))
    idx_files ={}
    # sensor_candidates = ["pressure","temperature","conductivity",]
    # sensor_cols = [
    #         col for col in sensor_candidates
    #         if col in df.columns]
    

    

    #Convert each cast into a DataFrame
#     for cast in casts.data:
#       df_cast = cast.to_dataframe().copy()
#       df_cast["file"] =Path(cast.attrs["path_to_source_file"]).name
#       frames.append(df_cast)

# #Combine the DataFrames
#     

    print("Data Shape" , df.shape)
    print(df.head())
    print("\nMissing values in each sensor")
    print(df[["pressure" , "temperature" , "conductivity"]].isna().sum())

    print("\nMissing percentages:")
    print((df[["pressure" , "temperature" , "conductivity"]].isna().mean() *100).round(2))

    # sensor_candidates = [
    #     "pressure",
    #     "temperature",
    #     "conductivity",]

    # sensor_cols = [
    #     col for col in sensor_candidates
    #     if col in df.columns]

    print("Available sensor columns:", sensor_cols)


# ========================================================
# VISUALIZATION: CHECK MISSING VALUES AND RAW CTD DATA
# ========================================================
    VISUALIZATION_ONLY = True


# ========================================================
# Plot 1: Missing values in each sensor
# ========================================================
    
    missing_counts = df[sensor_cols].isna().sum()
    missing_percentage = df[sensor_cols].isna().mean() * 100

    missing_summary = pd.DataFrame({
    "Missing values" : missing_counts , 
    "Missing Percentage" : missing_percentage
    })

    print("\n========== MISSING VALUE SUMMARY ==========")
    print(missing_summary.round(2))

    plt.figure(figsize= (9,5))

    
    
    sns.barplot(
        x = missing_summary.index , 
        y = missing_summary["Missing Percentage"]
    )

    plt.ylabel("Missing Values (%)")
    plt.xlabel("Sensor Type")
    plt.title("Percentage of Missing Values by Sensor")
    plt.xticks(rotation =45)
    plt.tight_layout()
    plt.show()

    print("\nMin / Max of each sensor")
    print(df[sensor_cols].describe().loc[["min", "max"]])

    # Which QC flag values actually appear?
    for col in sensor_cols:
        if f"{col}_qc" in df.columns:
            print(f"\n{col}_qc values:")
            print(df[f"{col}_qc"].value_counts().sort_index())

# ========================================================
# Plot 2: Missing values in each CTD cast
# ========================================================

    missing_by_cast = (
        df.groupby("file", sort=False)[sensor_cols]
        .apply(lambda group: group.isna().mean() * 100)
    )

    # Show the 30 casts with the highest average missingness
    missing_by_cast["average_missing"] = (
        missing_by_cast.mean(axis=1)
    )
    missing_by_cast = missing_by_cast.sort_values(
        "average_missing",
        ascending=False
    )

    print("\n========== CASTS WITH MOST MISSING VALUES ==========")
    print(missing_by_cast.head(10).round(2))

    plt.figure(figsize=(12, 7))
    sns.heatmap(
    missing_by_cast.head(30)[sensor_cols],
    cmap="viridis",
    vmin=0,
    vmax=100
    )

    plt.xlabel("Sensor Type")
    plt.ylabel("CTD cast")
    plt.title("Missing Values (%) in the 30 Most Incomplete Casts")
    plt.tight_layout()
    plt.show()

# ========================================================
# Plot 3: Raw sensor values in one CTD cast
# ========================================================
    sensor_cols = ["pressure", "temperature", "conductivity"]


    target = DATA_DIR /"EMB295"/"cnv"/"EMB295_013-03_CTD_0036.cnv"

    ds= parse(target)
    print(ds)
    print("\nDataset dimensions" , ds.sizes)
    print("Temperature dim" , ds["temperature"].dims)
    print("\nSensor Types" , ds["sensor"].values)

    test_df = ds[sensor_cols].to_dataframe().reset_index()
    print("Parsed scans:", ds.sizes["scan"])
    print("\ndataframe rows " , len(test_df))
    print(test_df[["pressure" , "temperature" , "conductivity"]].head())
   

    fig, axes = plt.subplots(
            len(sensor_cols), 
            1,
            figsize=(12, 8),
            squeeze=False
        )
    scan_numbers = range(ds.sizes["scan"])

    for i, col in enumerate(sensor_cols):
            # Convert this sensor's values to a one-dimensional array
            values = ds[col].values
            axes[i, 0].plot(
                scan_numbers,
                values ,
                linewidth=0.8 , color = 'red'
            )

            axes[i, 0].set_ylabel(col.capitalize())
            axes[i, 0].set_xlabel("Scan number")
            axes[i, 0].set_title(f"{col.capitalize()} — {target.stem}")

    plt.tight_layout()
    plt.show()

    #========================================================
# Plot 4: Good vs Bad observations using IDX labels
# ========================================================
    df = add_idx_labels(df , cast_name="EMB295_013-03_CTD_0036")
    plot_df = df[
    df["file"].apply(
        lambda x: Path(x).stem == "EMB295_013-03_CTD_0036"
    )
    ].copy()
    plot_idx_labels(plot_df)

      