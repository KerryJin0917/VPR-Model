import pandas as pd
import os
import glob

# Path to your flattened GSV-Cities directory on /work
root = "/work/users/j/i/jinkerry/gsv-cities"
dataframe_dir = os.path.join(root, "Dataframes")

# 1. Find all city CSV files
csv_files = glob.glob(os.path.join(dataframe_dir, "*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {dataframe_dir}")

print(f"Combining {len(csv_files)} city dataframes...")

# 2. Load and concatenate all dataframes
all_dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(all_dfs, ignore_index=True)

# 3. Generate unique identities based on UTM coordinates
# Combining coordinates ensures each unique location has a unique ID
df['identity'] = df['utm_east'].astype(str) + "_" + df['utm_north'].astype(str)

# 4. Convert identities into numeric class labels
unique_locations = df['identity'].unique()
id_map = {loc: i for i, loc in enumerate(unique_locations)}
df['identity'] = df['identity'].map(id_map)

# 5. Save the final merged file as train.parquet in the root
train_path = os.path.join(root, "train.parquet")
df.to_parquet(train_path)

print(f"Done! Created merged file at: {train_path}")
print(f"Total images: {len(df)} | Total unique locations: {len(unique_locations)}")