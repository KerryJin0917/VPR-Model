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
all_dfs = []
for f in csv_files:
    temp_df = pd.read_csv(f)
    # Get the city name from the filename (e.g., Bangkok from Bangkok.csv)
    city_name = os.path.basename(f).replace('.csv', '')
    # Build the path using the actual folder name on disk
    temp_df['image_path'] = "Images/" + city_name + "/" + temp_df['panoid'].astype(str) + ".jpg"
    all_dfs.append(temp_df)
df = pd.concat(all_dfs, ignore_index=True)

# Construct the image_path column
# We combine the city name and the panoid (from your head command) to match the file structure
# Based on your file list: Images/CityName/panoid.jpg
df['image_path'] = "Images/" + df['city_id'].astype(str) + "/" + df['panoid'].astype(str) + ".jpg"

# 3. Generate unique identities based on Latitude and Longitude
# As seen in your head command: lat, lon are the correct column names
print("Using 'lat' and 'lon' columns for identity mapping.")
df['identity'] = df['lat'].astype(str) + "_" + df['lon'].astype(str)

# 4. Convert these strings into numeric class labels for the model
unique_locations = df['identity'].unique()
id_map = {loc: i for i, loc in enumerate(unique_locations)}
df['identity'] = df['identity'].map(id_map)

# 5. Save as train.parquet in the root directory
train_path = os.path.join(root, "train.parquet")
df.to_parquet(train_path)

print(f"Done! Created merged file at: {train_path}")
print(f"Total images processed: {len(df)}")
print(f"Total unique locations (classes): {len(unique_locations)}")