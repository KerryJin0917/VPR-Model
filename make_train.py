import pandas as pd
import os
import glob

root = "/work/users/j/i/jinkerry/gsv-cities"
dataframe_dir = os.path.join(root, "Dataframes")
image_dir = os.path.join(root, "Images")

csv_files = glob.glob(os.path.join(dataframe_dir, "*.csv"))
all_dfs = []

for f in csv_files:
    city_df = pd.read_csv(f)
    city_id = str(city_df['city_id'].iloc[0])
    city_img_path = os.path.join(image_dir, city_id)

    if os.path.exists(city_img_path):
        # Map filenames to panoids: {panoid: full_filename}
        # We split by '_' and take the last part before '.jpg'
        files = os.listdir(city_img_path)
        file_map = {f.split('_')[-1].replace('.jpg', ''): f for f in files if f.endswith('.jpg')}

        # Create the image_path column by looking up the panoid in our map
        city_df['image_path'] = city_df['panoid'].map(lambda x: f"Images/{city_id}/{file_map[x]}" if x in file_map else None)
        all_dfs.append(city_df)

df = pd.concat(all_dfs, ignore_index=True)
initial_count = len(df)
df = df.dropna(subset=['image_path']) # Remove entries where image wasn't found

# Lat/Lon for identity
df['identity'] = df['lat'].astype(str) + "_" + df['lon'].astype(str)
unique_locs = df['identity'].unique()
df['identity'] = df['identity'].map({loc: i for i, loc in enumerate(unique_locs)})

df.to_parquet(os.path.join(root, "train.parquet"))
print(f"Done! Created train.parquet. Kept {len(df)}/{initial_count} images.")