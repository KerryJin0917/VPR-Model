import pandas as pd
import os
from pathlib import Path

# Updated Path: Points to the flattened directory on /work
root = "/work/users/j/i/jinkerry/gsv-cities"
# The script previously failed because it looked for test.parquet in a subfolder
# We now point directly to the parquet file in the root data folder
test_parquet = os.path.join(root, "test.parquet")

if not os.path.exists(test_parquet):
    raise FileNotFoundError(f"Could not find {test_parquet}. Ensure the file is in the GSV-Cities folder.")

# 1. Load the existing metadata
df = pd.read_parquet(test_parquet)

# 2. Generate unique identities based on UTM coordinates
# This creates a spatial 'class' for each unique coordinate pair
df['identity'] = df['utm_east'].astype(str) + "_" + df['utm_north'].astype(str)

# 3. Convert these strings into numeric class labels for the model
unique_locations = df['identity'].unique()
id_map = {loc: i for i, loc in enumerate(unique_locations)}
df['identity'] = df['identity'].map(id_map)

# 4. Save as train.parquet in the same /work directory
train_path = os.path.join(root, "train.parquet")
df.to_parquet(train_path)

print(f"Done! Created {train_path}")
print(f"Found {len(unique_locations)} unique places across {len(df)} images.")