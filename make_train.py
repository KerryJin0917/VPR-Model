import pandas as pd
import os
from pathlib import Path

# Path to your dataset
root = "project-vpr/datasets/dataset_a"
test_parquet = os.path.join(root, "test.parquet")

# 1. Load the existing metadata
df = pd.read_parquet(test_parquet)

# 2. Generate unique identities based on UTM coordinates
# Since identity was all 0s, we combine UTM_East and UTM_North to make a unique location ID
df['identity'] = df['utm_east'].astype(str) + "_" + df['utm_north'].astype(str)

# 3. Convert these strings into numeric class labels for the model
unique_locations = df['identity'].unique()
id_map = {loc: i for i, loc in enumerate(unique_locations)}
df['identity'] = df['identity'].map(id_map)

# 4. Save as train.parquet
train_path = os.path.join(root, "train.parquet")
df.to_parquet(train_path)

print(f"Done! Created {train_path}")
print(f"Found {len(unique_locations)} unique places across {len(df)} images.")