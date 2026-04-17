import os
import pandas as pd

ROOT = "/nas/longleaf/home/jinkerry/datasets/msls/train_val"

rows = []

for city in os.listdir(ROOT):
    city_path = os.path.join(ROOT, city)

    db_path = os.path.join(city_path, "database")
    q_path = os.path.join(city_path, "queries")

    if not os.path.exists(db_path):
        continue

    # -------------------
    # DATABASE
    # -------------------
    for fname in os.listdir(db_path):
        if fname.endswith((".jpg", ".png")):
            rows.append({
                "image_path": os.path.join(db_path, fname),
                "split": "database",
                "place_id": f"{city}_{fname.split('_')[0]}"
            })

    # -------------------
    # QUERIES
    # -------------------
    for fname in os.listdir(q_path):
        if fname.endswith((".jpg", ".png")):
            rows.append({
                "image_path": os.path.join(q_path, fname),
                "split": "query",
                "place_id": f"{city}_{fname.split('_')[0]}"
            })

df = pd.DataFrame(rows)
df.to_csv("msls_trainval_eval.csv", index=False)

print("Saved:", len(df), "images")