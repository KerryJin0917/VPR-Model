#!/usr/bin/env python3

import os
import pandas as pd


def build_msls_csv(root, output_csv="msls_eval.csv"):
    """
    Converts MSLS folder structure into a flat CSV:

    Expected structure:
    root/
        city_1/
            database/
            queries/
        city_2/
            database/
            queries/
    """

    rows = []

    cities = [
        c for c in os.listdir(root)
        if os.path.isdir(os.path.join(root, c))
    ]

    print(f"Found {len(cities)} cities")

    for city in cities:
        city_path = os.path.join(root, city)

        db_path = os.path.join(city_path, "database")
        q_path = os.path.join(city_path, "queries")

        if not os.path.exists(db_path) or not os.path.exists(q_path):
            print(f"Skipping {city} (missing database/queries)")
            continue

        # -------------------------
        # DATABASE IMAGES
        # -------------------------
        for fname in os.listdir(db_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                rows.append({
                    "image_path": os.path.join(db_path, fname),
                    "split": "database",
                    "city": city,
                    "place_id": f"{city}_{fname.split('_')[0]}"
                })

        # -------------------------
        # QUERY IMAGES
        # -------------------------
        for fname in os.listdir(q_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                rows.append({
                    "image_path": os.path.join(q_path, fname),
                    "split": "query",
                    "city": city,
                    "place_id": f"{city}_{fname.split('_')[0]}"
                })

    df = pd.DataFrame(rows)

    # sanity checks
    print("\nDataset summary:")
    print(df["split"].value_counts())
    print(df["city"].value_counts().head())

    df.to_csv(output_csv, index=False)

    print(f"\nSaved CSV → {output_csv}")
    print(f"Total images: {len(df)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output", type=str, default="msls_eval.csv")

    args = parser.parse_args()

    build_msls_csv(args.root, args.output)