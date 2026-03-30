"""Create a tiny sample dataset (CSV + HDF5) for quick local testing."""
import argparse
import numpy as np
import pandas as pd
import h5py
import os


def make_sample(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # create train and val CSVs
    train = pd.DataFrame([
        {"question": "What does the person pick up?", "video_id": "video1", "choices": "[]", "label": 0},
        {"question": "Why does the car stop?", "video_id": "video2", "choices": "[]", "label": 1},
    ])
    val = pd.DataFrame([
        {"question": "Where does the ball go?", "video_id": "video1", "choices": "[]", "label": 0},
    ])
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)

    # create small HDF5 files with (T, N, F) arrays
    for split in ("app_mot_train.h5", "app_mot_val.h5"):
        path = os.path.join(out_dir, split)
        with h5py.File(path, "w") as f:
            f.create_dataset("video1", data=np.random.randn(12, 2, 64).astype("float32"))
            f.create_dataset("video2", data=np.random.randn(10, 2, 64).astype("float32"))

    # map file
    import json

    map_path = os.path.join(out_dir, "map_vid_vidorID.json")
    with open(map_path, "w") as j:
        json.dump({"video1": "video1", "video2": "video2"}, j)

    print("Sample data created in", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets/sample")
    args = p.parse_args()
    make_sample(args.out)


if __name__ == "__main__":
    main()
