import os
import re
import sys
import numpy as np
from glob import glob

def extract_episode_ids(base_dir, identifier_1, identifier_2):
    all_episode_ids = []

    # Find all run directories matching identifier_1
    run_dirs = glob(os.path.join(base_dir, f"{identifier_1}*/states"))

    for run_path in run_dirs:
        # Find files matching identifier_2 followed by digits and ending in .npy
        pattern = os.path.join(run_path, f"{identifier_2}[0-9]*.npy")
        files = glob(pattern)

        if not files:
            continue  # skip this run if no matching file found

        # Extract episode IDs from filenames
        episode_ids = []
        for f in files:
            filename = os.path.basename(f)
            match = re.match(rf"{re.escape(identifier_2)}(\d+)", filename)
            if match:
                episode_ids.append(int(match.group(1)))

        if episode_ids:
            all_episode_ids.extend(episode_ids)

    return all_episode_ids

def print_statistics(episode_ids):
    if not episode_ids:
        print("No matching episode IDs found.")
        return

    arr = np.array(episode_ids)
    print(f"Number of episode IDs: {len(arr)}")
    print(f"Min: {arr.min():.2f}")
    print(f"Max: {arr.max():.2f}")
    print(f"Mean: {arr.mean():.2f}")
    print(f"Median: {np.median(arr):.2f}")
    print(f"Sample Std Dev: {arr.std(ddof=1):.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_episodes.py <identifier_1> <identifier_2>")
        sys.exit(1)

    identifier_1 = sys.argv[1]
    identifier_2 = sys.argv[2]
    base_dir = "run_logs"

    episode_ids = extract_episode_ids(base_dir, identifier_1, identifier_2)
    print_statistics(episode_ids)
