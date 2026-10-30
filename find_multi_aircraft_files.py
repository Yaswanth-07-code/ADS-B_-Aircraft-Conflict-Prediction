"""
find_multi_aircraft_files.py
-----------------------------
Scans processed_data folders (train/ and test/) and reports which files
have 2 or more different aircraft. These are the files that can be used
for multi-aircraft conflict detection, Monte Carlo simulations, and the dashboard.
"""

import os
import glob

DATA_ROOT = r"C:\Users\HP\Downloads\14866251\7days4\7days4\processed_data"

def scan_folder(folder_path, folder_name="Folder"):
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    print(f"\nScanning {folder_name}: {len(txt_files)} files found in {folder_path}...")

    multi_files = []
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                ids = set()
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        ids.add(parts[1])
            num_aircraft = len(ids)
            if num_aircraft >= 2:
                multi_files.append((filename, num_aircraft, file_path))
                print(f"  [+] {filename}: {num_aircraft} aircraft")
        except Exception as e:
            print(f"  [!] {filename}: error reading - {e}")

    print("-" * 50)
    print(f"{folder_name} Summary: {len(multi_files)} / {len(txt_files)} files have 2+ aircraft")
    return multi_files

if __name__ == "__main__":
    print("=" * 60)
    print("ADS-B Dataset Multi-Aircraft Scanner")
    print("=" * 60)

    test_folder = os.path.join(DATA_ROOT, "test")
    train_folder = os.path.join(DATA_ROOT, "train")

    test_multi = scan_folder(test_folder, "TEST folder")
    train_multi = scan_folder(train_folder, "TRAIN folder")

    total_multi = len(test_multi) + len(train_multi)
    print("\n" + "=" * 60)
    print(f"TOTAL MULTI-AIRCRAFT SCENES: {total_multi} files")
    print("=" * 60)
    print("Top multi-aircraft scenes in TEST set:")
    for fn, count, fp in sorted(test_multi, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {fn} ({count} aircraft) -> {fp}")
