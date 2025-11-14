from pathlib import Path
import csv

# This script assumes it lives in the same folder as your OOTP CSV files.
# It will:
# 1. List all CSV files in the folder
# 2. Ask you which one to preview
# 3. Print the header and first 10 data rows

BASE_DIR = Path(__file__).resolve().parent


def list_csv_files():
    csv_files = sorted(BASE_DIR.glob("*.csv"))
    return csv_files


def preview_csv(path: Path, max_rows: int = 10):
    print(f"\n=== Preview: {path.name} ===\n")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            print(row)
            if i >= max_rows:
                break


def main():
    csv_files = list_csv_files()

    if not csv_files:
        print("No CSV files found in this folder.")
        return

    print("CSV files in this folder:\n")
    for idx, path in enumerate(csv_files, start=1):
        print(f"{idx:2d}. {path.name}")

    # Ask you which file to preview
    print("\nType the number of the file you want to preview, then press Enter.")
    choice = input("> ").strip()

    if not choice.isdigit():
        print("That was not a number. Exiting.")
        return

    idx = int(choice)
    if idx < 1 or idx > len(csv_files):
        print("That number is out of range. Exiting.")
        return

    selected = csv_files[idx - 1]
    preview_csv(selected, max_rows=10)


if __name__ == "__main__":
    main()
