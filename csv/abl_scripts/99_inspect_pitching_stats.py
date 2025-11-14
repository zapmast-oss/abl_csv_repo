import pandas as pd

# Change this to match your actual file name/location
csv_path = "players_game_pitching_stats.csv"

# Load the CSV
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"❌ Could not find file: {csv_path}")
    exit()

# Inspect each column: show up to 5 unique values
print(f"✅ Loaded {csv_path} with {len(df)} rows\n")
print("=== Column Samples ===\n")
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    print(f"{col} ({len(unique_vals)} unique): {unique_vals[:5]}")
