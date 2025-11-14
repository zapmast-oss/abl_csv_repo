import pandas as pd
import glob

# Look at every CSV file in this folder
for path in glob.glob("*.csv"):
    try:
        # Read just the header row (nrows=0) to get column names
        df = pd.read_csv(path, nrows=0)
        print("FILE:", path)
        print("COLUMNS:", list(df.columns))
        print("-" * 60)
    except Exception as e:
        print("FILE:", path)
        print("ERROR READING:", e)
        print("-" * 60)
