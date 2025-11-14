import pandas as pd

print("Script is running.")

teams = pd.read_csv("teams.csv")

print(teams.head(10).to_string(index=False))
