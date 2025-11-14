import pandas as pd

# Load the file where you expect player_id 5493 to exist
df = pd.read_csv("players.csv")  # Change filename if needed

# Find the row with player_id == 5493
player = df[df["player_id"] == 5493]

# Print result
if player.empty:
    print("Player ID 5493 not found.")
else:
    print(player.to_string(index=False))
