# z_abl_pregame_pack.py CLI Options

```
usage: z_abl_pregame_pack.py [-h] [--base BASE] [--season SEASON]
                             [--week WEEK] [--league_id LEAGUE_ID]
                             [--matchups MATCHUPS] [--date DATE]
                             [--arsenal-top ARSENAL_TOP]
                             [--show-arsenal-count] [--bats-top BATS_TOP]

Generate ABL pregame pack.

options:
  -h, --help            show this help message and exit
  --base BASE           Repo root (optional).
  --season SEASON
  --week WEEK
  --league_id LEAGUE_ID
  --matchups MATCHUPS   Explicit matchups list, e.g., CHI@MIA,DEN@NAS
  --date DATE           Game date (YYYY-MM-DD) to derive matchups from
                        schedule/games
  --arsenal-top ARSENAL_TOP
                        Top N pitches to display for arsenal
  --show-arsenal-count  Show pitch count when available
  --bats-top BATS_TOP   Top N key bats per team
```
