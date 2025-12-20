League Pregame Dash (see boards below)
Arsenal top: 3, Bats top: 2

# ABL Pregame Pack - Season N/A Week N/A

## CHI at MIA
CHI (17-15, rank 2, GB 2) @ MIA (18-14, rank 2, GB 4)
Ballpark: Hurricanes Park | Park Env: Hitter (AVG 0.9758, HR 1.109)
Probable Starters: CHI John Jury (games.csv) vs MIA Bill Borden (games.csv)
### Managers
CHI: Matt Mead | 761-697 (0.522) | Titles 0 | Tendencies: Smallball: Swing Away | Hook: Standard | Platoon: Static
MIA: Seth Coe | 794-664 (0.545) | Titles 2 | Tendencies: Smallball: Balanced | Hook: Standard | Platoon: Static
CHI Park: Chicago Grounds (Cap: 18,500)
CHI Finances: Budget $10,700,000 | Payroll $8,400,000 | Cash $-920,321 | Revenue $7,500,000 | Balance $3,026,890 | Tier: Mid
CHI Fans/Market: Market 6 | Loyalty 7 | Att 1,832,310 (season) | Ticket $3.98 | Gate $3.5m
MIA Park: Hurricanes Park (Cap: 25,800)
MIA Finances: Budget $9,500,000 | Payroll $7,700,000 | Cash $-27,654 | Revenue $10,200,000 | Balance $5,426,671 | Tier: Mid
MIA Fans/Market: Market 7 | Loyalty 6 | Att 1,499,048 (season) | Ticket $4.69 | Gate $3.6m
### Key Bats
CHI: N/A (no batter profile source found)
MIA: N/A (no batter profile source found)
### Pitching Snapshot
CHI starter: John Jury
Arsenal (4): Curveball (best) 50, Changeup 50, Fastball 40
Best pitch: Curveball 50
MIA starter: Bill Borden
Arsenal (3): Changeup (best) 60, Slider 55, Fastball 50
Best pitch: Changeup 60
Arsenal sources: C:\sbv_repo\abl_csv_repo\csv\out\text_out\prep\pitcher_arsenal_all.txt
Starter source: games.csv starters where present; otherwise projected_starting_pitchers.csv

## Data Sources
- C:\sbv_repo\abl_csv_repo\csv\out\star_schema\dim_team_park.csv
- C:\sbv_repo\abl_csv_repo\csv\out\csv_out\z_ABL_DIM_Ballparks.csv
- C:\sbv_repo\abl_csv_repo\csv\out\csv_out\z_ABL_Matchup_History.csv
- C:\sbv_repo\abl_csv_repo\csv\out\star_schema\dim_player_pitching_ratings.csv
- C:\sbv_repo\abl_csv_repo\csv\out\star_schema\dim_player_batting_ratings.csv
- C:\sbv_repo\abl_csv_repo\csv\out\star_schema\fact_team_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_last_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\out\star_schema\fact_team_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_last_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\ootp_csv\team_history_financials.csv
- C:\sbv_repo\abl_csv_repo\csv\out\text_out\prep\pitcher_arsenal_all.txt

## Notes
- fact_team_financials columns: {'budget': 'Bgt', 'payroll': 'Pay', 'cash': None, 'revenue': 'Revenue', 'profit': None}
- team_last_financials.csv columns: {'budget': 'budget', 'payroll': None, 'cash': 'cash', 'revenue': 'total_revenue', 'profit': 'financial_balance'}
- team_history_financials load error: 'year'
- team_history_financials.csv season chosen 1980
