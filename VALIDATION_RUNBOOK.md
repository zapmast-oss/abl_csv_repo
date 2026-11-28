# EB Pack Regeneration and Validation (ABL 1972–1980)

Use these steps to regenerate EB pack fragments, rebuild packs, and run the validator across all seasons.

## Regenerate fragments and packs

From the repo root:
```
cd C:\sbv_repo\abl_csv_repo
foreach ($y in 1972,1973,1974,1975,1976,1977,1978,1979,1980) {
    python csv/abl_scripts/z_abl_month_glory_misery_any.py --season $y --league-id 200
    python csv/abl_scripts/z_abl_eb_schedule_context_any.py --season $y --league-id 200
    python csv/abl_scripts/_run_eb_regular_season_any.py --season $y --league-id 200
}
```

This regenerates the Month-of-Glory/Misery fragment, schedule-context fragment, and the full EB pack for each season.

## Run the validator

After regeneration, validate every season:
```
foreach ($y in 1972,1973,1974,1975,1976,1977,1978,1979,1980) {
    python csv/abl_scripts/z_abl_validate_eb_pack_any.py --season $y --league-id 200
}
```

Expected summary per season:
```
[SUMMARY] Season YYYY, league 200:
  - Preseason hype: PASS
  - Schedule context: PASS
  - Month-of-Glory/Misery: PASS
```

## If a season shows mismatches
- Schedule context mismatch: rerun the three regeneration commands for that season to refresh the schedule fragment and pack.
- Month-of-Glory not found or skipped: rerun the month fragment generator for that season, then rebuild the pack.
- Preseason hype should already PASS once the pack and fragments are rebuilt.

## Notes
- The validator reads dates and spans directly from the schedule fragment in the pack. Keep the fragment up to date when the grid parsing rules change.
- The Month-of-Glory bullets must follow the current format:
  `- Team (ABBR) - in Month went W-L (pct), delta vs season=+X.XXX`
- All commands assume league_id=200 and Python 3.12 in your environment.
