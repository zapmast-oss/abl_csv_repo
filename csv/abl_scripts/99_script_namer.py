# List of script names you want to create
scripts_with_comments = {
    "abl_momentum_windows.py": "Last-10 win% and change vs prior 10",
    "abl_pythag_over_under.py": "Actual W% vs Pythag; diff flag",
    "abl_one_run_record.py": "1-run game W-L and W%",
    "abl_blowout_resilience.py": "5+ run decision W-L and avg margin",
    "abl_late_inning_clutch.py": "7th+ run diff, comebacks, blown leads",
    "abl_home_road_splits.py": "Home vs Road W% + park run factor note",
    "abl_sos_last14.py": "Opponents' avg W% over last 14 days",
    "abl_division_leverage.py": "Record vs division + remaining intra-division",
    "abl_defense_package.py": "Team DER, ZR sum, errors/9, DP/9",
    "abl_offense_profile.py": "% runs via HR, SB attempts/g, 2-out RBI",
    "abl_bullpen_stress.py": "Reliever IP last 7/14, back-to-backs, hi-lev proxy",
    "abl_rotation_stability.py": "Starts by top 5, QS%, avg IP/GS, SP injuries note",
    "abl_team_babip_luck.py": "Team batting & pitching BABIP vs league",
    "abl_basepath_pressure.py": "SB%, attempts/g, (XBT% if present)",
    "abl_grit_index.py": "Record trailing after 6th, extras, walk-offs",
    "abl_hitter_leadoff_firestarter.py": "Leadoff (spot=1) OBP & 1st-inning runs",
    "abl_hitter_risp_delta.py": "OPS with RISP vs overall OPS",
    "abl_hitter_platoons.py": "OPS vR vs vL deltas; platoon standouts",
    "abl_hitter_two_strike.py": "BA/K% with two strikes; 14-day trend",
    "abl_hitter_batted_ball.py": "GB/FB/LD, pull/oppo, HR/FB profile",
    "abl_hitter_heat_check.py": "7-day rolling OPS (min 20 PA) with prior 7",
    "abl_rookie_watch.py": "Rookie WAR pace, OPS+, defensive runs",
    "abl_role_profiles.py": "Table-setter vs clearer: RBI/PA & leadoff OBP",
    "abl_hard_luck_hitters.py": "Low BABIP outliers with rebound hints",
    "abl_pitcher_fip_gap.py": "ERA – FIP over/under-performers",
    "abl_pitcher_whiff_k9.py": "K/9 and (CSW% proxy if present); pitch type if available",
    "abl_pitcher_gb_savants.py": "GB% and DP/9 inducement",
    "abl_reliever_high_lev.py": "Saves+holds, blown saves, IR% (1–IRS/IR)",
    "abl_tto_splits.py": "OPS allowed by times-through-order",
    "abl_command_trend.py": "BB% trend and first-pitch strike last 3 GS",
    "abl_zone_rating_spotlight.py": "Team ZR by position; top 2 by innings",
    "abl_catcher_battery_value.py": "CS% and pitcher ERA by catcher",
    "abl_outfield_arms.py": "OF assists per 1000 INN; 'hold/no-go' note",
    "abl_travel_fatigue.py": "Consecutive road days; first-game-after-travel W%",
    "abl_scoring_innings_heatmap.py": "Scoring/allowing by inning (team heat)",
    "abl_manager_tendencies.py": "SBs, bunts, hook speed, platoon usage rate",
    "abl_power_surge.py": "Weekly HR/PA deltas by team & park",
    "abl_system_crash.py": "OPS drop ≥ .200 over last 50 PA (slumps)",
    "abl_runway_projection.py": "Next 10 games: opponent W% + park factor → env",
    "abl_milestones_horizon.py": "Upcoming career/season milestones within 5",
}

for filename, comment in scripts_with_comments.items():
    target_name = filename if filename.startswith("z_") else f"z_{filename}"
    with open(target_name, "w", encoding="utf-8") as f:
        f.write(f"# {comment}\n")
        f.write(f"# Placeholder for {target_name}\n")
        f.write("# TODO: implement report logic\n")
