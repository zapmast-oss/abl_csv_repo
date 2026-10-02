This is the final schema for any eb_game_pack_YYYY-MM-DD_AAA_at_BBB.csv broadcast pack.

Each row = one game.
Prefixes: away_ and home_ for team-specific fields.

1. Game meta (14)

game_date – YYYY-MM-DD (e.g. 1981-05-11)

season_year – integer (e.g. 1981)

week_label – e.g. Week06

game_id – OOTP game_id (string)

series_game_number – 1, 2, 3…

series_game_count – total games in series (e.g. 3)

series_label – short text, e.g. CHI at MIA

ballpark_name

ballpark_city

ballpark_roof_type – open / dome / retractable

weather_temp_f

weather_wind_mph

weather_wind_dir – text like LF to RF, In from CF

weather_conditions – Clear, Overcast, Light Rain, etc.

2. Team context (each side)
Away team (CHI)

away_team_id

away_team_abbr – CHI

away_team_name – Chicago Fire

away_team_nickname – Fire / Chi-Fi

away_league – NBC / ABC

away_division – East / Central / West

away_wins

away_losses

away_win_pct

away_games_behind_div

away_home_record – text 11-3

away_road_record – text 6-12

away_last10_record – text 4-6

away_one_run_record – text 4-4

away_extra_innings_record – text 0-3

Home team (MIA)

home_team_id

home_team_abbr – MIA

home_team_name – Miami Hurricanes

home_team_nickname – Hurricanes / Canes

home_league

home_division

home_wins

home_losses

home_win_pct

home_games_behind_div

home_home_record – text 5-4

home_road_record – text 13-10

home_last10_record – text 7-3

home_one_run_record – text 3-1

home_extra_innings_record – text 1-0

3. Run / offense profile (per team)
Away offense profile

away_runs_per_game

away_obp

away_slg

away_ops

away_iso

away_sb_attempts_per_game

away_ubr_per_game – baserunning runs per game

away_two_out_rbi_share – % of RBI with 2 outs

away_risp_woba

away_run_profile_tag – ENUM: SLUG_PRESSURE / SLUG_ONLY / BALANCED / SCRATCH

away_clutch_index – numeric composite (1-run + late/close)

away_clutch_tag – ENUM: POSITIVE / NEUTRAL / NEGATIVE

Home offense profile

home_runs_per_game

home_obp

home_slg

home_ops

home_iso

home_sb_attempts_per_game

home_ubr_per_game

home_two_out_rbi_share

home_risp_woba

home_run_profile_tag

home_clutch_index

home_clutch_tag

4. Pitching / run prevention profile (per team)
Away pitching profile

away_team_era

away_team_fip

away_rot_era

away_rot_fip

away_rot_ip_per_start

away_pen_era

away_pen_fip

away_pen_inherited_runners

away_pen_inherited_scored_pct

away_pen_avg_leverage_index

away_pitching_profile_tag – e.g. STURDY_ROTATION_LEAKY_PEN, SHORT_ROTATION_HIGH_WIRE_PEN

Home pitching profile

home_team_era

home_team_fip

home_rot_era

home_rot_fip

home_rot_ip_per_start

home_pen_era

home_pen_fip

home_pen_inherited_runners

home_pen_inherited_scored_pct

home_pen_avg_leverage_index

home_pitching_profile_tag

5. Featured primary hitters (one per side)
Away featured hitter

away_feat_hit_player_id

away_feat_hit_player_name

away_feat_hit_pos

away_feat_hit_bats – L / R / S

away_feat_hit_pa

away_feat_hit_avg

away_feat_hit_obp

away_feat_hit_slg

away_feat_hit_ops

away_feat_hit_wrc_plus

away_feat_hit_hr

away_feat_hit_rbi

away_feat_hit_war

away_feat_hit_plus_number – your “plus number” from OPS (e.g. 89)

Home featured hitter

home_feat_hit_player_id

home_feat_hit_player_name

home_feat_hit_pos

home_feat_hit_bats

home_feat_hit_pa

home_feat_hit_avg

home_feat_hit_obp

home_feat_hit_slg

home_feat_hit_ops

home_feat_hit_wrc_plus

home_feat_hit_hr

home_feat_hit_rbi

home_feat_hit_war

home_feat_hit_plus_number

6. Probable starting pitchers (one per side)
Away starter

away_sp_player_id

away_sp_player_name

away_sp_throws – L / R

away_sp_gs – games started to date

away_sp_ip

away_sp_era

away_sp_fip

away_sp_war

away_sp_k_per_9

away_sp_bb_per_9

away_sp_hr_per_9

away_sp_qs_pct – quality start %

Home starter

home_sp_player_id

home_sp_player_name

home_sp_throws

home_sp_gs

home_sp_ip

home_sp_era

home_sp_fip

home_sp_war

home_sp_k_per_9

home_sp_bb_per_9

home_sp_hr_per_9

home_sp_qs_pct

7. Bullpen wild cards (one per side)

These are the “this guy could tilt the night” relievers you asked to add to the template.

Away bullpen wild card

away_bp_wc_player_id

away_bp_wc_player_name

away_bp_wc_role – Setup, Closer, Fireman, etc.

away_bp_wc_ip

away_bp_wc_era

away_bp_wc_avg_leverage_index

away_bp_wc_inherited_runners

away_bp_wc_inherited_scored_pct

Home bullpen wild card

home_bp_wc_player_id

home_bp_wc_player_name

home_bp_wc_role

home_bp_wc_ip

home_bp_wc_era

home_bp_wc_avg_leverage_index

home_bp_wc_inherited_runners

home_bp_wc_inherited_scored_pct