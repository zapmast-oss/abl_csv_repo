# Next Sortable Capture Checklist — 1981 as of 1981-07-19

Promotion remains **blocked**. Create a second staging folder for team statistics, staff, and team context. Capture every report below; do not place these directly into `csv/abl_statistics/`.

## P0 - critical

- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_batting_stats.csv` — OOTP category: **team stats** — Provides the 24-team offensive totals needed to reconcile and enrich team identity.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_batting_xtra.csv` — OOTP category: **team stats** — Adds advanced team batting/baserunning denominators not guaranteed by the player-only capture.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_1.csv` — OOTP category: **team stats** — Provides team catcher-defense and running-control evidence for defensive context.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_c_fielding_2.csv` — OOTP category: **team stats** — Provides team catcher-defense and running-control evidence for defensive context.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_pitching_1.csv` — OOTP category: **team stats** — Provides core 24-team pitching totals and workload needed for team run-prevention enrichment.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_pitching_2.csv` — OOTP category: **team stats** — Provides advanced 24-team pitching rates and components for validated run-prevention profiles.
## P1 - high

- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_abl_staff.csv` — OOTP category: **staff** — Supplies the team-to-manager/staff layer required for management context and later manager signals.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_team_cur_rec_hist.csv` — OOTP category: **team context** — Provides a supplemental OOTP team-record/history report for reconciliation against raw-game standings; it cannot establish the cutoff.
## P2 - context

- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_team_finan.csv` — OOTP category: **financials** — Provides payroll, budget, attendance, and financial context for organization and fan stories.
- [ ] `abl_statistics_team_statistics___info_-_sortable_stats_team_pers_park.csv` — OOTP category: **park/personality** — Provides park and team personality context unavailable in a player-statistics-only capture.

## Required completion checks

- [ ] The second staging folder contains all ten filenames or documented structurally equivalent exports.
- [ ] Re-run structural reconciliation across both staging folders.
- [ ] Create SHA-256 capture manifests for both folders.
- [ ] Confirm all 20 governed existing source contracts have a reviewed replacement or an explicit retain-old decision.
- [ ] Resolve the six proposed new-source registrations and all overlap holds.
- [ ] Prepare a dry-run promotion plan and request explicit authorization before copying or replacing any source.
