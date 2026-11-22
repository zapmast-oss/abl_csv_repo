"""Read production receipts and build the local dashboard's data file."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

from abl_production_workflow import MODES, TRIGGERS, read_json, read_log, recent_sessions, utc_now
from abl_newsroom_verify import LEAGUE_ID, inventory, iso_date, registered_sources, rows, source_digest


DASHBOARD = Path(__file__).resolve().parent / "dashboard/index.html"
REGISTRY = DASHBOARD.parents[1] / "registry.json"


def link(path, label):
    if path and Path(path).exists():
        return {"label": label, "href": Path(path).absolute().as_uri()}


def evidence(path, label):
    return {"label": label, "path": str(path), **(link(path, label) or {})}


def file_time(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def latest_file(directory, pattern):
    return max(directory.glob(pattern), key=lambda path: path.stat().st_mtime_ns, default=None)


def maybe_json(path, warnings, *, expected=False):
    try:
        if path.is_file() or expected:
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object.")
            return value
    except (OSError, ValueError) as error:
        warnings.append({"title": f"Could not read {path.name}", "detail": str(error)})
    return {}


def verified_desk(paths, warnings):
    current = maybe_json(paths.newsroom / "current.json", warnings, expected=True)
    if not current:
        return {}, {}
    edition = paths.newsroom / "editions" / Path(current.get("edition", "")).name
    receipt = maybe_json(edition / "verification.json", warnings, expected=True)
    if any(not current.get(key) or current[key] != receipt.get(key)
           for key in ("source_id", "cutoff", "league_date", "season")):
        warnings.append({"title": "Newsroom pointer needs verification",
                         "detail": "The edition receipt and current pointer do not identify the same export."})
        return {}, {}
    verified = {key: current.get(key) for key in ("source_id", "cutoff", "league_date", "season", "raw_dir")}
    verified["day"] = receipt.get("day")
    verified["receipt"] = str(edition / "verification.json")
    verified["origin"] = "Verified newsroom edition"
    return current, verified


def compare_exports(paths, verified, warnings):
    try:
        registered = set(registered_sources(paths.repo))
    except (OSError, ValueError, KeyError) as error:
        warnings.append({"title": "OOTP source registry unavailable", "detail": str(error)})
        return []
    result = []
    for name, folder in (("Live OOTP CSV", paths.live_export), ("Repository CSV", paths.input_data)):
        card = {"label": name, "link": link(folder, "Open CSV folder"), "status": "Unavailable"}
        try:
            digest = source_digest(inventory(folder, registered))
            card.update(source_id=digest, at=file_time(folder / "games.csv"))
            if digest == verified.get("source_id"):
                card.update(status="Verified capture", detail=f"{len(registered)} matching CSVs")
            else:
                row = next(row for row in rows(folder / "leagues.csv") if row["league_id"] == LEAGUE_ID)
                card.update(status="Different capture",
                            detail=f"Export reports {iso_date(row['current_date'])}. Check its cutoff before use.")
        except (OSError, ValueError, KeyError, StopIteration) as error:
            card["detail"] = str(error) or "No league 200 row"
            warnings.append({"title": f"{name} could not be checked", "detail": card["detail"]})
        result.append(card)
    return result


def upcoming(verified, display_date, warnings):
    if not verified or display_date != verified["league_date"]:
        return {}
    try:
        raw = Path(verified["raw_dir"])
        schedule = {}
        for game in rows(raw / "games.csv"):
            if game["league_id"] == LEAGUE_ID and game["game_type"] == "0" and game["played"] == "0":
                when = iso_date(game["date"])
                if when >= display_date and when.startswith(str(verified["season"])):
                    schedule.setdefault(when, []).append(game)
        if not schedule:
            return {}
        day = min(schedule)
        clubs = {club["team_id"]: club["abbr"] for club in rows(raw / "teams.csv") if club["league_id"] == LEAGUE_ID}
        games = [{"away": clubs[game["away_team"]], "home": clubs[game["home_team"]], "id": game["game_id"]}
                 for game in sorted(schedule[day], key=lambda g: (int(g["time"]), int(g["game_id"])))]
        return {"date": day, "games": games}
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.append({"title": "Verified schedule is unavailable", "detail": str(error)})
        return {}


def show_session(session, paths):
    folder = paths.control / "sessions" / session["id"]
    stages = []
    for stage in session["stages"].values():
        stages.append({**{key: stage.get(key) for key in
                          ("label", "status", "message", "at", "started_at", "source_cutoff")},
                       "log": link(stage.get("log"), "Run log"),
                       "outputs": [output for i, path in enumerate(stage.get("outputs", []))
                                   if (output := link(path, Path(path).name or f"Output {i + 1}"))]})
    return {
        "id": session["id"], "mode": MODES[session["mode"]], "trigger": TRIGGERS[session["trigger"]],
        "window": session["window"], "league_date": session["league_date"], "at": session["updated_at"],
        "closed_at": session["closed_at"], "stages": stages,
        "record": link(folder / "session.json", "Session record"),
        "checkpoint": link(folder / "checkpoint.json", "Protection checkpoint"),
    }


def mirror_receipt(paths):
    note = {"status": "Not started", "detail": "No verification receipt is available."}
    log = latest_file(paths.scripts, "MirrorCheck_*.txt")
    if log:
        totals = re.findall(r"^\s*(?:Dirs|Files)\s*:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                            read_log(log)[-4500:], re.MULTILINE)
        if len(totals) == 2:
            differences = sum(int(row[i]) for row in totals for i in (1, 3, 5))
            failed = sum(int(row[4]) for row in totals)
            status = "Failed" if failed else "Requires review" if differences else "Succeeded"
            detail = f"{differences} copy/mismatch/extra entries, {failed} failed entries at last read-only verification."
        else:
            status, detail = "Failed", "The last verification log has no complete Robocopy summary."
        note.update(status=status, detail=detail, at=file_time(log), link=link(log, "Mirror verification log"))
    return note


def registry_data(paths):
    registry = read_json(REGISTRY)
    path_roots = {"scripts:": paths.scripts, "save:": paths.save, "home:": Path.home()}
    for item in registry["projects"]:
        resolved = []
        for source in item["evidence"]:
            for prefix, root in path_roots.items():
                if source["path"].startswith(prefix):
                    path = root / source["path"].removeprefix(prefix)
                    break
            else:
                path = paths.repo / source["path"]
            resolved.append(evidence(path, source["label"]))
        item["evidence"] = resolved
    return registry["projects"]


def gather_status(paths):
    warnings = []
    sessions = recent_sessions(paths)
    latest = sessions[0] if sessions else {}
    saved = next((session for session in sessions if session["stages"]["save"]["status"] == "Succeeded"), {})
    current, verified = verified_desk(paths, warnings)
    assignments = (maybe_json(Path(current["editorial"]) / "slate.json", warnings, expected=True)
                   if current.get("editorial") else {})
    captures = [dict(session["verified"], origin="Verified session export")
                for session in sessions if session.get("verified") and Path(session["verified"]["receipt"]).is_file()]
    if verified or captures:
        verified = max([capture for capture in [verified, *captures] if capture],
                       key=lambda capture: (capture["league_date"], capture["cutoff"]))
    displayed_date = verified.get("league_date")
    date_source = verified.get("origin")
    if saved and saved["league_date"] > (displayed_date or ""):
        displayed_date = saved["league_date"]
        date_source = "Commissioner-confirmed OOTP save"
    slate = upcoming(verified, displayed_date, warnings)
    poll = maybe_json(paths.news_logs / "poll-state.json", warnings)
    if poll.get("status") == "failed":
        warnings.append({"title": "Newsroom poll failed", "detail": poll.get("last_error", "Inspect the poll log.")})
    runs = [(session["updated_at"], MODES[session["mode"]]) for session in sessions]
    if poll.get("last_success_utc"):
        runs.append((poll["last_success_utc"], "Newsroom"))
    last_run = dict(zip(("at", "label"), max(runs))) if runs else {}
    projects = registry_data(paths)
    desk = maybe_json(Path(current["queue"]), warnings, expected=True) if current.get("queue") else {}
    issue = (maybe_json(Path(current["publication"]) / "articles.json", warnings, expected=True)
             if current.get("publication") else {})
    for project in projects:
        if project.get("team_id"):
            beat = assignments.get("teams", {}).get(str(project["team_id"]), {})
            code = project["id"].removeprefix("beat-").upper()
            if current.get("reports"):
                report = Path(current["reports"]) / f"{project['team_id']:02d}-{code}.pdf"
                project.update(record=beat.get("current_record"), report=link(report, "Read beat report"))
        elif project["id"].startswith("division-"):
            division = next((item for item in desk.get("divisions", []) if item["division"] == project["title"]), {})
            project.update(movement=division.get("movement"), next_series=division.get("next"))
    releases = []
    for session in sessions:
        for release in session.get("releases", []):
            releases.append({**release, "session": session["id"],
                             "file": link(release["artifact"], "Review artifact")})
    inputs_log = latest_file(paths.repo / "logs/input_updates", "ootp_input_update_*.log")
    poll_log = latest_file(paths.news_logs, "*.log")
    html_path = paths.save / "news/html/index.html"
    return {
        "generated_at": utc_now(),
        "baseball": {"league_date": displayed_date, "date_source": date_source, "cutoff": verified.get("cutoff"),
                     "cutoff_day": verified.get("day"), "verified": link(verified.get("receipt"), "Verification receipt"),
                     "window": latest.get("window") if latest and latest["league_date"] == displayed_date else None,
                     "slate": slate, "exports": compare_exports(paths, verified, warnings), "last_run": last_run},
        "desk": {"cutoff": current.get("cutoff"),
                 "links": [item for item in (
                     link(paths.newsroom / "index.html", "Newsroom reading room"),
                     link(current.get("abl_411"), "ABL 411"),
                     link(current.get("slate"), "Story slate"),
                     link(current.get("queue"), "Desk evidence and candidates"),
                     link(current.get("results_memo"), "Results memo"),
                     link(current.get("observer"), "Baseball Observer"),
                     link(Path(current["publication"]) / "index.html" if current.get("publication") else None,
                          "Reviewed local issue"),
                 ) if item],
                 "reviewed": [item["title"] for item in issue.get("articles", [])]},
        "operations": {"session": show_session(latest, paths) if latest else None,
                       "sessions": [show_session(session, paths) for session in sessions[1:]],
                       "poll": poll, "poll_log": link(poll_log, "Newsroom poll log"),
                       "input_log": link(inputs_log, "Last native CSV update log"),
                       "h_ready": Path(paths.backup.anchor).is_dir(), "i_ready": Path("I:/").is_dir(),
                       "mirror": mirror_receipt(paths), "html": link(html_path, "Open generated OOTP reports"),
                       "html_at": file_time(html_path) if html_path.is_file() else None},
        "releases": releases, "warnings": warnings, "projects": projects,
        "runbook": link(paths.repo / "docs/ABL_COMMAND_CENTER.md", "Open runbook"),
        "verification": link(paths.repo / "docs/ABL_COMMAND_CENTER_VERIFICATION.md", "Verification record"),
        "registry": link(REGISTRY, "Master registry"),
    }


def refresh(paths):
    data = gather_status(paths)
    target = DASHBOARD.with_name("status.js")
    scratch = target.with_suffix(f".{uuid4().hex}.tmp")
    try:
        scratch.write_text("window.ABL_STATUS = " + json.dumps(data, ensure_ascii=True) + ";\n", encoding="utf-8")
        scratch.replace(target)
    finally:
        scratch.unlink(missing_ok=True)
    return DASHBOARD
