"""Checkpoint a played ABL session and coordinate its existing production tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
from uuid import uuid4


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "csv/abl_scripts")]
from abl_newsroom_control import claim_operation, current_issue_matches, utc_now, write_json
from abl_newsroom_verify import (
    capture,
    default_live_dir,
    inventory,
    registered_sources,
    source_digest,
    verify_export,
)


MODES = {"quick": "Quick Play", "standard": "Standard Session", "featured": "Featured Production"}
TRIGGERS = {
    "post-game": "Post-game",
    "post-day": "Post-day",
    "post-series": "Post-series",
    "monday": "Monday",
    "milestone": "Milestone",
    "season-end": "End of season",
}
STAGES = {
    "play": "Play",
    "save": "Save",
    "backup": "League backup",
    "capture": "Capture OOTP CSV",
    "validate": "Validate OOTP CSV",
    "sync": "Synchronize ABL Input Data",
    "desk": "Update the Desk",
    "journalism": "Prepare journalism",
    "review": "Commissioner review",
    "publish": "External publication",
    "archive": "Archive checkpoint",
    "close": "Close session",
    "html": "OOTP HTML report bundle",
    "statsplus": "StatsPlus enrichment",
    "video": "Preserve SBV videos",
    "mirror": "Read-only mirror check",
}
CSV_STAGES = ("capture", "validate", "sync", "desk", "journalism")
MANUAL_LANES = ("html", "statsplus", "mirror")
RELEASE_LANES = ("editorial", "video", "html-website", "statsplus")
SESSION_ID = re.compile(r"\d{8}T\d{6}Z-[a-f0-9]{8}\Z")


@dataclass(frozen=True)
class Paths:
    repo: Path
    scripts: Path
    live_export: Path
    backup: Path
    control: Path
    snapshots: Path
    newsroom: Path

    @classmethod
    def local(cls):
        return cls(
            REPO,
            Path.home() / "Scripts",
            default_live_dir(),
            Path("H:/ABL_SAFETY_COPY/OOTP_Backups/ABL/Current Backup/Action Baseball League.lg"),
            REPO / "logs/production_control",
            REPO / "output/newsroom/snapshots",
            REPO / "csv/out/story/newsroom",
        )

    @property
    def save(self):
        return self.live_export.parents[1]

    @property
    def news_logs(self):
        if self.newsroom == self.repo / "csv/out/story/newsroom":
            return self.repo / "logs/newsroom_control"
        return self.control / "newsroom_control"

    @property
    def input_data(self):
        return self.repo / "csv/ootp_csv"


class Deferred(Exception):
    """The commissioner can revisit this optional work after closing."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_inventory(folder):
    files = {path.name: path for path in folder.glob("*.dat") if path.is_file()}
    required = {"flag_save_completed.dat", "world.dat", "teams.dat", "players.dat"}
    if not required <= files.keys() or any(not files[name].stat().st_size for name in required):
        raise ValueError(f"The completed-save marker or core OOTP data files are missing from {folder}")
    return {name: sha256(path) for name, path in sorted(files.items())}


def read_log(path):
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig", errors="replace")


def recent_sessions(paths):
    directory = paths.control / "sessions"
    return [read_json(folder / "session.json") for folder in sorted(directory.glob("*"), reverse=True)
            if SESSION_ID.fullmatch(folder.name) and (folder / "session.json").is_file()]


def ask(prompt):
    return input(f"{prompt} [y/N] ").strip().casefold() in {"y", "yes"}


def powershell(script, *arguments, transcript=None):
    if not script.is_file():
        raise FileNotFoundError(f"Production Control operation missing: {script}")
    # The inspected updater documents a process-local Bypass invocation for its unsigned file.
    command = ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
               "-File", str(script), *map(str, arguments)]
    if transcript:
        with transcript.open("wb") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        print(read_log(transcript))
        return result.returncode
    return subprocess.run(command, check=False).returncode


def newsroom_python():
    if find_spec("reportlab") is not None:
        return sys.executable
    runtime = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
    if runtime.is_file():
        return str(runtime)
    raise FileNotFoundError("Use the existing Newsroom Python runtime with ReportLab to prepare the Desk.")


class Session:
    def __init__(self, paths, state):
        self.paths = paths
        self.state = state
        self.folder = paths.control / "sessions" / state["id"]
        self.file = self.folder / "session.json"

    @classmethod
    def create(cls, paths, mode, trigger, league_date, window):
        league_date = date.fromisoformat(league_date).isoformat()
        timestamp = utc_now()
        key = timestamp[:19].replace("-", "").replace(":", "") + "Z-" + uuid4().hex[:8]
        state = {
            "id": key, "mode": mode, "trigger": trigger, "league_date": league_date,
            "window": window, "created_at": timestamp, "updated_at": timestamp, "closed_at": None,
            "stages": {name: {"label": label, "status": "Not started", "attempts": []}
                       for name, label in STAGES.items()},
            "releases": [],
        }
        session = cls(paths, state)
        session.folder.mkdir(parents=True)
        session.persist()
        return session

    @classmethod
    def load(cls, paths, key):
        if not SESSION_ID.fullmatch(key):
            raise ValueError("Choose the session ID shown by Production Control or the dashboard.")
        return cls(paths, read_json(paths.control / "sessions" / key / "session.json"))

    def persist(self):
        self.state["updated_at"] = utc_now()
        write_json(self.file, self.state)

    def succeeded(self, name):
        return self.state["stages"][name]["status"] == "Succeeded"

    def update(self, name, status, message, *, log=None, outputs=(), started_at=None):
        stage = self.state["stages"][name]
        cutoff = self.state.get("verified", {}).get("cutoff")
        entry = {
            "status": status, "message": message, "at": utc_now(), "started_at": started_at,
            "source_cutoff": cutoff, "log": str(log) if log else None,
            "outputs": [str(path) for path in outputs],
        }
        stage["attempts"].append(entry)
        stage.update(entry)
        self.persist()

    def defer(self, names, reason):
        for name in names:
            stage = self.state["stages"][name]
            if stage["status"] in {"Not started", "Deferred"} and stage.get("message") != reason:
                self.update(name, "Deferred", reason)

    def execute(self, name, action):
        if self.succeeded(name):
            return True
        stage = self.state["stages"][name]
        if stage["status"] == "Running":
            self.update(name, "Failed", "Interrupted before a success receipt was recorded.",
                        log=stage.get("log"), started_at=stage.get("started_at"))
        attempt = self.folder / f"{name}-{uuid4().hex[:8]}"
        attempt.mkdir()
        log = attempt / "result.log"
        started = utc_now()
        log.write_text(f"{started} {STAGES[name]} started\n", encoding="utf-8")
        self.update(name, "Running", "Working", log=log, started_at=started)
        try:
            message, outputs = action(attempt)
            status = "Succeeded"
        except Deferred as error:
            message, outputs, status = str(error), list(attempt.glob("*.log")), "Deferred"
        except (Exception, KeyboardInterrupt) as error:
            message = str(error) or "Interrupted. Resume the session to retry."
            outputs, status = list(attempt.iterdir()), "Failed"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {status}: {message}\n")
        self.update(name, status, message, log=log, outputs=outputs, started_at=started)
        print(f"{STAGES[name]} | {status} | {message}")
        return status == "Succeeded"

    def check_same_save(self):
        if save_inventory(self.paths.save) != self.state["save_hashes"]:
            raise ValueError("The OOTP save has changed since this checkpoint. Start a new session for the current save.")

    def record_save(self, _):
        hashes = save_inventory(self.paths.save)
        if hashes != save_inventory(self.paths.save):
            raise ValueError("OOTP is still saving. Wait for it to finish, then resume.")
        self.state["save_hashes"] = hashes
        self.state["save_marker_ns"] = (self.paths.save / "flag_save_completed.dat").stat().st_mtime_ns
        return f"Commissioner confirmed the save. Read and hashed {len(hashes)} OOTP data files.", [self.paths.save]

    def backup(self, attempt):
        if not self.paths.backup.anchor or not Path(self.paths.backup.anchor).is_dir():
            raise FileNotFoundError(f"League backup drive is unavailable: {self.paths.backup.anchor}")
        self.check_same_save()
        log = attempt / "robocopy.log"
        result = powershell(
            self.paths.scripts / "Backup ABL League.ps1", "-Source", self.paths.save,
            "-Destination", self.paths.backup, "-LogFile", log, "-NoPause",
        )
        if result >= 8 or not log.is_file():
            raise ValueError(f"League backup failed with exit code {result}. Review {attempt}.")
        copied = save_inventory(self.paths.backup)
        expected = self.state["save_hashes"]
        if any(copied.get(name) != digest for name, digest in expected.items()):
            raise ValueError("The rolling backup failed the OOTP data-file SHA-256 comparison.")
        self.check_same_save()
        self.state["league_backup"] = {"destination": str(self.paths.backup), "log": str(log),
                                       "robocopy_exit": result, "verified_dat_files": len(expected)}
        return f"Rolling safety copy completed. SHA-256 matched {len(expected)} saved data files.", [log, self.paths.backup]

    def capture_csv(self, _):
        self.check_same_save()
        registry = registered_sources(self.paths.repo)
        source = self.paths.live_export
        hashes = inventory(source, set(registry))
        digest = source_digest(hashes)
        raw = capture(source, self.paths.snapshots, hashes, digest, registry)
        self.check_same_save()
        self.state["capture"] = {
            "source_id": digest, "raw_dir": str(raw), "captured_at": utc_now(),
            "games_csv_mtime_ns": (source / "games.csv").stat().st_mtime_ns,
        }
        return f"Staged {len(hashes)} registered exports by SHA-256.", [raw / "manifest.json"]

    def validate_csv(self, attempt):
        captured = self.state["capture"]
        result = verify_export(self.paths.repo, attempt / "preflight", Path(captured["raw_dir"]),
                               snapshot_root=self.paths.snapshots)
        if result["source_id"] != captured["source_id"]:
            raise ValueError("The verified CSV fingerprint differs from the recorded capture.")
        if result["league_date"] != self.state["league_date"] or result["cutoff"] != self.state["expected_cutoff"]:
            raise ValueError(f"Export reports {result['league_date']} with finals through {result['cutoff']}. "
                             "Check the recorded date and re-export the saved game.")
        trigger = self.state["trigger"]
        if trigger == "post-day" and result["day"]["remaining"]:
            raise ValueError("The cutoff day's export still has unplayed games. Use a post-game checkpoint.")
        game_id = self.state.get("expected_game")
        if game_id and not any(game["game_id"] == game_id and game["date"] == result["cutoff"]
                               for game in result["games"]):
            raise ValueError(f"Game {game_id} is absent from the completed games on the stated cutoff.")
        if (trigger == "post-game" and not game_id and result["cutoff"] == result["league_date"]
                and captured["games_csv_mtime_ns"] < self.state["save_marker_ns"]):
            raise ValueError("The same-day CSV predates the save. Re-export or supply the completed game ID.")
        receipt = next((attempt / "preflight").glob("*.json"))
        self.state["verified"] = {
            **{key: result[key] for key in ("source_id", "league_date", "cutoff", "season", "day", "raw_dir")},
            "games": len(result["games"]), "receipt": str(receipt), "verified_at": utc_now(),
        }
        return f"Current-state preflight passed with {len(result['games'])} finals through {result['cutoff']}.", [receipt]

    def confirm_current_capture(self):
        verified = self.state["verified"]
        self.check_same_save()
        needed = set(registered_sources(self.paths.repo))
        if source_digest(inventory(self.paths.live_export, needed)) != verified["source_id"]:
            raise ValueError("The live OOTP export has advanced or changed. Use a new session for current production.")
        raw = Path(verified["raw_dir"])
        if source_digest(inventory(raw, needed)) != verified["source_id"]:
            raise ValueError("The archived CSV capture no longer matches its verified fingerprint.")
        return raw, needed

    def sync_input(self, attempt):
        raw, needed = self.confirm_current_capture()
        script = self.paths.scripts / "Update the Repo.ps1"
        args = ("-Source", raw, "-Destination", self.paths.input_data, "-LogDirectory", attempt)
        result = powershell(script, "-Mode", "Preview", *args)
        previews = list(attempt.glob("ootp_input_preview_*.log"))
        if result or len(previews) != 1 or "preview complete" not in read_log(previews[0]):
            raise ValueError(f"ABL Input Data Preview did not produce a success receipt (exit {result}).")
        outputs = [*previews, self.paths.input_data]
        if source_digest(inventory(self.paths.input_data, needed)) == self.state["verified"]["source_id"]:
            return "Preview and SHA-256 confirmed all repository CSVs already match.", outputs
        if not ask("Proceed to the existing Update prompt"):
            raise Deferred("Preview is available. Update awaits the commissioner.")
        self.confirm_current_capture()
        result = powershell(script, "-Mode", "Update", *args)
        updates = list(attempt.glob("ootp_input_update_*.log"))
        if result:
            raise ValueError(f"ABL Input Data Update exited {result}. Review the native Update log.")
        if not updates:
            raise Deferred("The existing Update prompt was cancelled.")
        if source_digest(inventory(self.paths.input_data, needed)) != self.state["verified"]["source_id"]:
            raise ValueError("The repository CSVs do not match the validated capture after Update.")
        return "Existing Update ran and all repository CSVs matched the verified SHA-256 set.", [*outputs, *updates]

    def prepare_desk(self, attempt):
        self.paths.news_logs.mkdir(parents=True, exist_ok=True)
        verified = self.state["verified"]
        log = attempt / "newsroom.log"
        with claim_operation(self.paths.news_logs) as acquired:
            if not acquired:
                raise Deferred("The existing Newsroom poll is active. Resume after it finishes.")
            raw, _ = self.confirm_current_capture()
            command = [newsroom_python(), "-B", str(self.paths.repo / "csv/abl_scripts/abl_newsroom.py"),
                       "--source", str(raw), "--archive", str(self.paths.newsroom)]
            if current_issue_matches(verified["source_id"], self.paths.newsroom):
                command.append("--status")
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=self.paths.repo, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False).returncode
            if result or not current_issue_matches(verified["source_id"], self.paths.newsroom):
                raise ValueError(f"Newsroom did not verify a complete matching edition. Review {log}.")
            current = read_json(self.paths.newsroom / "current.json")
            self.state["desk"] = {key: current.get(key) for key in ("edition", "editorial", "queue", "slate", "abl_411")}
        return f"Desk ready for edition {current['edition']}.", [log, self.paths.newsroom / "index.html"]

    def prepare_journalism(self, _):
        desk = self.state["desk"]
        outputs = [Path(desk[key]) for key in ("slate", "queue", "abl_411")]
        if not all(path.is_file() for path in outputs):
            raise ValueError("The prepared slate, 411, or franchise queue is missing.")
        return "Slate, ABL 411 and the franchise queue are ready for story selection and review.", outputs

    def backup_video(self, attempt):
        if not Path("H:/").is_dir():
            raise FileNotFoundError("The SBV video backup requires H:.")
        script = self.paths.scripts / "Backup SBV Videos.ps1"
        transcript = attempt / "video-preview.log"
        preview = powershell(script, "-Mode", "Preview", transcript=transcript)
        if not 0 <= preview < 8 or "PREVIEW ONLY. No files were copied." not in read_log(transcript):
            raise ValueError(f"SBV Video Backup Preview failed (exit {preview}).")
        if not ask("Run SBV Video Backup Update for completed recordings"):
            raise Deferred("Video Preview ran. Update awaits the commissioner.")
        result = powershell(script, "-Mode", "Update", transcript=attempt / "video-update.log")
        destination = Path("H:/SBV Video Backup")
        if result >= 8 or result < 0:
            raise ValueError(f"SBV Video Backup Update failed (exit {result}).")
        logs = sorted((destination / "logs").glob("sbv_video_backup_*.log"),
                      key=lambda path: path.stat().st_mtime_ns, reverse=True)
        if not logs or logs[0].stat().st_mtime_ns < (attempt / "result.log").stat().st_mtime_ns:
            raise ValueError("SBV Video Backup Update did not leave a new log.")
        if not all(label in read_log(logs[0]) for label in ("OBS ORIGINAL MKV BACKUP", "PRODUCTION MASTER MP4 BACKUP")):
            raise ValueError(f"The video backup log is missing a copy lane: {logs[0]}")
        return "The native video backup logged both the MKV originals and MP4 masters.", [logs[0], destination]

    def checkpoint(self, _):
        file = self.folder / "checkpoint.json"
        if file.exists():
            previous = read_json(file)
            if previous["session"] != self.state["id"] or previous["save_hashes"] != self.state["save_hashes"]:
                raise ValueError(f"Checkpoint evidence conflicts with this session: {file}")
        else:
            write_json(file, {
                "session": self.state["id"], "recorded_at": utc_now(), "league_date": self.state["league_date"],
                "window": self.state["window"], "backup": self.state["league_backup"],
                "save_hashes": self.state["save_hashes"], "verified_export": self.state.get("verified"),
            })
        return "Protection receipt archived with the saved data hashes and backup log.", [file]

    def close(self, _):
        self.state["closed_at"] = self.state["closed_at"] or utc_now()
        return "The baseball checkpoint is closed. Deferred work remains in the queue.", [self.file]

    def csv_cycle(self):
        if not self.succeeded("validate"):
            if not ask("Capture CSV exported in OOTP from this saved session"):
                self.defer(CSV_STAGES, "Export and validate this saved session before producing coverage.")
                return
            expected = input("Last completed game date in this export (YYYY-MM-DD): ").strip()
            game = input("Completed game ID on that date, if known [Enter to skip]: ").strip()
            try:
                self.state["expected_cutoff"] = date.fromisoformat(expected).isoformat()
                if self.state["expected_cutoff"] > self.state["league_date"] or (game and not game.isdecimal()):
                    raise ValueError("Check the cutoff and optional numeric game ID.")
            except ValueError as error:
                print(error)
                self.defer(CSV_STAGES, "Record a valid cutoff and optional game ID to continue.")
                return
            self.state["expected_game"] = game or None
            if self.succeeded("capture"):
                self.update("capture", "Deferred", "Re-capture requested after the previous validation attempt.")
            if not self.execute("capture", self.capture_csv) or not self.execute("validate", self.validate_csv):
                self.defer(CSV_STAGES, "Hold until the OOTP capture passes validation.")
                return
        if not self.succeeded("sync"):
            if ask("Preview ABL Input Data synchronization"):
                if not self.execute("sync", self.sync_input) and self.state["stages"]["sync"]["status"] == "Failed":
                    self.defer(("desk", "journalism"), "Review the failed input synchronization first.")
                    return
            elif self.state["stages"]["sync"]["status"] == "Failed":
                return
            else:
                self.defer(("sync",), "Review the independent ABL Input Data Preview when ready.")
        if not self.succeeded("journalism"):
            if ask("Prepare the local Desk and journalism slate"):
                if self.execute("desk", self.prepare_desk):
                    self.execute("journalism", self.prepare_journalism)
                else:
                    self.defer(("journalism",), "Waiting for a verified Desk edition.")
            else:
                self.defer(("desk", "journalism"), "Run the newsroom preparation for the validated capture.")

    def run(self, *, resumed=False):
        if not self.succeeded("save"):
            if not ask(f"Baseball played and OOTP save completed for {self.state['league_date']}"):
                print(f"Resume {self.state['id']} once the save finishes.")
                return False
            self.execute("play", lambda _: ("Playing window confirmed by the commissioner.", []))
            if not self.execute("save", self.record_save):
                return False
        if not self.execute("backup", self.backup):
            return False
        if resumed or self.state["mode"] != "quick":
            self.csv_cycle()
        else:
            self.defer(CSV_STAGES, "Quick Play checkpoint. Resume to capture and produce coverage if needed.")
        if self.state["mode"] == "featured" and not self.succeeded("video"):
            if ask("Preview backup of closed OBS recordings and rendered masters"):
                self.execute("video", self.backup_video)
            else:
                self.defer(("video",), "Preserve OBS originals and production masters after recording and rendering.")
        self.defer(("review", "publish"), "Queue a selected artifact when it is ready for commissioner review.")
        return self.execute("archive", self.checkpoint) and self.execute("close", self.close)

    def record_lane(self, stage, evidence, note):
        self.require_protected()
        if evidence is None:
            self.update(stage, "Deferred", note)
        else:
            if not evidence.is_file():
                raise FileNotFoundError(f"Evidence file missing: {evidence}")
            self.update(stage, "Succeeded", f"Commissioner verified {note} | SHA-256 {sha256(evidence)}",
                        outputs=[evidence])

    def require_protected(self):
        if not self.succeeded("backup"):
            raise ValueError("Complete the verified league backup for this session first.")

    def queue_release(self, lane, channel, title, file):
        self.require_protected()
        if lane in {"editorial", "statsplus"} and not self.succeeded("validate"):
            raise ValueError("Validate this session's OOTP CSV before queuing this release.")
        if lane in {"editorial", "statsplus"} and self.state["stages"]["sync"]["status"] == "Failed":
            raise ValueError("Resolve the failed verification in ABL Input Data before queuing this release.")
        if not file.is_file():
            raise FileNotFoundError(f"Release artifact missing: {file}")
        digest = sha256(file)
        for previous in recent_sessions(self.paths):
            if any(r["channel"].casefold() == channel.casefold() and r["sha256"] == digest
                   for r in previous.get("releases", [])):
                raise ValueError("This exact artifact already has a record for the selected channel.")
        release = {
            "id": uuid4().hex[:12], "lane": lane, "title": title, "channel": channel, "artifact": str(file),
            "sha256": digest, "queued_at": utc_now(), "approval": None, "receipt": None,
            "source_cutoff": self.state.get("verified", {}).get("cutoff"),
        }
        self.state["releases"].append(release)
        self.release_stages()
        return release["id"]

    def release(self, key):
        record = next((release for release in self.state["releases"] if release["id"] == key), None)
        if record is None:
            raise ValueError("Release ID not found in this session.")
        if sha256(Path(record["artifact"])) != record["sha256"]:
            raise ValueError("The release artifact has changed. Queue the revised artifact for a new approval.")
        return record

    def approve_release(self, key, reviewer):
        record = self.release(key)
        if record["approval"]:
            raise ValueError("This release already has an approval receipt.")
        record["approval"] = {"reviewer": reviewer, "at": utc_now(), "sha256": record["sha256"]}
        self.release_stages()

    def approved_artifact(self, key):
        record = self.release(key)
        if not record["approval"] or record["receipt"]:
            raise ValueError("The release must have approval and no existing publication receipt.")
        return record

    def record_publication(self, key, url):
        record = self.approved_artifact(key)
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Record the actual public HTTP(S) URL.")
        for previous in recent_sessions(self.paths):
            if any((r.get("receipt") or {}).get("url") == url for r in previous.get("releases", [])):
                raise ValueError("That public URL is already recorded.")
        record["receipt"] = {"url": url, "at": utc_now()}
        self.release_stages()

    def release_stages(self):
        records = self.state["releases"]
        awaiting = sum(not record["approval"] for record in records)
        sent = sum(bool(record["receipt"]) for record in records)
        self.update("review", "Requires approval" if awaiting else "Succeeded",
                    f"{len(records) - awaiting} of {len(records)} exact artifacts approved.", outputs=[self.file])
        status = "Succeeded" if sent == len(records) else "Requires approval" if awaiting else "Deferred"
        self.update("publish", status, f"{sent} of {len(records)} manual publication receipts recorded.", outputs=[self.file])
