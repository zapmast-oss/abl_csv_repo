"""Entry point shared by the Production Control menu and Windows Task Scheduler."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "csv" / "abl_scripts"))
from abl_newsroom_verify import default_live_dir, inventory, registered_sources, source_digest


CONTROL = ROOT / "logs" / "newsroom_control"
CLI = ROOT / "csv" / "abl_scripts" / "abl_newsroom.py"
ARCHIVE = ROOT / "csv" / "out" / "story" / "newsroom"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def announce(message):
    # pythonw.exe has no console when Task Scheduler launches the poll.
    if sys.stdout is not None:
        print(message)


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8",
                                     suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(handle.name, path)


def live_digest(source, schema):
    return source_digest(inventory(source, set(schema)))


def current_issue_matches(source_id, archive=ARCHIVE):
    pointer = archive / "current.json"
    if not pointer.is_file():
        return False
    current = json.loads(pointer.read_text(encoding="utf-8"))
    if current.get("source_id") != source_id:
        return False
    editions = archive / "editions"
    edition = (editions / current["edition"]).resolve()
    if edition.parent != editions.resolve():
        raise ValueError("Current edition escapes the newsroom archive")
    if Path(current["reports"]) != edition/"teams" or Path(current["story_menu"]) != edition/"story_menu.csv":
        return False
    def complete(folder, required):
        if not (folder/"files.json").is_file():
            return False
        receipt=json.loads((folder/"files.json").read_text(encoding="utf-8"))
        return required <= receipt.keys() and all((folder/n).is_file() for n in receipt)
    desk=archive/"desks"/current["edition"]
    if (current.get("editorial")!=str(desk) or not (archive/"index.html").is_file()
            or not complete(desk,{"desk.json","queue.json","abl-411.pdf","story-slate.pdf","slate.json"})
            or not complete(edition,{"coverage.json","assignments.json","story_menu.csv"})
            or not (Path(current["raw_dir"])/"manifest.json").is_file()):
        return False
    if json.loads((desk/"desk.json").read_text(encoding="utf-8"))["source_id"]!=source_id:
        return False
    root=archive/"issues"/current["edition"]
    active=root/"active.json"
    if not active.is_file():
        return current.get("publication") is None and not root.exists()
    selection=json.loads(active.read_text(encoding="utf-8"))
    issue=(root/selection["revision"]).resolve()
    if (issue.parent!=root.resolve() or current.get("publication")!=str(issue) or selection["source_id"]!=source_id
            or not complete(issue,{"articles.json","reviews.json","index.html"})):
        return False
    manifest=json.loads((issue/"articles.json").read_text(encoding="utf-8"))
    published=json.loads((issue/"files.json").read_text(encoding="utf-8"))
    return (manifest["source_id"]==source_id and bool(manifest["articles"])
            and all({a["path"][:-3]+ext for ext in (".html",".md",".pdf")}<=published.keys()
                    for a in manifest["articles"]))


@contextmanager
def claim_operation(directory):
    # Windows releases this byte lock on process termination, including crashes.
    import msvcrt
    with (directory / "operation.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno not in (13, 36):
                raise
            yield False
            return
        try:
            yield True
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def run_cycle(source, log):
    python = Path(sys.executable).with_name("python.exe")
    cmd = [str(python), "-B", "-u", str(CLI), "--source", str(source)]
    return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW, check=False).returncode


def process(mode, source, schema, directory, runner=run_cycle, issue_check=current_issue_matches):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "poll-state.json"
    logfile = directory / (utc_now()[:10] + ".log")
    with claim_operation(directory) as acquired:
        if not acquired:
            announce("Newsroom operation already in progress.")
            return 0
        state = {}
        with logfile.open("a", encoding="utf-8", buffering=1) as log:
            try:
                if receipt.exists():
                    state = json.loads(receipt.read_text(encoding="utf-8"))
                required = registered_sources(ROOT) if schema is None else schema
                digest = live_digest(source, required)
                state.update(last_check_utc=utc_now(), mode=mode, live_export=str(source), observed_sha256=digest)
                may_skip = mode == "poll" and state.get("last_success_sha256") == digest
                try:
                    issue_is_current = may_skip and issue_check(digest)
                except (OSError, ValueError, KeyError, TypeError) as issue_error:
                    print(f"{utc_now()} issue index needs full verification: {issue_error}", file=log)
                    issue_is_current = False
                if issue_is_current:
                    state.update(status="unchanged", last_error=None)
                    write_json(receipt, state)
                    announce("No new export. The verified newsroom issue is current.")
                    return 0
                print(f"{utc_now()} {mode}: verify export {digest}", file=log, flush=True)
                result = runner(source, log)
                if result:
                    raise RuntimeError(f"Newsroom verifier/reporting command exited {result}. See {logfile}")
                # If another export began during the run, let the next poll recheck it.
                if digest != live_digest(source, required):
                    raise RuntimeError("Live export changed during reporting. Next poll will retry.")
                if not issue_check(digest):
                    raise RuntimeError("Reporting returned before a matching, complete edition was available.")
                state.update(status="success", last_success_utc=utc_now(), last_success_sha256=digest, last_error=None)
                write_json(receipt, state)
                print(f"{utc_now()} completed {digest}", file=log)
                announce(f"Newsroom cycle completed. Run log: {logfile}")
                announce(f"Current newsroom issue: {ROOT / 'csv/out/story/newsroom/current.json'}")
                return 0
            except Exception as error:
                traceback.print_exc(file=log)
                message = str(error)
                state.update(status="failed", last_check_utc=utc_now(), mode=mode, last_error=message)
                write_json(receipt, state)
                announce(f"Newsroom needs another attempt: {message}. Log: {logfile}")
                return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "poll"))
    args = parser.parse_args()
    sys.exit(process(args.mode, default_live_dir(), None, CONTROL))
