"""Open the ABL Command Center or run a production session from PowerShell."""

import argparse
import os
from pathlib import Path
import sys

from abl_production_status import refresh
from abl_production_workflow import (
    MANUAL_LANES,
    MODES,
    RELEASE_LANES,
    TRIGGERS,
    Paths,
    Session,
    ask,
    claim_operation,
    recent_sessions,
)


def choice(options, prompt):
    for index, label in enumerate(options.values(), 1):
        print(f"  {index}. {label}")
    value = input(f"{prompt}: ").strip()
    if value.isdecimal() and 1 <= int(value) <= len(options):
        return list(options)[int(value) - 1]
    raise ValueError("Select a number from the list.")


def required(value, label):
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Enter {label}.")
    return cleaned


def list_sessions(paths):
    sessions = recent_sessions(paths)
    if not sessions:
        print("No Command Center sessions recorded yet.")
        return
    for session in sessions:
        status = "Closed" if session["closed_at"] else "Open"
        mode = MODES[session["mode"]]
        print(f"{session['id']} | {session['league_date']} | {mode} | {status} | {session['window']}")


def start(paths, args=None):
    mode = getattr(args, "mode", None) or choice(MODES, "Session mode")
    trigger = getattr(args, "trigger", None) or choice(TRIGGERS, "Checkpoint trigger")
    league_date = getattr(args, "date", None) or input("Date shown in the saved OOTP league (YYYY-MM-DD): ").strip()
    window = getattr(args, "window", None) or input("Playing window (game, day, series or milestone): ").strip()
    session = Session.create(paths, mode, trigger, league_date, required(window, "the playing window"))
    print(f"Session {session.state['id']}")
    return session.run()


def pick_session(paths, key=None):
    if key is None:
        list_sessions(paths)
        key = input("Session ID: ").strip()
    return Session.load(paths, key)


def publication_console(paths):
    session = pick_session(paths)
    for release in session.state["releases"]:
        state = "Published" if release["receipt"] else "Approved" if release["approval"] else "Requires approval"
        print(f"{release['id']} | {state} | {release['channel']} | {release['title']}")
    action = choice({"queue": "Queue an artifact", "approve": "Approve an exact artifact",
                     "ready": "Check an approved artifact before upload",
                     "receipt": "Record a manual publication"}, "Publication tracking")
    if action == "queue":
        lane = choice(dict(zip(RELEASE_LANES, ("Editorial", "Video", "OOTP HTML website", "StatsPlus"))), "Release lane")
        key = session.queue_release(
            lane, required(input("Publication channel: "), "the channel"),
            required(input("Title: "), "the title"),
            Path(input("Local artifact or upload manifest path: ").strip().strip('"')).resolve(),
        )
        print(f"Release {key} requires commissioner approval.")
    else:
        key = input("Release ID: ").strip()
        if action == "approve":
            artifact = session.release(key)
            print(f"{artifact['title']} | {artifact['channel']}\n{artifact['artifact']}\nSHA-256 {artifact['sha256']}")
            if ask("Approve these exact bytes for this channel"):
                session.approve_release(key, required(input("Commissioner/reviewer name: "), "the reviewer name"))
        elif action == "ready":
            show_approved(session, key)
        else:
            session.record_publication(key, input("Public URL from the completed manual publication: ").strip())


def show_approved(session, key):
    artifact = session.approved_artifact(key)
    print(f"Approved bytes verified | {artifact['channel']}\n{artifact['artifact']}\nSHA-256 {artifact['sha256']}")


def manual_lane_console(paths):
    session = pick_session(paths)
    stage = choice(dict(zip(MANUAL_LANES, ("OOTP HTML reports", "StatsPlus enrichment", "Read-only mirror check"))),
                   "Manual lane")
    evidence = input("Completed receipt or manifest path [Enter to defer]: ").strip().strip('"')
    note = required(input("What was verified or remains pending: "), "the lane note")
    session.record_lane(stage, Path(evidence).resolve() if evidence else None, note)


def work(paths, action):
    paths.control.mkdir(parents=True, exist_ok=True)
    try:
        with claim_operation(paths.control) as acquired:
            if not acquired:
                raise ValueError("Another Command Center operation is running. Open the dashboard to read its status.")
            return action()
    finally:
        try:
            refresh(paths)
        except (OSError, ValueError, KeyError) as error:
            print(f"Dashboard refresh needs attention: {error}", file=sys.stderr)


def console(paths):
    while True:
        print("\nABL Production Sessions")
        print("  1. Checkpoint a new session")
        print("  2. Resume a session or its deferred work")
        print("  3. Manage publication approvals and receipts")
        print("  4. Record or defer a manual lane")
        print("  5. Refresh the Command Center")
        print("  0. Return to Production Control")
        selection = input("Select: ").strip()
        try:
            if selection == "0":
                return 0
            if selection == "1":
                work(paths, lambda: start(paths))
            elif selection == "2":
                work(paths, lambda: pick_session(paths).run(resumed=True))
            elif selection == "3":
                work(paths, lambda: publication_console(paths))
            elif selection == "4":
                work(paths, lambda: manual_lane_console(paths))
            elif selection == "5":
                print(refresh(paths))
            else:
                print("Select a listed operation.")
        except (OSError, ValueError, KeyError) as error:
            print(error, file=sys.stderr)


def preview(paths):
    print("Session Preview")
    for title, folder in (("OOTP saved league", paths.save), ("OOTP CSV export", paths.live_export),
                          ("Rolling league backup volume", Path(paths.backup.anchor))):
        print(f"{title} | {'Available' if folder.is_dir() else 'Unavailable'} | {folder}")
    for name in ("ABL-Production-Control.ps1", "Backup ABL League.ps1", "Update the Repo.ps1",
                 "Backup SBV Videos.ps1", "Verify H to I Mirror.ps1"):
        print(f"{name} | {'Found' if (paths.scripts / name).is_file() else 'Missing'}")
    print("Quick Play protects the save and closes the checkpoint. Standard adds optional CSV, Desk and review.")
    print("Featured also offers Video Backup Preview and Update. The mirror and external publication are manual.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("open", "refresh", "preview", "console", "sessions"):
        commands.add_parser(name)
    new = commands.add_parser("start", help="Checkpoint a played and saved OOTP session")
    new.add_argument("--mode", choices=MODES)
    new.add_argument("--trigger", choices=TRIGGERS)
    new.add_argument("--date", help="Date shown by OOTP after saving")
    new.add_argument("--window", help="Playing window or milestone")
    for name in ("resume", "lane", "queue", "approve", "ready", "receipt"):
        commands.add_parser(name).add_argument("session")
    lane = commands.choices["lane"]
    lane.add_argument("stage", choices=MANUAL_LANES)
    lane.add_argument("--evidence", type=Path, help="Finished lane receipt or manifest. Omit to defer.")
    lane.add_argument("--note", required=True, help="What was verified, or what remains pending")
    queue = commands.choices["queue"]
    queue.add_argument("lane", choices=RELEASE_LANES)
    queue.add_argument("--channel", required=True)
    queue.add_argument("--title", required=True)
    queue.add_argument("--artifact", required=True, type=Path)
    for name, flag in (("approve", "--reviewer"), ("receipt", "--url")):
        commands.choices[name].add_argument("release")
        commands.choices[name].add_argument(flag, required=True)
    commands.choices["ready"].add_argument("release")
    return parser.parse_args()


def session_action(paths, args):
    session = pick_session(paths, args.session)
    if args.command == "resume":
        return session.run(resumed=True)
    if args.command == "lane":
        session.record_lane(args.stage, args.evidence.resolve() if args.evidence else None,
                            required(args.note, "what was checked"))
    elif args.command == "queue":
        key = session.queue_release(args.lane, required(args.channel, "the channel"),
                                    required(args.title, "the title"), args.artifact.resolve())
        print(f"Release {key} requires commissioner approval.")
    elif args.command == "approve":
        session.approve_release(args.release, required(args.reviewer, "the reviewer name"))
    elif args.command == "ready":
        show_approved(session, args.release)
    elif args.command == "receipt":
        session.record_publication(args.release, args.url.strip())
    return True


def main():
    args = parse_args()
    paths = Paths.local()
    try:
        if args.command in {"open", "refresh"}:
            dashboard = refresh(paths)
            print(dashboard)
            if args.command == "open":
                os.startfile(dashboard)
        elif args.command == "preview":
            preview(paths)
        elif args.command == "console":
            return console(paths)
        elif args.command == "sessions":
            list_sessions(paths)
        elif args.command == "start":
            return 0 if work(paths, lambda: start(paths, args)) else 1
        else:
            return 0 if work(paths, lambda: session_action(paths, args)) else 1
    except (OSError, ValueError, KeyError, EOFError) as error:
        print(error, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Saved receipts and pending stages remain available on the dashboard.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
