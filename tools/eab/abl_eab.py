"""Operate the local Experiencing Action Baseball publication."""

import argparse
import json
import os
from pathlib import Path
import sys

from abl_eab_release import Publisher, catalog_for


def open_site(publisher):
    path = publisher.dist / "index.html"
    if not path.is_file():
        raise ValueError("Publish or preview a validated EAB edition first.")
    print(path)
    os.startfile(path)


def update(publisher, source=None, expected=None):
    item, status = publisher.publish(source, expected=expected)
    result = {
        "status": status, "release": item["id"], "league_date": item["league_date"],
        "cutoff": item["cutoff"], "finals": item["finals"], "site": str(publisher.dist / "index.html"),
    }
    print(json.dumps(result))
    return item


def status(publisher):
    data = catalog_for(publisher.dist)
    if not data["current"]:
        print("No EAB edition has been published locally.")
    for edition in reversed(data["editions"]):
        current = " | Current" if data["current"] == edition["id"] else ""
        print(f"{edition['id']} | Finals through {edition['cutoff']} | {edition['published_at']}{current}")


def console(publisher):
    while True:
        print("\nExperiencing Action Baseball")
        print("  1. Publish latest verified update locally")
        print("  2. Preview latest update")
        print("  3. List local editions")
        print("  4. Restore an earlier local edition")
        print("  5. Build deployment ZIP")
        print("  0. Return to Production Control")
        answer = input("Select: ").strip()
        try:
            if answer == "0":
                return
            if answer == "1":
                update(publisher)
            elif answer == "2":
                draft = Publisher(output=publisher.output / "preview")
                update(draft)
                open_site(draft)
            elif answer == "3":
                status(publisher)
            elif answer == "4":
                status(publisher)
                key = input("Edition ID to restore [Enter to cancel]: ").strip()
                if key:
                    print("Selected " + publisher.rollback(key)["id"])
            elif answer == "5":
                print(publisher.package())
            else:
                print("Select a listed operation.")
        except (OSError, ValueError, KeyError) as exc:
            print(exc, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("open", "status", "console", "package", "verify"):
        commands.add_parser(name)
    apply = commands.add_parser("update", help="Publish a verified edition to the local website")
    apply.add_argument("--source", type=Path)
    apply.add_argument("--source-id")
    preview = commands.add_parser("preview")
    preview.add_argument("--open", action="store_true")
    commands.add_parser("rollback").add_argument("release_id")
    args = parser.parse_args()
    publisher = Publisher()
    actions = {
        "open": lambda: open_site(publisher), "status": lambda: status(publisher),
        "console": lambda: console(publisher), "package": lambda: print(publisher.package()),
        "verify": lambda: print(f"Verified {len(publisher.paths_for_package()[1])} public files"),
        "rollback": lambda: print(publisher.rollback(args.release_id)["id"]),
        "update": lambda: update(publisher, args.source, args.source_id),
    }
    try:
        if args.command == "preview":
            draft = Publisher(output=publisher.output / "preview")
            update(draft)
            if args.open:
                open_site(draft)
        else:
            actions[args.command]()
    except (OSError, ValueError, KeyError, EOFError) as exc:
        print(f"EAB: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
