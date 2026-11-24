"""Operate the local and public Experiencing Action Baseball editions."""

import argparse
import json
import os
from pathlib import Path
import sys

from abl_eab_release import Publisher, catalog_for
from github_pages import GitHubPages, SITE


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


def show_publication(result):
    if result["status"] == "pending":
        print(f"GitHub accepted {result['commit'][:12]}. Pages has not confirmed {result['release']} yet.")
    elif result["status"] == "failed":
        print(result["error"])
    else:
        print(f"GitHub Pages {result['status']} | Finals through {result['cutoff']} | {result['url']}")
    if result.get("capacity_warning"):
        print("EAB is above 80% of the GitHub Pages site size limit.")


def public_status(publisher, pages):
    status(publisher)
    last = None
    if pages.log.is_file():
        last = json.loads(pages.log.read_text(encoding="utf-8").splitlines()[-1])
        print(f"Last deployment attempt | {last['at']} | {last['status']}")
        if last.get("error"):
            print(last["error"])
    entry, _, _ = pages.live()
    print(f"Public edition | {entry['id']} | Finals through {entry['cutoff']} | {SITE}")
    if last and last["status"] in {"submitted", "pending"} and entry["id"] != last["release"]:
        if failure := pages.failed_run(last["commit"]):
            print(failure)


def console(publisher):
    pages = GitHubPages(publisher)
    while True:
        print("\nExperiencing Action Baseball")
        print("  1. Publish latest verified edition")
        print("  2. Preview latest update")
        print("  3. Check local and public publication status")
        print("  4. Restore an earlier local and public edition")
        print("  5. Build deployment ZIP")
        print("  6. Open the public EAB website")
        print("  7. Retry public deployment of the current local edition")
        print("  0. Return to Production Control")
        answer = input("Select: ").strip()
        try:
            if answer == "0":
                return
            if answer == "1":
                update(publisher)
                show_publication(pages.deploy())
            elif answer == "2":
                draft = Publisher(output=publisher.output / "preview", approval_dir=publisher.approvals)
                update(draft)
                open_site(draft)
            elif answer == "3":
                public_status(publisher, pages)
            elif answer == "4":
                status(publisher)
                key = input("Edition ID to restore [Enter to cancel]: ").strip()
                if key:
                    print("Selected " + publisher.rollback(key)["id"])
                    show_publication(pages.deploy(rollback=True, retry=True))
            elif answer == "5":
                print(publisher.package())
            elif answer == "6":
                os.startfile(SITE)
            elif answer == "7":
                show_publication(pages.deploy(retry=True))
            else:
                print("Select a listed operation.")
        except (OSError, ValueError, KeyError) as exc:
            print(exc, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("open", "status", "console", "package", "verify"):
        commands.add_parser(name)
    commands.add_parser("pages-status")
    approval = commands.add_parser("approve-article", help="Record the human approval of an exact sealed EAB article")
    approval.add_argument("manuscript", type=Path)
    approval.add_argument("--sha256", required=True)
    approval.add_argument("--reviewer", required=True)
    deploy = commands.add_parser("deploy", help="Publish the current verified local edition to GitHub Pages")
    deploy.add_argument("--source-id")
    deploy.add_argument("--retry", action="store_true")
    deploy.add_argument("--wait", type=int, default=45)
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
        "pages-status": lambda: public_status(publisher, GitHubPages(publisher)),
        "rollback": lambda: print(publisher.rollback(args.release_id)["id"]),
        "update": lambda: update(publisher, args.source, args.source_id),
    }
    try:
        if args.command == "preview":
            draft = Publisher(output=publisher.output / "preview", approval_dir=publisher.approvals)
            update(draft)
            if args.open:
                open_site(draft)
        elif args.command == "approve-article":
            entry = publisher.approve_article(args.manuscript, args.sha256, args.reviewer)
            print(json.dumps({field: entry[field] for field in ("title", "source_cutoff", "sha256", "approval")}))
        elif args.command == "deploy":
            receipt = GitHubPages(publisher).deploy(expected_source=args.source_id, retry=args.retry, wait=args.wait)
            print(json.dumps(receipt))
            return {"failed": 1, "pending": 2}.get(receipt["status"], 0)
        else:
            actions[args.command]()
    except (OSError, ValueError, KeyError, EOFError) as exc:
        print(f"EAB: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
